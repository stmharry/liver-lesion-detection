import abc
import enum
import functools
import nibabel as nib
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

from detection.models import ResUNet18
from detection.ops import resample_to_output
from detection.ops import resample_from_to
from detection.ops import resize_to_output
from detection.ops import resize_from_to
from detection.ops import largest_connected
from detection.utils import on_cpu
from detection.utils import with_output

flags.DEFINE_string(
    'test_dir', './images', 'Directory containing nifti files for testing.')
flags.DEFINE_string(
    'output_dir', './results', 'Root directory to write outputs.')
FLAGS = flags.FLAGS


class Label(enum.Enum):
    BACKGROUND = 0
    LIVER = 1
    LESION = 2


class Pipeline(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def output_path(self, study_name):
        pass

    def __init__(self, batch_size, model_cls, checkpoint_path):
        self.batch_size = batch_size
        self.model = model_cls()
        self.checkpoint_path = checkpoint_path

    def _normalize_image(self, image, q=99):
        image_max = tf.contrib.distributions.percentile(
            image, q=q, axis=[0, 1, 2], keep_dims=True)
        image = tf.clip_by_value(image / image_max, 0, 1) - 0.5

        return image

    @abc.abstractmethod
    def _input_generator(self):
        pass

    @abc.abstractmethod
    def _preprocess(self, example):
        pass

    @abc.abstractmethod
    def _postprocess(self, example):
        pass

    def input_fn(self):
        study_paths = [
            os.path.join(FLAGS.test_dir, study_path)
            for study_path in os.listdir(FLAGS.test_dir)
            if '.nii' in study_path]

        dataset = tf.data.Dataset.from_generator(
            functools.partial(
                self._input_generator, study_paths=study_paths),
            output_types=self._input_generator.output_types,
            output_shapes=self._input_generator.output_shapes)
        dataset = dataset.map(self._preprocess)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def model_fn(self, batch):
        _mask_logit = self.model(batch['image'], training=False)

        batch.update({'_mask_logit': _mask_logit})
        batch.update(tf.map_fn(
            self._postprocess, batch, dtype=self._postprocess.output_types))

        return batch

    @on_cpu
    def _predict_generator(self):
        logging.info(f'{self.__class__.__name__} making predictions...')

        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(self.checkpoint_path)

        for batch in self.input_fn():
            batch = self.model_fn(batch)

            for num in range(self.batch_size):
                yield {
                    key: value[num].numpy()
                    for (key, value) in batch.items()}

    def predict(self):
        predictions = self._predict_generator()

        df = pd.DataFrame(predictions)
        for (_, item) in df.iterrows():
            study_name = item.study_name.decode()
            mask_nib = nib.Nifti1Image(
                item._mask.astype(np.uint8), affine=item.affine_raw)
            mask_path = self.output_path(study_name)

            logging.info(f'Saving study <{study_name}> ({mask_path})')
            nib.save(mask_nib, mask_path)


class LiverDetection(Pipeline):
    @staticmethod
    def output_path(study_name):
        return os.path.join(FLAGS.output_dir, f'{study_name}.liver.nii.gz')

    def __init__(self,
                 batch_size=1,
                 model_cls=functools.partial(ResUNet18, num_classes=2),
                 checkpoint_path='./checkpoints/ckpt-liver',
                 voxel_dims=[4, 4, 4]):

        super(LiverDetection, self).__init__(
            batch_size=batch_size,
            model_cls=model_cls,
            checkpoint_path=checkpoint_path)

        self.voxel_dims = voxel_dims

    @with_output('study_name', dtype=tf.string, shape=[])
    @with_output('study_path', dtype=tf.string, shape=[])
    @with_output('image_raw', dtype=tf.float32, shape=[None, None, None, 4])
    @with_output('affine_raw', dtype=tf.float32, shape=[4, 4])
    def _input_generator(self, study_paths):
        for study_path in study_paths:
            study_name = os.path.basename(study_path).split('.')[0]

            logging.info(f'Loading study <{study_name}> ({study_path})')

            image_nib = nib.load(study_path)
            image = image_nib.get_fdata().astype(np.float32)
            affine = image_nib.affine

            yield {
                'study_name': study_name,
                'study_path': study_path,
                'image_raw': image,
                'affine_raw': affine}

    def _preprocess(self, example):
        image = self._normalize_image(example['image_raw'])
        (image, affine) = resample_to_output(
            image=image,
            affine=example['affine_raw'],
            voxel_dims=self.voxel_dims)

        example.update({'image': image, 'affine': affine})
        return example

    @with_output('_mask_logit', dtype=tf.float32)
    @with_output('_mask_prob', dtype=tf.float32)
    @with_output('_mask', dtype=tf.int32)
    def _postprocess(self, example):
        _mask_logit = resample_from_to(
            image=example['_mask_logit'],
            affine=example['affine'],
            to_shape=tf.shape(example['image_raw']),
            to_affine=example['affine_raw'],
            order=1)

        _mask_prob = tf.nn.softmax(_mask_logit, axis=-1)
        _mask = tf.argmax(_mask_logit, axis=-1)
        _mask = largest_connected(_mask, index=[Label.LIVER.value])

        return {
            '_mask_logit': _mask_logit,
            '_mask_prob': _mask_prob,
            '_mask': _mask}


class LesionDetection(Pipeline):
    @staticmethod
    def output_path(study_name):
        return os.path.join(FLAGS.output_dir, f'{study_name}.lesion.nii.gz')

    def __init__(self,
                 batch_size=1,
                 model_cls=functools.partial(ResUNet18, num_classes=3),
                 checkpoint_path='./checkpoints/ckpt-lesion'):

        super(LesionDetection, self).__init__(
            batch_size=batch_size,
            model_cls=model_cls,
            checkpoint_path=checkpoint_path)

    @with_output('study_name', dtype=tf.string, shape=[])
    @with_output('study_path', dtype=tf.string, shape=[])
    @with_output('image_raw', dtype=tf.float32, shape=[None, None, None, 4])
    @with_output('roi_raw', dtype=tf.int32, shape=[None, None, None])
    @with_output('affine_raw', dtype=tf.float32, shape=[4, 4])
    def _input_generator(self, study_paths):
        for study_path in study_paths:
            study_name = os.path.basename(study_path).split('.')[0]

            logging.info(f'Loading study <{study_name}> ({study_path})')

            image_nib = nib.load(study_path)
            image = image_nib.get_fdata().astype(np.float32)
            affine = image_nib.affine

            roi_path = LiverDetection.output_path(study_name)
            logging.info(f'Loading roi <{study_name}> ({roi_path})')

            roi_nib = nib.load(roi_path)
            roi = roi_nib.get_fdata().astype(np.int32)

            yield {
                'study_name': study_name,
                'study_path': study_path,
                'image_raw': image,
                'roi_raw': roi,
                'affine_raw': affine}

    def _preprocess(self, example):
        image = resize_to_output(
            image=example['image_raw'], roi=example['roi_raw'])
        image = self._normalize_image(image)

        example.update({'image': image})
        return example

    @with_output('_mask_logit', dtype=tf.float32)
    @with_output('_mask_prob', dtype=tf.float32)
    @with_output('_mask', dtype=tf.int32)
    def _postprocess(self, example):
        _mask_logit = example['_mask_logit']
        _mask_prob = tf.nn.softmax(_mask_logit, axis=-1)

        _mask_liver = tf.cast(tf.round(
            _mask_prob[..., Label.LIVER.value]
            + _mask_prob[..., Label.LESION.value]), dtype=tf.int32)
        _mask_lesion = tf.cast(tf.round(
            _mask_prob[..., Label.LESION.value]), dtype=tf.int32)
        _mask_largest = largest_connected(_mask_liver, index=[1])

        _mask = _mask_largest * (
            _mask_lesion * Label.LESION.value
            + (1 - _mask_lesion) * _mask_liver * Label.LIVER.value)

        _mask = resize_from_to(image=_mask, roi=example['roi_raw'])

        return {
            '_mask_logit': _mask_logit,
            '_mask_prob': _mask_prob,
            '_mask': _mask}


def main(_):
    logging.set_verbosity(logging.INFO)
    tf.enable_eager_execution()

    os.path.isdir(FLAGS.output_dir) or os.makedirs(FLAGS.output_dir)

    LiverDetection().predict()
    LesionDetection().predict()


if __name__ == '__main__':
    app.run(main)
