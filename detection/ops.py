import nibabel as nib
import nibabel.processing as nib_processing
import numpy as np
import skimage.measure
import tensorflow as tf


def on_cpu(f):
    def _f(*args, **kwargs):
        with tf.device('CPU'):
            return f(*args, **kwargs)
    return _f


def resample_to_output(image, affine, voxel_dims, order=3):
    def _vox2out_vox_numpy(shape, affine):
        _shape = shape[:-1]
        _channel = shape[-1]

        (_to_shape, to_affine) = nib_processing.vox2out_vox(
            (_shape, affine), voxel_dims)
        to_shape = np.r_[_to_shape, _channel]

        return (
            to_shape.astype(np.int32),
            to_affine.astype(np.float32))

    (to_shape, to_affine) = tf.py_func(
        _vox2out_vox_numpy,
        [tf.shape(image), affine],
        Tout=[tf.int32, tf.float32])

    to_shape.set_shape([4])
    to_affine.set_shape([4, 4])

    return (
        resample_from_to(image, affine, to_shape, to_affine, order=order),
        to_affine)


def resample_from_to(image, affine, to_shape, to_affine, order=3):
    def _resample_from_to_numpy(image, affine, to_shape, to_affine):
        _to_shape = to_shape[:-1]
        _channel = image.shape[-1]
        to_shape = np.r_[_to_shape, _channel]

        image_nib = nib.Nifti1Image(image, affine=affine)
        to_image_nib = nib_processing.resample_from_to(
            image_nib, (to_shape, to_affine), order=order)

        return to_image_nib.get_fdata().astype(np.float32)

    to_image = tf.py_func(
        _resample_from_to_numpy,
        [image, affine, to_shape, to_affine],
        Tout=tf.float32)

    if image.shape.ndims == 4:
        to_image.set_shape([None, None, None, image.shape[3]])
    else:
        to_image.set_shape([None, None, None])

    return to_image


def resize_to_output(image, roi):
    roi_bin = tf.cast(roi, tf.bool)

    axes = list(range(roi.shape.ndims))
    slices = []
    for axis in range(image.shape.ndims):
        if axis < roi.shape.ndims:
            _axes = axes.copy()
            _axes.remove(axis)

            ind = tf.where(tf.reduce_any(roi_bin, axis=_axes))
            ind = tf.cast(ind, tf.int32)

            ind_min = tf.reduce_min(ind)
            ind_max = tf.reduce_max(ind)

            slices.append([ind_min, ind_max - ind_min + 1])
        else:
            slices.append([0, tf.shape(image)[axis]])

    (begin, size) = zip(*slices)
    return tf.slice(image, begin=tf.stack(begin), size=tf.stack(size))


def resize_from_to(image, roi):
    roi_bin = tf.cast(roi, tf.bool)

    axes = list(range(roi.shape.ndims))
    paddings = []
    for axis in range(image.shape.ndims):
        if axis < roi.shape.ndims:
            _axes = axes.copy()
            _axes.remove(axis)

            ind = tf.where(tf.reduce_any(roi_bin, axis=_axes))
            ind = tf.cast(ind, tf.int32)

            ind_min = tf.reduce_min(ind)
            ind_max = tf.reduce_max(ind)

            paddings.append([ind_min, tf.shape(roi)[axis] - ind_max - 1])
        else:
            paddings.append([0, 0])

    return tf.pad(image, paddings=paddings)


def largest_connected(mask, index, connectivity=3):
    def _largest_connected_numpy(mask):
        mask_onehot = mask[..., None] == index

        mask_bin = []
        for num_index in range(len(index)):
            _mask_index = mask_onehot[..., num_index]
            _mask_label = skimage.measure.label(
                _mask_index, connectivity=connectivity)
            _count = np.bincount(_mask_label.flatten())
            _mask_bin = _mask_label == (np.argmax(_count[1:]) + 1)

            mask_bin.append(_mask_bin)

        mask_bin = np.stack(mask_bin, axis=-1)

        return np.sum(mask_bin * index, axis=-1).astype(np.int32)

    _mask = tf.py_func(
        _largest_connected_numpy,
        [mask],
        Tout=tf.int32)
    _mask.set_shape(mask.shape)
    return _mask
