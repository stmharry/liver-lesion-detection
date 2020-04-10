import functools
import tensorflow as tf

from absl import logging
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU


BatchNormalization = functools.partial(
    BatchNormalization,
    axis=-1,
    fused=True)

Conv3D = functools.partial(
    Conv3D,
    padding='SAME',
    use_bias=False)

Conv3DTranspose = functools.partial(
    Conv3DTranspose,
    padding='SAME',
    use_bias=False)


def central_crop3d(x, shape):
    crop_shape = tf.shape(x)[1:4] - shape
    crop_begin = (crop_shape + 1) // 2
    crop_end = shape + crop_begin
    slices = (
        [slice(None)]
        + [slice(*args) for args in zip(
            tf.unstack(crop_begin), tf.unstack(crop_end))]
        + [slice(None)])

    return x[slices]


class BasicBlock(tf.keras.Model):
    def __init__(self,
                 filters,
                 strides,
                 skip_conv,
                 isotropic=True,
                 conv_cls=Conv3D,
                 name='basic_block'):

        super(BasicBlock, self).__init__()

        self.strides = strides
        self.skip_conv = skip_conv
        self._name = name

        if isotropic:
            strides = [strides, strides, strides]
        else:
            strides = [strides, strides, 1]

        self.bn1 = BatchNormalization()
        self.relu = ReLU()

        if self.skip_conv:
            self.conv = conv_cls(
                filters=filters, kernel_size=1, strides=strides)

        self.conv1 = conv_cls(filters=filters, kernel_size=3, strides=strides)

        self.bn2 = BatchNormalization()
        self.conv2 = Conv3D(filters=filters, kernel_size=3)

    def call(self, x, training):
        _x = x
        x = self.bn1(x, training=training)
        x = self.relu(x)

        if self.skip_conv:
            _x = self.conv(x)

        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)

        logging.debug(f'{self.name}: shape = {x.shape}')
        return x + _x


class BottleneckBlock(tf.keras.Model):
    def __init__(self,
                 filters,
                 strides,
                 skip_conv,
                 isotropic=True,
                 conv_cls=Conv3D,
                 name='bottleneck_block'):

        super(BottleneckBlock, self).__init__()

        self.skip_conv = skip_conv
        self._name = name

        if isotropic:
            strides = [strides, strides, strides]
        else:
            strides = [strides, strides, 1]

        self.bn1 = BatchNormalization()
        self.relu = ReLU()

        if self.skip_conv:
            self.conv = conv_cls(
                filters=4 * filters, kernel_size=1, strides=strides)

        self.conv1 = Conv3D(filters=filters, kernel_size=1, strides=1)

        self.bn2 = BatchNormalization()
        self.conv2 = conv_cls(filters=filters, kernel_size=3, strides=strides)

        self.bn3 = BatchNormalization()
        self.conv3 = Conv3D(filters=4 * filters, kernel_size=1, strides=1)

    def call(self, x, training):
        _x = x
        x = self.bn1(x, training=training)
        x = self.relu(x)

        if self.skip_conv:
            _x = self.conv(x)

        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.bn3(x, training=training)
        x = self.relu(x)
        x = self.conv3(x)

        logging.debug(f'{self.name}: shape = {x.shape}')
        return x + _x


class DownLayer(tf.keras.Model):
    def __init__(self,
                 num_blocks,
                 filters,
                 strides,
                 isotropic,
                 block_cls,
                 name='block_layer'):

        super(DownLayer, self).__init__()

        self._name = name

        self._models = [
            block_cls(
                filters=filters,
                strides=(1 if num_block else strides),
                skip_conv=(num_block == 0),
                isotropic=isotropic,
                conv_cls=Conv3D,
                name=f'block{num_block}')
            for num_block in range(num_blocks)]

    def call(self, x, training):
        for _model in self._models:
            x = _model(x, training=training)

        logging.debug(f'{self.name}: shape = {x.shape}')
        return x


class UpLayer(tf.keras.Model):
    def __init__(self,
                 num_blocks,
                 filters,
                 strides,
                 isotropic,
                 block_cls,
                 name='block_layer'):

        super(UpLayer, self).__init__()

        self._name = name

        self._models = [
            block_cls(
                filters=filters,
                strides=(1 if num_block else strides),
                skip_conv=(num_block == 0),
                isotropic=isotropic,
                conv_cls=(Conv3D if num_block else Conv3DTranspose),
                name=f'block{num_block}')
            for num_block in range(num_blocks)]

    def call(self, x, training, _x=None):
        for (num_block, _model) in enumerate(self._models):
            x = _model(x, training=training)

            if (_x is not None) and (num_block == 0):
                x = central_crop3d(x, shape=tf.shape(_x)[1:4])
                x = tf.concat([x, _x], axis=-1)

        logging.debug(f'{self.name}: shape = {x.shape}')
        return x


class _ResNet(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 block_sizes,
                 filters=64,
                 isotropic=True,
                 block_cls=BasicBlock):

        super(_ResNet, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(block_sizes)

        self.conv1 = Conv3D(filters=64, kernel_size=7, strides=2)
        self.conv1._name = 'conv1'
        self.max_pool = MaxPool3D(pool_size=3, strides=2, padding='SAME')

        self._down_models = [
            DownLayer(
                num_blocks=block_sizes[num_layer],
                filters=(filters * 2 ** num_layer),
                strides=(2 if num_layer else 1),
                isotropic=isotropic,
                block_cls=block_cls,
                name=f'conv{num_layer + 2}')
            for num_layer in range(self.num_layers)]


class ResUNet(_ResNet):
    def __init__(self,
                 num_classes,
                 block_sizes,
                 filters=64,
                 isotropic=True,
                 block_cls=BasicBlock):

        super(ResUNet, self).__init__(
            num_classes=num_classes,
            block_sizes=block_sizes,
            filters=filters,
            isotropic=isotropic,
            block_cls=block_cls,
        )

        self._up_models = (
            [
                UpLayer(
                    num_blocks=1,
                    filters=(filters * 2 ** num_layer),
                    strides=2,
                    isotropic=isotropic,
                    block_cls=block_cls,
                    name=f'upconv{num_layer + 2}')
                for num_layer in reversed(range(self.num_layers - 1))]
            + [
                UpLayer(
                    num_blocks=1,
                    filters=filters,
                    strides=2,
                    isotropic=True,
                    block_cls=BasicBlock,
                    name=f'upconv1')])

        self.bn0 = BatchNormalization()
        self.relu0 = ReLU()
        self.upconv0 = Conv3DTranspose(
            filters=num_classes, kernel_size=3, strides=2, use_bias=True)

    def call(self, x, training):
        shape = tf.shape(x)[1:4]

        x2 = self.conv1(x)
        x4 = self.max_pool(x2)

        xs = [x2]
        x = x4
        for _model in self._down_models:
            x = _model(x, training=training)
            xs.append(x)

        x = xs.pop(-1)
        for (_model, _x) in zip(self._up_models, reversed(xs)):
            x = _model(x, _x=_x, training=training)

        x = self.bn0(x, training=training)
        x = self.relu0(x)
        x = self.upconv0(x)
        x = central_crop3d(x, shape=shape)

        return x


ResUNet18 = functools.partial(
    ResUNet, block_sizes=[2, 2, 2, 2], block_cls=BasicBlock)
ResUNet34 = functools.partial(
    ResUNet, block_sizes=[3, 4, 6, 3], block_cls=BasicBlock)
ResUNet50 = functools.partial(
    ResUNet, block_sizes=[3, 4, 6, 3], block_cls=BottleneckBlock)
