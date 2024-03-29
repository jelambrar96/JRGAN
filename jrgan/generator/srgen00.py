
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Add
from keras.layers import Input
# from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.models import Model

from .basicgen import BasicGen


class SRGen00(BasicGen):

    def __init__(self, input_shape, filters=32, n_blocks=16):
        self._input_shape = input_shape
        self._gf = filters
        self._channels = self._input_shape[2]
        self.n_residual_blocks = n_blocks
        # Low resolution image input
        img_lr = Input(shape=self._input_shape)
        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)
        # Propogate through residual blocks
        r = self._residual_block(c1, self._gf)
        for _ in range(self.n_residual_blocks - 1):
            r = self._residual_block(r, self._gf)
        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])
        # Upsampling
        u1 = self._deconv2d(c2)
        u2 = self._deconv2d(u1)
        # Generate high resolution output
        gen_hr = Conv2D(self._channels, kernel_size=9, strides=1,
                        padding='same', activation='tanh')(u2)
        self._model = Model(img_lr, gen_hr)

    # create an residual
    def _residual_block(self, layer_input, filters):
        # """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = Activation('relu')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, layer_input])
        return d

    # create an deconv
    def _deconv2d(self, layer_input):
        # """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
        return u
    """
    def _add_residual_block(self, filters):
        # self._model.add()
        # conv2d =
        self._model.add(Conv2D(filters=filters, kernel_size=(3, 3), strides=1,
                        padding='same', activation='relu'))
        self._model.add(BatchNormalization(momentum=0.8))
        self._model.add(Conv2D(filters=filters, kernel_size=(3, 3), strides=1,
                        padding='same'))
        self._model.add(BatchNormalization(momentum=0.8))

    def _add_deconv2d(self):
        self._model.add(UpSampling2D(size=2))
        self._model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    """