from keras.layers import Dense
# from keras.layers import Input
# from keras.engine.input_layer import Input
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
# from keras.layers.convolutional import Conv2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.models import Sequential

from .basicdis import BasicDis


class SRDis00(BasicDis):
    """
    def __init__(self, output_shape, filters=32):
        # Input img
        self._df = filters
        self._output_shape = output_shape
        print('d0')
        d0 = Input(shape=self._output_shape)
        # hidden
        print('d1')
        d1 = self._d_block(d0, self._df, bn=False)
        print('d2')
        d2 = self._d_block(d1, self._df, strides=2)
        d3 = self._d_block(d2, self._df*2)
        d4 = self._d_block(d3, self._df*2, strides=2)
        d5 = self._d_block(d4, self._df*4)
        d6 = self._d_block(d5, self._df*4, strides=2)
        d7 = self._d_block(d6, self._df*8)
        d8 = self._d_block(d7, self._df*8, strides=2)
        # final dense
        d9 = Dense(self._df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)
        # create model
        self._model = Model(d0, validity)

    def _d_block(layer_input, filters, strides=1, bn=True):
        # Discriminator layer
        d = Conv2D(filters, kernel_size=3, strides=strides,
                   padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d
    """

    def __init__(self, output_shape, filters=32):
        self._df = filters
        self._output_shape = output_shape
        self._model = Sequential()
        self._hidden_block(self._df, strides=2)
        self._hidden_block(self._df)
        self._hidden_block(self._df, strides=2)
        self._hidden_block(self._df)
        self._hidden_block(self._df, strides=2)
        self._hidden_block(self._df)
        self._hidden_block(self._df, strides=2)
        self._model.add(Dense(self._df * 16))
        self._model.add(LeakyReLU(alpha=0.2))
        self._model.add(Dense(1, activation='sigmoid'))

    def _hidden_block(self, filters, strides=1, bn=True):
        layer = Conv2D(filters, kernel_size=(3, 3), strides=strides,
                       padding='same')
        self._model.add(layer)
        self._model.add(BatchNormalization(momentum=0.8))
