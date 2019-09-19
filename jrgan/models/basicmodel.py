# basicmodel.py


from keras.models import Sequential, Model
from keras.models import model_from_json, save_model


class BasicModel:

    def __init__(self, input_shape=(64, 64, 3), scale=4):
        print('base model contructor init')
        # input shape
        self._input_shape = input_shape
        # channels
        self._channels = 1
        if len(input_shape) == 2:
            self._channels = 1
            self._input_shape = self._input_shape + (1, )
        else:
            self._channels = input_shape[2]
        # scale
        self._upscale = scale
        # output
        self._output_shape = (
            self._input_shape[0] * self._upscale,
            self._input_shape[1] * self._upscale,
            self._channels
            )
        # self._learning_rate = learning_rate
        self._dis = Model()
        self._gen = Model()
        print('base model contructor finish')
        
    def _build_discriminator(self):
        return Model()

    def _build_generator(self):
        return Model()

    def train(epochs, batch_size=1, sample_interval=50):
        pass


    def test(self):
        pass


    def predict(self):
        pass

    def to_json(self):
        return {'discriminator': self._dis.to_json(), 'generator': self._gen.to_json()}

    def load_gen(self):
        pass

    def load_dis(self):
        pass

    def load_gan(self):
        pass

    def save_weights_dis(self, filename):
        # with open(filename) as w_file:
        self._dis.save_weights(filename)

    def save_weights_gen(self, filename):
        # with open(filename) as w_file:
        self._gen.save_weights(filename)

    def save_weights(self, dis_filename, gen_filename):
        self.save_weights_dis(dis_filename)
        self.save_weights_gen(gen_filename)

    def summary_dis(self, filename=None):
        if filename != None:
            with open(filename, 'w') as fout:
                self._dis.summary(print_fn=lambda line: fout.write(line + '\n'))
        return self._dis.summary()
    
    def summary_gen(self, filename=None):
        if filename != None:
            with open(filename, 'w') as fout:
                self._gen.summary(print_fn=lambda line: fout.write(line + '\n'))
        return self._gen.summary()


