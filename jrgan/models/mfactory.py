# factory


from .srmodel00 import SRModel00

_models = {
        0: 'SRMODEL00'
        }


class MFactory:

    def getModel(type_model, input_shape, upscale, learning_rate, path_dataset,
                 filters_gen=32, filters_dis=32):
        if isinstance(type_model, int):
            print('converting int to key...')
            type_model = _models[type_model]
            print(type_model)
        if type_model in ['SRMODEL00', '00', '0']:
            return SRModel00(input_shape, upscale, learning_rate, path_dataset,
                             filters_gen, filters_gen)
        raise Exception('INVALID INPUT')
