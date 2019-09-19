

from .srgen00 import SRGen00


_list_gen = {
    0: 'SRGEN00'
}


class FactoryGen:

    def getgen(type_gen, input_shape, filters, model=False):
        if isinstance(type_gen, int):
            type_gen = _list_gen[type_gen]
        gen = None
        if type_gen.upper() in ['SRGEN00', '00', '0']:
            gen = SRGen00(input_shape=input_shape, filters=filters)

        if model:
            return gen.get_model()
        else:
            return gen
