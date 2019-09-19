
from .srdis00 import SRDis00

_list_type = {
    0: 'SRDIS00'
}


class FactoryDis:

    def getdis(type_dis, output_shape, filters, model=False):
        if isinstance(type_dis, int):
            type_dis = _list_type[type_dis]
        srdis = None
        # factory
        if type_dis.upper() in ['SRDIS00', '00', '0']:
            srdis = SRDis00(output_shape, filters)
        # return object or model
        if model:
            return srdis.get_model()
        else:
            return srdis

        # if type_dis in ['srdis']

