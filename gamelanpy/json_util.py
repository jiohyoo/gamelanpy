import base64
import json
import numpy as np
from sparse_gmm import SparseGMM
from nonparanormal import NPNTransformer

class GamelanPyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SparseGMM):
            dct = obj.__dict__
            dct['__sparse_gmm__'] = True
            return dct
        if isinstance(obj, NPNTransformer):
            print 'hello'
            dct = obj.__dict__
            dct['__npn_transformer__'] = True
            return dct
        if isinstance(obj, np.ndarray):
            data_base64 = base64.b64encode(obj.data)
            return dict(__ndarray__=data_base64, dtype=str(obj.dtype), shape=obj.shape)

        return json.JSONEncoder(self, obj)

def gamelan_json_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    if isinstance(dct, dict) and '__sparse_gmm__' in dct:
        sgmm = SparseGMM()
        sgmm.__dict__ = dct
        return sgmm
    if isinstance(dct, dict) and '__npn_transformer__' in dct:
        npn = NPNTransformer()
        npn.__dict__ = dct
        return npn
    return dct
