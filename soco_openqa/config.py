import os
from yacs.config import CfgNode as CN
import soco_openqa.helper as helper

_C = CN()

_C.config_name = None

# dataset 
_C.data = CN()
_C.data.lang = 'en'
_C.data.name = None
_C.data.split = None

# ranker
_C.ranker = CN()
_C.ranker.use_cached = True
_C.ranker.cached_ranker_file = None


## online ranker if use_cached = False
_C.ranker.model = CN()
_C.ranker.model.name = None
_C.ranker.model.es_index_name = None


# reader
_C.reader = CN()
_C.reader.model_id = None
_C.reader.use_reranker = False
_C.reader.rerank_size = None

# experiment setting
_C.param = CN()
_C.param.score_weight = 0.8
_C.param.top_k = 50
_C.param.n_gpu = 2


def get_config(config_file):

    config = _C.clone()
    config.merge_from_file(config_file)

    # add config file name for saving log
    config_name = helper.get_name_from_path(config_file)
    config.merge_from_list(['config_name', config_name])
    config.freeze()

    return config
