import logging
import time

import urllib3
from elasticsearch import Elasticsearch, RequestsHttpConnection
from soco_encoders.model_loaders import EncoderLoader

from soco_openqa.helper import QueryGenerator as QG

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger('elasticsearch')
logger.setLevel(logging.WARNING)

class RankerBases(object):
    def rank(self, query, size=10):
        raise NotImplementedError


class BM25Ranker(RankerBases):
    def __init__(self, config):
        pass
    

class SpartaRanker(RankerBases):
    def __init__(self, config):
        self.index = config.ranker.model.es_index_name
        self.lang = config.data.lang
        self.top_k = config.param.top_k
        self.tokenizers = dict()

        es_url = 'https://elastic:13-socoES@search-new-wiki-qsd6ejfqfwyva7sok6ag32s72u.us-east-2.es.amazonaws.com'

        es = Elasticsearch(
            hosts=[es_url],
            ca_certs=False,
            verify_certs=False,
            connection_class=RequestsHttpConnection
        )
        # print("Create ES client {}".format(es_url))
        self.es = es


    def _get_query(self, query, query_embedded):
        es_query = QG.tscore_search(query, query_embedded)
        es_query['size'] = self.top_k
        es_query['_source'] = {'excludes': ['embedding_vector*', 'term_scores']}
        return es_query

    def _load_tokenizer(self):
        if self.lang not in self.tokenizers:
            if self.lang == 'zh':
                self.tokenizers[self.lang] = EncoderLoader.load_tokenizer('bert-base-chinese-zh_v4-10K')
            elif self.lang == 'en':
                self.tokenizers[self.lang] = EncoderLoader.load_tokenizer('bert-base-uncased')
            else:
                raise NotImplementedError

        return self.tokenizers[self.lang]


    def _postprocess(self, res):
        res = [{'score': p['_score'],
                'answer': p['_source']['answer']['context']}
                for p in res['hits']['hits']]

        return res


    def rank(self, query):
        """
        search inside
        """
        tokenizer = self._load_tokenizer()
        tokens = tokenizer.tokenize(query, mode='all')
        es_query = self._get_query(query, tokens)
        res = self.es.search(index=self.index, body=es_query, request_timeout=500)

        return self._postprocess(res)
