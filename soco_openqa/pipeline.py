import json
import logging
from os.path import isfile

from tqdm import tqdm

import soco_openqa.helper as helper
from soco_openqa.ranker import BM25Ranker, SpartaRanker
from soco_openqa.reader import Reader
from soco_openqa.cloud_bucket import CloudBucket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ranker_map = {'sparta': SpartaRanker, 'bm25': BM25Ranker}

class OpenQA(object):
    def __init__(self, config):
        self.use_cached_rank = True if config.ranker.use_cached else False
        self.top_k = config.param.top_k
        self.ranker = self._load_ranker(config)
        self.reader = Reader(config)
        self.cloud_bucket = CloudBucket('us')

    def _load_ranker(self, config):
        if self.use_cached_rank:
            logger.info('Use cached ranker file')
            data_name = config.data.name
            ranker_file = config.ranker.cached_ranker_file
            ranker = helper.load_json(file_dir=data_name, file_name=ranker_file)

        else:
            ranker_name = config.ranker.model.name
            ranker = ranker_map[ranker_name](config)

        return ranker


    def predict(self, data):
        predictions = dict()
        no_ans_cnt = 0
        logger.info("Execution started")
        for d in tqdm(data['data']):
            for p in tqdm(d['paragraphs']):
                for qa in p['qas']:
                    if len(predictions) > 0 and len(predictions) % 1000 == 0:
                        logger.info("{}: {} no answer".format(len(predictions), no_ans_cnt))
                    _id = str(qa['id'])
                    query = qa['question']
                    if self.use_cached_rank:
                        top_passages = self.ranker[_id][:self.top_k]
                    else:
                        top_passages = self.ranker.rank(query)

                    try:
                        answers = self.reader.predict(query, top_passages)
                        if len(answers) > 0:
                            predictions[_id] = {'answer': answers[0]['value'], 'passages': top_passages}
                        else:
                            raise ValueError("No Answer")
                    except ValueError as e:
                        no_ans_cnt += 1
                        predictions[_id] = {'answer': 'NO_ANSWER', 'passages': top_passages}

        return predictions