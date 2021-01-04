import datetime
import json
import logging
import os
from soco_device import DeviceCheck
from soco_openqa.cloud_bucket import CloudBucket


logger = logging.getLogger(__name__)


def get_name_from_path(path):
    def strip_path(path):
        return path.rsplit('/', 1)[-1]

    def remove_ext(name):
        return name.rsplit('.', 1)[0]

    return remove_ext(strip_path(path))


def find_device(n_gpu):
    '''
    return available device number given requested number
    '''
    device_check = DeviceCheck()
    device_name, device_ids = device_check.get_device(n_gpu=n_gpu)
    n_gpu = len(device_ids)

    return n_gpu


def load_json(file_dir, file_name, region='us'):
    cloud_bucket = CloudBucket(region)
    cloud_bucket.download_file(file_dir=file_dir, file_name=file_name)
    res = json.load(open(os.path.join('data', file_dir, file_name)))
    return res


def save_logs(config, results, save_path='reports', save_name='eval_results'):
    os.makedirs(save_path, exist_ok=True)
    save_name = '{}/{}.txt'.format(save_path, save_name)
    results = '\n' + ' '.join(['{}={:.3f}'.format(k, v) for k, v in results.items()]) + '\n\n'
    
    curr_time = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')
    with open(save_name, 'a') as f:
        f.write('{}\n'.format('*'*20))
        f.write('{}\n'.format(curr_time))
        f.write(config)
        f.write(results)
    logger.info('Saved results to {}'.format(save_name))


def load_jsonl(file_name):
    data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))

    logger.info('Loaded {} lines from {}'.format(len(data), file_name))
    return data


def sparta_postprocess(ranker_id):
    data = load_jsonl('./data/{}.jsonl'.format(ranker_id))
    
    # convert a list of dict to {id1: [{‘answer’:1, ‘score’:1}, {‘answer’:2, ‘score’:2}, …], id2: [], …}
    id_to_data_map = dict()
    for d in data:
        if d['id'] not in id_to_data_map:
            id_to_data_map[d['id']] = []
            for response in d['responses']:
                res_map = dict()
                res_map['score'] = response['_score']
                res_map['answer'] = response['_source']['context'].get('context', '')
                id_to_data_map[d['id']].append(res_map)
        else:
            raise ValueError('{} already in map'.format(d['id']))
    
    # save processed version to json
    json.dump(id_to_data_map, open('./data/{}.json'.format(ranker_id), 'w'), ensure_ascii=False)

    return id_to_data_map


class QueryGenerator(object):
    @classmethod
    def sent_bm25(cls, query):
        es_query = {"query": {'match': {'q': {'query': query}}}}
        return es_query

    @classmethod
    def context_bm25(cls, query):
        es_query = {"query":  {'match': {'context.context': {'query': query}}}}
        return es_query

    @classmethod
    def tscore_search(cls, query, query_embedded, alpha_bm25=0.0, max_l2r=-1):
        # convert to string only
        query_embedded = [t if type(t) is str else '__'.join(t) for t in query_embedded if t]

        main_query = [{'rank_feature': {'field': 'term_scores.{}'.format(t),
                                        "log": {"scaling_factor": 1.0}
                                        }} for t in query_embedded]
        es_query = {"query": {"bool": {"should": main_query}}}

        if alpha_bm25 > 0:
            window_size = max(100, max_l2r)
            es_query['rescore'] = {
                "window_size": window_size,
                "query": {
                    "score_mode": "total",
                    "rescore_query": {
                        "match": {"q": {"query": query}},
                    },
                    "query_weight": 1.0,
                    "rescore_query_weight": alpha_bm25
                }}

        return es_query

