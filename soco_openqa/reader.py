import soco_openqa.helper as helper
from soco_openqa.soco_mrc.mrc_model import MrcModel, MrcRerankerModel
from collections import defaultdict
import numpy as np


class Reader(object):
    def __init__(self, config):
        gpu_request = config.param.n_gpu
        n_gpu = helper.find_device(gpu_request)
        print('number of gpus: {}'.format(n_gpu))

        if config.reader.use_reranker:
            self.reader = MrcRerankerModel('us', n_gpu=n_gpu)
            self.use_reranker = True
            self.rerank_size = config.reader.rerank_size
        else:
            self.reader = MrcModel('us', n_gpu=n_gpu)
            self.use_reranker = False
        self.model_id = config.reader.model_id
        self.thresh = config.param.score_weight

    def predict(self, query, top_passages):
        batch = [{'q': query, 'doc': p['answer']} for p in top_passages]
        preds = self.reader.batch_predict(
                    self.model_id, 
                    batch, 
                    merge_pred=True,
                    stride=128,
                    batch_size=50
                )

        # combine with ranking score
        if not self.use_reranker:
            candidates = defaultdict(list)
            for a_id, a in enumerate(preds):
                if a.get('missing_warning'):
                    continue
                score = self.thresh * (a['score']) + (1 - self.thresh) * (top_passages[a_id]['score'])
                candidates[a['value']].append(score)
            
            candidates = [{'value': k, 'score': np.max(v)} for k, v in candidates.items()]

            answers = sorted(candidates, key=lambda x: x['score'], reverse=True)
        else:
            # first sort by cls_score, then get topn and rank by combined score
            candidates = defaultdict(lambda: defaultdict(list))
            for a_id, a in enumerate(preds):
                if a.get('missing_warning'):
                    continue
                score = self.thresh * (a['score']*a['cls_prob']) + (1 - self.thresh) * (top_passages[a_id]['score'])
                
                candidates[a['value']]['score'].append(score)
                candidates[a['value']]['cls_prob'].append(a['cls_prob'])

            candidates = [{'value': k, 'score': np.max(v['score']), 'cls_prob':np.max(v['cls_prob'])} for k, v in candidates.items()]

            rerank_candidates = sorted(candidates, key=lambda x: x['cls_prob'], reverse=True)[:self.rerank_size]
            answers = sorted(rerank_candidates, key=lambda x: x['score'], reverse=True)

        return answers
