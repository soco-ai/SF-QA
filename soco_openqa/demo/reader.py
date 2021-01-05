import numpy as np
from soco_openqa.soco_mrc.mrc_model import MrcModel
from collections import defaultdict



class Reader:
    def __init__(self, model):
        self.model_id = model
        self.reader = MrcModel('us', n_gpu=1)
        self.thresh = 0.8
    
    def predict(self, query, top_passages):
        batch = [{'q': query, 'doc': p['answer']} for p in top_passages]
        preds = self.reader.batch_predict(
                    self.model_id,
                    batch, 
                    merge_pred=True,
                    stride=128,
                    batch_size=50
                )

        candidates = defaultdict(list)
        for a_id, a in enumerate(preds):
            if a.get('missing_warning'):
                continue
            score = self.thresh * (a['score']) + (1 - self.thresh) * (top_passages[a_id]['score'])
            candidates[a['value']].append({'combined_score': score, 
                                        'reader_score':a['score'], 
                                        'ranker_score':top_passages[a_id]['score'], 
                                        'idx': a_id, 
                                        'prob': a['prob'], 
                                        'answer_span': a['answer_span']})

        # get best passages with best answer
        answers = []
        for k, v in candidates.items():
            combined_scores = [x['combined_score'] for x in v]
            reader_scores = [x['reader_score'] for x in v]
            ranker_scores = [x['ranker_score'] for x in v]
            idxes = [x['idx'] for x in v]
            best_idx = int(np.argmax(combined_scores))
            best_a_id = idxes[best_idx]
            answers.append({'value': k,
                            'score': combined_scores[best_idx],
                            'reader_score': reader_scores[best_idx],
                            'ranker_score': ranker_scores[best_idx],
                            'prob': v[best_idx]['prob'],
                            'answer_span': v[best_idx]['answer_span'],
                            "source": {
                                'context': top_passages[best_a_id]['answer'],
                                'url': top_passages[best_a_id].get('meta', {}).get('url'),
                                'doc_id': top_passages[best_a_id].get('meta', {}).get('doc_id')
                            }
                           })

        answers = sorted(answers, key=lambda x: x['score'], reverse=True)
        return answers
    
    