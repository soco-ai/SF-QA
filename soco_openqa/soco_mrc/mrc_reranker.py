import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import os
import torch
import re
from collections import namedtuple
import logging
from soco_device import DeviceCheck
from soco_openqa.soco_mrc import util
from soco_openqa.soco_mrc.models.MRCReranker import BertForQuestionAnsweringWithReranker

class MRCRerankerModel(object):
    

    def __init__(self, region, use_gpu=False, max_answer_length=64, fp16=False, quantize=False, multiprocess=False):
        print("Op in {} region".format(region))
        self.use_gpu = use_gpu
        self.fp16 = fp16
        self.quantize = quantize
        self.multiprocess = multiprocess
        self._models = dict()
        self.max_answer_length = max_answer_length
        self.max_input_length = 512
        self.device_check = DeviceCheck()

    def _load_model(self, model_id):
        # a naive check. if too big, just reset
        if len(self._models) > 20:
            self._models = dict()

        if model_id not in self._models:
            path = os.path.join('resources', model_id)
            model_class = self.__class__.__name__
            print('class: {}'.format(model_class))
            model = BertForQuestionAnsweringWithReranker.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

            device = self.device_check.get_device_by_model(model_id) if self.use_gpu else 'cpu'
            print('device: {}'.format(device))

            if self.fp16 and 'cuda' in device:
                print('Use fp16')
                model.half()
            if self.quantize and device == 'cpu':
                print('Use quantization')
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8)
            model.to(device)

            self._models[model_id] = (tokenizer, model, device)

        else:
            # if loaded as cpu, check if gpu is available
            _, _, device = self._models[model_id]
            if self.use_gpu and device == 'cpu':
                new_device = self.device_check.get_device_by_model(model_id)
                if new_device != device:
                    print('Reloading')
                    self._models.pop(model_id)
                    self._load_model(model_id)

        return self._models[model_id]

    def _normalize_text(self, text):
        return re.sub('\s+', ' ', text)

    def cap_to(self, seq, max_len):
        prefix = seq[0:-1][0:max_len - 1]
        return prefix + [seq[-1]]

    def batch_predict(self, model_id, data, n_best_size=1):
        tokenizer, model, device = self._load_model(model_id)

        features = []
        too_long = False
        for d in data:
            doc = self._normalize_text(d['doc'])
            q = self._normalize_text(d['q'])
            temp = tokenizer.encode_plus(q, doc, return_offsets_mapping=True)
            # cut by max_input_len
            input_ids = temp.data['input_ids']
            if len(input_ids) > self.max_input_length:
                too_long = True
                print("Input length {} is too big. Cap to {}".format(len(input_ids), self.max_input_length))
                for k, v in temp.data.items():
                    temp.data[k] = self.cap_to(v, self.max_input_length)

            temp['doc'] = doc
            temp['q'] = q
            features.append(temp)

        results = []
        for batch in util.chunks(features, 10):
            max_len = max([len(f['input_ids']) for f in batch])
            for f in batch:
                f_len = len(f['input_ids'])
                f['length'] = f_len
                f['input_ids'] = f['input_ids'] + [0] * (max_len - f_len)
                f['token_type_ids'] = f['token_type_ids'] + [0] * (max_len - f_len)
                f['attention_mask'] = f['attention_mask'] + [0] * (max_len - f_len)

            input_ids = [f['input_ids'] for f in batch]
            token_type_ids = [f['token_type_ids'] for f in batch]
            attn_masks = [f['attention_mask'] for f in batch]

            with torch.no_grad():
                start_scores, end_scores, cls_scores = model(torch.tensor(input_ids).to(device),
                                                 token_type_ids=torch.tensor(token_type_ids).to(device),
                                                 attention_mask=torch.tensor(attn_masks).to(device))
                start_probs = torch.softmax(start_scores, dim=1)
                end_probs = torch.softmax(end_scores, dim=1)
                cls_probs = torch.softmax(cls_scores, dim=1)

            for b_id in range(len(batch)):
                all_tokens = tokenizer.convert_ids_to_tokens(input_ids[b_id])
                legal_length = batch[b_id]['length']
                b_start_score = start_scores[b_id][0:legal_length]
                b_end_score = end_scores[b_id][0:legal_length]
                token2char = batch[b_id]['offset_mapping']
                for t_id in range(legal_length):
                    if token2char[t_id] is None or token2char[t_id] == (0, 0):
                        b_start_score[t_id] = -10000
                        b_end_score[t_id] = -10000

                _, top_start_id = torch.topk(b_start_score, 2, dim=0)
                _, top_end_id = torch.topk(b_end_score, 2, dim=0)

                s_prob = start_probs[b_id, top_start_id[0]].item()
                e_prob = end_probs[b_id, top_end_id[0]].item()
                s_logit = start_scores[b_id, top_start_id[0]].item()
                e_logit = end_scores[b_id, top_end_id[0]].item()
                # get has answer confidence
                cls_score = cls_scores[b_id][1].item()
                cls_prob = cls_probs[b_id][1].item()

                prob = (s_prob + e_prob) / 2
                score = (s_logit + e_logit) / 2

                doc = batch[b_id]['doc']
                doc_offset = input_ids[b_id].index(102)

                res = all_tokens[top_start_id[0]:top_end_id[0] + 1]
                char_offset = token2char[doc_offset + 1][0]

                if not res or res[0] == "[CLS]" or res[0] == '[SEP]':
                    # ans = helper.get_ans_span(all_tokens[top_start_id[1]:top_end_id[1] + 1])
                    prediction = {'missing_warning': True,
                                  'prob': prob,
                                  'start_end_prob': [s_prob, e_prob],
                                  'score': score,
                                  'cls_score': cls_score,
                                  'cls_prob': cls_prob,
                                  'start_end_score': [s_logit, e_logit],
                                  'value': "", 'answer_start': -1}
                else:
                    # ans = helper.get_ans_span(res)
                    start_map = token2char[top_start_id[0].item()]
                    end_map = token2char[top_end_id[0].item()]
                    span = [start_map[0] - char_offset, end_map[1] - char_offset]
                    ans = doc[span[0]: span[1]]

                    prediction = {'value': ans,
                                  'answer_start': span[0],
                                  'answer_span': span,
                                  'prob': prob,
                                  'start_end_prob': [s_prob, e_prob],
                                  'score': score,
                                  'cls_score': cls_score,
                                  'cls_prob': cls_prob,
                                  'start_end_score': [s_logit, e_logit],
                                  'tokens': res}

                results.append(prediction)
                # results.extend(self._batch_predict(batch, model, tokenizer, device))

        return results
