import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import os
import torch
import re
from collections import namedtuple
import logging
from soco_device import DeviceCheck
from soco_openqa.soco_mrc import util
from soco_openqa.soco_mrc.models.bert_model import BertForQuestionAnsweringWithReranker
from soco_openqa.cloud_bucket import CloudBucket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tokenization_utils").setLevel(logging.ERROR)

MODEL_MAP = {
    'MrcModel': AutoModelForQuestionAnswering, 
    'MrcRerankerModel': BertForQuestionAnsweringWithReranker, 
    }



class ModelBase(object):
    def __init__(self,
        region, 
        n_gpu=0,
        fp16=False, 
        quantize=False, 
        multiprocess=False
    ):
        logger.info("Op in {} region".format(region))
        self.n_gpu_request = n_gpu
        self.region = region
        self.fp16 = fp16
        self.quantize = quantize
        self.multiprocess = multiprocess
        self.cloud_bucket = CloudBucket(region)
        self._models = dict()
        self.max_input_length = 512
        self.device_check = DeviceCheck()


    def _load_model(self, model_id):
        # a naive check. if too big, just reset
        if len(self._models) > 20:
            self._models = dict()

        if model_id not in self._models:
            path = os.path.join('resources', model_id)
            self.cloud_bucket.download_model('mrc-models', model_id)
            model_class = self.__class__.__name__
            model = MODEL_MAP[model_class].from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

            device_name, device_ids = self.device_check.get_device_by_model(model_id, n_gpu=self.n_gpu_request)
            self.n_gpu_allocate = len(device_ids)
            device = '{}:{}'.format(device_name, device_ids[0]) if self.n_gpu_allocate == 1 else device_name

            if self.fp16 and 'cuda' in device:
                logger.info('Use fp16')
                model.half()
            if self.quantize and device == 'cpu':
                logger.info('Use quantization')
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8)
            model.to(device)

            # multi gpu inference
            if self.n_gpu_allocate > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model, device_ids=device_ids)

            self._models[model_id] = (tokenizer, model, device)

        else:
            # if loaded as cpu, check if gpu is available
            _, _, device = self._models[model_id]
            if self.n_gpu_request > 0 and device == 'cpu':
                device_name, device_ids = self.device_check.get_device_by_model(model_id, n_gpu=1)
                new_device = '{}:{}'.format(device_name, device_ids[0]) if len(device_ids) == 1 else device_name
                if new_device != device:
                    logger.info('Reloading')
                    self._models.pop(model_id)
                    self._load_model(model_id)

        return self._models[model_id]


    def batch_predict(self, model_id, data, **kwargs):
        raise NotImplementedError

    def _get_param(self, kwargs):
        batch_size = kwargs.pop('batch_size', 10)
        merge_pred = kwargs.pop('merge_pred', False)
        stride = kwargs.pop('stride', 0)

        return batch_size, merge_pred, stride



class MrcModel(ModelBase):
    def batch_predict(self, model_id, data, **kwargs):
        batch_size, merge_pred, stride = self._get_param(kwargs)

        tokenizer, model, device = self._load_model(model_id)

        features = util.convert_examples_to_features(
                                                tokenizer, 
                                                data, 
                                                self.max_input_length,
                                                merge_pred,
                                                stride)

        results = []
        for batch in util.chunks(features, batch_size):
            padded = util.pad_batch(batch)
            input_ids, token_type_ids, attn_masks = padded

            with torch.no_grad():
                start_scores, end_scores = model(torch.tensor(input_ids).to(device),
                                                token_type_ids=torch.tensor(token_type_ids).to(device),
                                                attention_mask=torch.tensor(attn_masks).to(device))
                start_probs = torch.softmax(start_scores, dim=1)
                end_probs = torch.softmax(end_scores, dim=1)

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

                prob = (s_prob + e_prob) / 2
                score = (s_logit + e_logit) / 2

                doc = batch[b_id]['doc']
                doc_offset = input_ids[b_id].index(102)

                res = all_tokens[top_start_id[0]:top_end_id[0] + 1]
                char_offset = token2char[doc_offset + 1][0]
                example_idx = batch[b_id]['example_idx']

                if not res or res[0] == "[CLS]" or res[0] == '[SEP]' or top_start_id[0].item() <= doc_offset:
                    prediction = {'missing_warning': True,
                                'prob': prob,
                                'start_end_prob': [s_prob, e_prob],
                                'score': score,
                                'start_end_score': [s_logit, e_logit],
                                'value': "", 
                                'answer_start': -1,
                                'example_idx': example_idx}
                else:
                    if not merge_pred:
                        start_map = token2char[top_start_id[0].item()]
                        end_map = token2char[top_end_id[0].item()]
                        span = [start_map[0] - char_offset, end_map[1] - char_offset]
                        ans = doc[span[0]: span[1]]
                    else:
                        base_idx = batch[b_id]['base_idx']
                        orig_doc = batch[b_id]['orig_doc']
                        orig_token2char = batch[b_id]['orig_offset_mapping']

                        # map token index, then use offset mapping to map to original position
                        orig_start_map = orig_token2char[top_start_id[0].item() + base_idx - doc_offset - 1]
                        orig_end_map = orig_token2char[top_end_id[0].item() + base_idx - doc_offset - 1]
                        span = [orig_start_map[0], orig_end_map[1]]
                        ans = orig_doc[span[0]: span[1]]
                        try:
                            start_map = token2char[top_start_id[0].item()]
                            end_map = token2char[top_end_id[0].item()]
                            debug_span = [start_map[0] - char_offset, end_map[1] - char_offset]
                            debug_ans = doc[debug_span[0]: debug_span[1]]
                            assert debug_ans == ans
                        except Exception as e:
                            print(e)
                            print('chunk ans: {} '.format(debug_ans))
                            print('doc ans: {} '.format(ans))
                            print('chunk span: {} vs doc span: {}'.format(debug_span, span))

                    prediction = {'value': ans,
                                'answer_start': span[0],
                                'answer_span': span,
                                'prob': prob,
                                'start_end_prob': [s_prob, e_prob],
                                'score': score,
                                'start_end_score': [s_logit, e_logit],
                                'tokens': res,
                                'example_idx': example_idx}

                results.append(prediction)
                

        # merge predictions
        if merge_pred:
            results = util.merge_predictions(results)

        return results


class MrcRerankerModel(ModelBase):

    def batch_predict(self, model_id, data, **kwargs):
        batch_size, merge_pred, stride = self._get_param(kwargs)

        tokenizer, model, device = self._load_model(model_id)

        features = util.convert_examples_to_features(
                                                tokenizer, 
                                                data, 
                                                self.max_input_length,
                                                merge_pred,
                                                stride)

        results = []
        for batch in util.chunks(features, batch_size):
            padded = util.pad_batch(batch)
            input_ids, token_type_ids, attn_masks = padded

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
                example_idx = batch[b_id]['example_idx']

                if not res or res[0] == "[CLS]" or res[0] == '[SEP]' or top_start_id[0].item() <= doc_offset:
                    prediction = {'missing_warning': True,
                                  'prob': prob,
                                  'start_end_prob': [s_prob, e_prob],
                                  'score': score,
                                  'cls_score': cls_score,
                                  'cls_prob': cls_prob,
                                  'start_end_score': [s_logit, e_logit],
                                  'value': "", 
                                  'answer_start': -1,
                                  'example_idx': example_idx}
                else:
                    if not merge_pred:
                        start_map = token2char[top_start_id[0].item()]
                        end_map = token2char[top_end_id[0].item()]
                        span = [start_map[0] - char_offset, end_map[1] - char_offset]
                        ans = doc[span[0]: span[1]]
                    else:
                        base_idx = batch[b_id]['base_idx']
                        orig_doc = batch[b_id]['orig_doc']
                        orig_token2char = batch[b_id]['orig_offset_mapping']

                        # map token index, then use offset mapping to map to original position
                        orig_start_map = orig_token2char[top_start_id[0].item() + base_idx - doc_offset - 1]
                        orig_end_map = orig_token2char[top_end_id[0].item() + base_idx - doc_offset - 1]
                        span = [orig_start_map[0], orig_end_map[1]]
                        ans = orig_doc[span[0]: span[1]]
                        try:
                            start_map = token2char[top_start_id[0].item()]
                            end_map = token2char[top_end_id[0].item()]
                            debug_span = [start_map[0] - char_offset, end_map[1] - char_offset]
                            debug_ans = doc[debug_span[0]: debug_span[1]]
                            assert debug_ans == ans
                        except Exception as e:
                            print(e)
                            print('chunk ans: {} '.format(debug_ans))
                            print('doc ans: {} '.format(ans))
                            print('chunk span: {} vs doc span: {}'.format(debug_span, span)) 

                    prediction = {'value': ans,
                                  'answer_start': span[0],
                                  'answer_span': span,
                                  'prob': prob,
                                  'start_end_prob': [s_prob, e_prob],
                                  'score': score,
                                  'cls_score': cls_score,
                                  'cls_prob': cls_prob,
                                  'start_end_score': [s_logit, e_logit],
                                  'tokens': res,
                                  'example_idx': example_idx}

                results.append(prediction)
    
        # merge predictions
        if merge_pred:
            results = util.merge_predictions(results)

        return results
