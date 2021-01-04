
import logging
import re
import string
from collections import Counter, OrderedDict
import nltk
from typing import Any, Callable, Dict, Generator, Sequence

logger = logging.getLogger(__name__)

def chunks(l: Sequence, n: int = 5) -> Generator[Sequence, None, None]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def stride_chunks(l: Sequence, win_len: int, stride_len: int):
    s_id = 0
    e_id = min(len(l), win_len)

    while True:
        yield s_id, l[s_id:e_id]

        if e_id == len(l):
            break

        s_id = s_id + stride_len
        e_id = min(s_id + win_len, len(l))


class DefaultListOrderedDict(OrderedDict):
    def __missing__(self,k):
        self[k] = []
        return self[k]


def convert_examples_to_features(tokenizer, data, max_input_length, merge_pred=False, stride=0):
    features = []
    for example_idx, d in enumerate(data):
        doc = _normalize_text(d['doc'])
        q = _normalize_text(d['q'])
        if not merge_pred:
            temp = tokenizer.encode_plus(q, doc, return_offsets_mapping=True, truncation=False)
            # cut by max_input_length
            input_ids = temp.data['input_ids']
            if len(input_ids) > max_input_length:
                logger.info("Input length {} is too big. Cap to {}".format(len(input_ids), max_input_length))
                for k, v in temp.data.items():
                    temp.data[k] = cap_to(v, max_input_length)

            temp['doc'] = doc
            temp['q'] = q
            temp['example_idx'] = example_idx
            features.append(temp)
        else:
            seq_pair_added_toks = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
            q_toks = tokenizer.tokenize(q)
            q_enc = tokenizer.encode_plus(q, return_offsets_mapping=True)
            window_len = max_input_length - len(q_toks) - seq_pair_added_toks
            doc_enc = tokenizer.encode_plus(doc, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
            for base_idx, chunk_mapping in stride_chunks(doc_enc['offset_mapping'], window_len, stride):
                chunk_st = chunk_mapping[0][0]
                chunk_ed = chunk_mapping[-1][1]
                chunk = doc[chunk_st: chunk_ed]

                new_dict = {}
                # add last [SEP]
                chunk_input_ids = doc_enc['input_ids'][base_idx:base_idx+len(chunk_mapping)] + [102]
                chunk_token_type_ids = [1]*len(chunk_mapping) + [1]
                chunk_attention_mask = [1]*len(chunk_mapping) + [1]
                tmp_chunk_offset_mapping = doc_enc['offset_mapping'][base_idx:base_idx+len(chunk_mapping)]
                base_offset = tmp_chunk_offset_mapping[0][0]
                chunk_offset_mapping = []
                for offset in tmp_chunk_offset_mapping:
                    chunk_offset_mapping.append((offset[0]-base_offset, offset[1]-base_offset))
                chunk_offset_mapping.append((0, 0))

                new_dict['input_ids'] = q_enc['input_ids'] + chunk_input_ids
                new_dict['token_type_ids'] = q_enc['token_type_ids'] + chunk_token_type_ids
                new_dict['attention_mask'] = q_enc['attention_mask'] + chunk_attention_mask
                new_dict['offset_mapping'] = q_enc['offset_mapping'] + chunk_offset_mapping
                
                new_dict['doc'] = chunk
                new_dict['orig_doc'] = doc
                new_dict['q'] = q
                new_dict['example_idx'] = example_idx
                new_dict['base_idx'] = base_idx
                new_dict['orig_offset_mapping'] = doc_enc['offset_mapping']

                features.append(new_dict)

    return features

def merge_predictions(results, strategy='max'):
    """
    merge chunk predictions indicated by example_idx
    :param results: batched results
    :type results: dict
    :param strategy: 'max' or 'merge' indicating how to merge results, defaults to 'max'
                    'max': only keep results with highest probability
                    'merge': keep all predicted results
    :type strategy: str, optional
    :return: a list of dictionary containing batch number of results
    :rtype: list
    """

    idx_res_map = OrderedDict()
    for r in results:
        example_idx = r.get('example_idx')
        if example_idx not in idx_res_map:
            idx_res_map[example_idx] = r
        else:
            if strategy == 'max':   # use max prob answer
                if r['prob'] > idx_res_map[example_idx]['prob'] and not r.get('missing_warning'):
                    idx_res_map[example_idx] = r
            elif strategy == 'merge':   # merge all chunk answer
                if not r.get('missing_warning'):
                    if idx_res_map[example_idx].get('missing_warning'):
                        idx_res_map[example_idx] = r
                        continue
                    idx_res_map[example_idx]['value_type'] = r.pop('value_type')
                    idx_res_map[example_idx]['example_idx'] = r.pop('example_idx')
                    # only keep value that is different
                    keep_idx = [i for i, v in enumerate(r.get('value')) if v not in idx_res_map[example_idx].get('value')]
                    for k in r.keys():                        
                        idx_res_map[example_idx][k].extend([v for i, v in enumerate(r[k]) if i in keep_idx])

    results = [v for v in idx_res_map.values()]
    return results

    

def pad_batch(batch):
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

    return input_ids, token_type_ids, attn_masks
        

def _normalize_text(text):
    return re.sub('\s+', ' ', text)

def cap_to(seq, max_len):
    prefix = seq[0:-1][0:max_len - 1]
    return prefix + [seq[-1]]

def get_span_from_ohe(bio_labels):
    left = 0
    right = 1
    found_st = False
    found_ed = False
    span_indexes = []

    while right < len(bio_labels):
        if not found_st and not found_ed:
            if bio_labels[right] == 0:
                right += 1
                continue
            else:
                found_st = True
                left = right
        if found_st:
            if bio_labels[right] == 1:
                right += 1
                continue
            else:
                span_indexes.append((left, right-1))
                left = right
                right = left + 1
                found_st = False
                found_ed = False

    if set(bio_labels[left:right]) == {1}:
        span_indexes.append((left, right-1))
    
    return span_indexes

def get_ans_span(res):
    if not res:
        return ""

    for i, t in enumerate(res):
        if t.startswith("##"):
            res[i - 1] += t[2:]
            res[i] = ""

    value = " ".join([x for x in res if x != ""])
    return value

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def token2char(orig_str, tokens):
    norm_tokens = [t.replace('##', '') for t in tokens]

    token_id = 0
    token_char_id = 0
    token2char_map = {}  # token_id -> [start, end]

    token2char_map[token_id] = [0, None]
    for c_id, c in enumerate(orig_str):
        if is_whitespace(c):
            token2char_map[token_id][1] = c_id
            token_id += 1
            token_char_id = 0
            token2char_map[token_id] = [c_id+1, None]
            continue

        if token_char_id < len(norm_tokens[token_id]) and c == norm_tokens[token_id][token_char_id]:
            token_char_id += 1
        else:
            token2char_map[token_id][1] = c_id
            token_id += 1
            token_char_id = 0
            token2char_map[token_id] = [c_id, None]

            if c == norm_tokens[token_id][token_char_id]:
                token_char_id += 1

    token2char_map[token_id][1] = c_id+1
    # print(token2char_map)
    return token2char_map
