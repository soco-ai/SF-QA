from collections import Counter, OrderedDict, defaultdict
import string
import re
import argparse
import json
import random
import sys
import nltk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvalMrcEn:
    def evaluate(self, dataset, predictions, config):
            f1 = exact_match = total = 0
            recall_grid = defaultdict(int)

            for article in dataset:
                for paragraph in article['paragraphs']:
                    for qa in paragraph['qas']:
                        f1, exact_match, total, recall_grid = self._single_evaluate(qa, predictions, config, f1, exact_match, total, recall_grid)        

            exact_match = 100.0 * exact_match / total
            f1 = 100.0 * f1 / total
            for r, v in recall_grid.items():
                recall_grid[r] = v * 100.0 / total

            res = {'exact_match': exact_match, 'f1': f1}
            res.update(recall_grid)

            return res


    def _normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def _f1_score(self, prediction, ground_truth):
        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


    def _exact_match_score(self, prediction, ground_truth):
        return (self._normalize_answer(prediction) == self._normalize_answer(ground_truth))


    def _metric_max_over_ground_truths(self, metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def _meteric_recall(self, passages, ground_truth, at_n):
        context_hit = False

        passages = passages[0:at_n]
        for p in passages:
            for a in ground_truth:
                if a.lower() in p['answer'].lower():
                    context_hit = True

                if context_hit:
                    break
        return context_hit


    def _single_evaluate(self, qa, predictions, config, f1, exact_match, total, recall_grid):
        total += 1
        qa['id'] = str(qa['id'])
        if qa['id'] not in predictions:
            message = 'Unanswered question ' + qa['id'] + \
                        ' will receive score 0.'
            print(message, file=sys.stderr)
            return 
        ground_truths = list(map(lambda x: x['text'], qa['answers']))

        best_prediction = predictions[qa['id']]['answer']
        best_exact_match = self._metric_max_over_ground_truths(
            self._exact_match_score, best_prediction, ground_truths)
        exact_match += best_exact_match
        f1 += self._metric_max_over_ground_truths(
            self._f1_score, best_prediction, ground_truths)

        for n in [1, 5, 10, 50, 100]:
            if n > config.param.top_k:
                break
            c_h = self._meteric_recall(predictions[qa['id']]['passages'], ground_truths, n)
            recall_grid['R@{}'.format(n)] += int(c_h)
        
        return f1, exact_match, total, recall_grid


class EvalMrcZh:
    # -*- coding: utf-8 -*-
    '''
    Evaluation script for CMRC 2018
    version: v5 - special
    Note: 
    v5 - special: Evaluate on SQuAD-style CMRC 2018 Datasets
    v5: formatted output, add usage description
    v4: fixed segmentation issues
    '''
    def evaluate(self, ground_truth_file, prediction_file, config):
        f1 = 0
        em = 0
        total_count = 0
        recall_grid = defaultdict(int)
        for instance in ground_truth_file:
            for para in instance["paragraphs"]:
                for qas in para['qas']:
                    f1, em, total_count, recall_grid = self._single_evaluate(qas, prediction_file, config, f1, em, total_count, recall_grid)

        f1_score = 100.0 * f1 / total_count
        em_score = 100.0 * em / total_count

        for r, v in recall_grid.items():
            recall_grid[r] = v * 100.0 / total_count
        res = {'exact_match': em_score, 'f1': f1_score}
        res.update(recall_grid)

        return res


    def _mixed_segmentation(self, in_str, rm_punc=False):
        in_str = str(in_str).lower().strip()
        segs_out = []
        temp_str = ""
        sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
                '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
                '「','」','（','）','－','～','『','』']
        for char in in_str:
            if rm_punc and char in sp_char:
                continue
            if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
                if temp_str != "":
                    ss = nltk.word_tokenize(temp_str)
                    segs_out.extend(ss)
                    temp_str = ""
                segs_out.append(char)
            else:
                temp_str += char

        #handling last part
        if temp_str != "":
            ss = nltk.word_tokenize(temp_str)
            segs_out.extend(ss)

        return segs_out


    # remove punctuation
    def _remove_punctuation(self, in_str):
        in_str = str(in_str).lower().strip()
        sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
                '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
                '「','」','（','）','－','～','『','』']
        out_segs = []
        for char in in_str:
            if char in sp_char:
                continue
            else:
                out_segs.append(char)
        return ''.join(out_segs)


    # find longest common string
    def _find_lcs(self, s1, s2):
        m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
        mmax = 0
        p = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i+1][j+1] = m[i][j]+1
                    if m[i+1][j+1] > mmax:
                        mmax=m[i+1][j+1]
                        p=i+1
        return s1[p-mmax:p], mmax


    def _meteric_recall(self, passages, ground_truth, at_n):
        context_hit = False

        passages = passages[0:at_n]
        for p in passages:
            for a in ground_truth:
                if a.lower() in p['answer'].lower():
                    context_hit = True

                if context_hit:
                    break
        return context_hit


    def _single_evaluate(self, qas, prediction_file, config, f1, em, total_count, recall_grid):
        total_count += 1
        query_id    = qas['id'].strip()
        query_text  = qas['question'].strip()
        answers 	= [x["text"] for x in qas['answers']]

        if query_id not in prediction_file:
            sys.stderr.write('Unanswered question: {}\n'.format(query_id))
            return

        prediction 	= str(prediction_file[query_id]['answer'])
        f1 += self._calc_f1_score(answers, prediction)
        em += self._calc_em_score(answers, prediction)


        for n in [1, 5, 10, 50, 100]:
            if n > config.param.top_k:
                break
            c_h = self._meteric_recall(prediction_file[query_id]['passages'], answers, n)
            recall_grid['R@{}'.format(n)] += int(c_h)

        return f1, em, total_count, recall_grid


    def _calc_f1_score(self, answers, prediction):
        f1_scores = []
        for ans in answers:
            ans_segs = self._mixed_segmentation(ans, rm_punc=True)
            prediction_segs = self._mixed_segmentation(prediction, rm_punc=True)
            lcs, lcs_len = self._find_lcs(ans_segs, prediction_segs)
            if lcs_len == 0:
                f1_scores.append(0)
                continue
            precision 	= 1.0*lcs_len/len(prediction_segs)
            recall 		= 1.0*lcs_len/len(ans_segs)
            f1 			= (2*precision*recall)/(precision+recall)
            f1_scores.append(f1)
        return max(f1_scores)


    def _calc_em_score(self, answers, prediction):
        em = 0
        for ans in answers:
            ans_ = self._remove_punctuation(ans)
            prediction_ = self._remove_punctuation(prediction)
            if ans_ == prediction_:
                em = 1
                break
        return em


def evaluate(lang, data, predictions, config):
    if lang == 'en':
        eval_func = EvalMrcEn()
    elif lang == 'zh':
        eval_func = EvalMrcZh()
    else:
        raise ValueError('lang {} not recognized'.format(lang))

    results = eval_func.evaluate(data['data'], predictions, config)
    return results