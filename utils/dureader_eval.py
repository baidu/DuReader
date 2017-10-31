# -*- coding:utf8 -*-
###############################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
###############################################################################
"""
This module computes evaluation metrics for DuReader dataset.

Authors: liuyuan(liuyuan04@baidu.com), wangyizhong(wangyizhong01@baidu.com)
Date: 2017/10/26 12:00:00
"""
import argparse
import json
import sys
import zipfile

from collections import Counter
from bleu_metric.bleu import Bleu
from rouge_metric.rouge import Rouge

EMPTY = ''
YESNO_LABELS = set(['None', 'Yes', 'No', 'Depends'])


def normalize(s):
    """
    Normalize strings to space joined chars.

    Args:
        s: a list of strings.

    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        normalized.append(' '.join(tokens))
    return normalized


def data_check(obj):
    """
    Check data.

    Raises:
        Raises AssertionError when data is not legal.
    """
    assert 'query_id' in obj, "Missing 'query_id' field."
    assert 'yesno_answers' in obj, \
            "Missing 'yesno_answers' filed. query_id: {}".format(obj['query_id'])
    assert 'entities' in obj, \
            "Missing 'entities' field. query_id: {}".format(obj['query_id'])
    assert 'query_type' in obj, \
            "Missing 'query_type' field. query_id: {}".format(obj['query_type'])

    assert isinstance(obj['entities'], list) and len(obj['entities']) > 0, \
            r"""'entities' field must be a list, and has at least one element,
            which can be a empty list. query_id: {}""".format(obj['query_id'])

    assert isinstance(obj['yesno_answers'], list), \
            r"""'yesno_answers' field must be a list, if the 'query_type' is not
            'YES_NO', then this field should be an empty list.
            query_id: {}""".format(obj['query_id'])

    if obj['query_type'] == 'YES_NO':
        assert len(obj['answers']) == len(obj['yesno_answers']), \
                r"""'yesno_answers' and 'answers' must have same length.
                query_id: {}""".format(obj['quer_id'])


def read_file(file_name, type='predict'):
    """
    Read predict answers or reference answers from file.

    Args:
        file_name: the name of the file containing predict result or reference
                   result.

    Returns:
        A dictionary mapping query_id to the result information. The result
        information itself is also a dictionary with has four keys:
        - query_type: type of the query.
        - yesno_answers: A list of yesno answers corresponding to 'answers'.
        - answers: A list of predicted answers.
        - entities: A list, each element is also a list containing the entities
                    tagged out from the corresponding answer string.
    """
    results = {}
    keys = ['answers', 'yesno_answers', 'entities', 'query_type']
    if type == 'reference':
        keys += ['source']

    zf = zipfile.ZipFile(file_name, 'r')
    for fn in zf.namelist():
        for line in zf.open(fn, 'r'):
            try:
                obj = json.loads(line.strip())
            except ValueError:
                raise ValueError("Every line of data should be legal json")
            data_check(obj)
            qid = obj['query_id']
            assert qid not in results, "Duplicate query_id: {}".format(qid)
            results[qid] = {}
            for k in keys:
                results[qid][k] = obj[k]
    return results


def compute_bleu_rouge(pred_dict, ref_dict, bleu_order=4):
    """
    Compute bleu and rouge scores.
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(ref_dict, pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['bleu_%d' % (i + 1)] = bleu_score
    rouge_score, _ = Rouge().compute_score(ref_dict, pred_dict)
    scores['rouge_l'] = rouge_score
    return scores


def local_prf(pred_list, ref_list):
    """
    Compute local precision recall and f1-score,
    given only one prediction list and one reference list
    """
    common = Counter(pred_list) & Counter(ref_list)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(pred_list)
    r = 1.0 * num_same / len(ref_list)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def compute_prf(pred_dict, ref_dict):
    """
    Compute precision recall and f1-score.
    """
    pred_query_ids = set(pred_dict.keys())
    ref_query_ids = set(ref_dict.keys())
    correct_preds, total_correct, total_preds = 0, 0, 0
    for query_id in ref_query_ids:
        pred_entity_list = pred_dict.get(query_id, [[]])
        assert len(pred_entity_list) == 1, \
            'the number of entity list for query_id {} is not 1.'.format(query_id)
        pred_entity_list = pred_entity_list[0]
        all_ref_entity_lists = ref_dict[query_id]
        best_local_f1 = 0
        best_ref_entity_list = None
        for ref_entity_list in all_ref_entity_lists:
            local_f1 = local_prf(pred_entity_list, ref_entity_list)[2]
            if local_f1 > best_local_f1:
                best_ref_entity_list = ref_entity_list
                best_local_f1 = local_f1
        if best_ref_entity_list is None:
            if len(all_ref_entity_lists) > 0:
                best_ref_entity_list = sorted(all_ref_entity_lists,
                        key=lambda x: len(x))[0]
            else:
                best_ref_entity_list = []
        gold_entities = set(best_ref_entity_list)
        pred_entities = set(pred_entity_list)
        correct_preds += len(gold_entities & pred_entities)
        total_preds += len(pred_entities)
        total_correct += len(gold_entities)
    p = float(correct_preds) / total_preds if correct_preds > 0 else 0
    r = float(correct_preds) / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return {'precision': p, 'recall': r, 'f1': f1}


def prepare_prf(pred_dict, ref_dict):
    """
    Prepares data for calculation of prf scores.
    """
    preds = {k: v['entities'] for k, v in pred_dict.items()}
    refs = {k: v['entities'] for k, v in ref_dict.items()}
    return preds, refs


def get_metrics(pred_result, ref_result, task, source):
    """
    Computes metrics.
    """
    metrics = {}

    ref_result_filtered = {}
    pred_result_filtered = {}
    if source == 'both':
        ref_result_filtered = ref_result
        pred_result_filtered = pred_result
    else:
        for query_id, info in ref_result.items():
            if info['source'] == source:
                ref_result_filtered[query_id] = info
                if query_id in pred_result:
                    pred_result_filtered[query_id] = pred_result[query_id]

    if task == 'basic' or task == 'yesno' or task == 'all':
        pred_dict, ref_dict = prepare_bleu(pred_result_filtered,
                ref_result_filtered,
                task)
        metrics = compute_bleu_rouge(pred_dict, ref_dict)
    elif task == 'entity':
        pred_dict, ref_dict = prepare_prf(pred_result_filtered,
                ref_result_filtered)
        pred_dict_bleu, ref_dict_bleu = prepare_bleu(pred_result_filtered,
                ref_result_filtered,
                task)
        metrics = compute_prf(pred_dict, ref_dict)
        metrics.update(compute_bleu_rouge(pred_dict_bleu, ref_dict_bleu))
    else:
        raise ValueError("Illegal task name: {}".format(task))

    return metrics


def prepare_bleu(pred_result, ref_result, task):
    """
    Prepares data for calculation of bleu and rouge scores.
    """
    pred_list, ref_list = [], []
    qids = ref_result.keys()
    for qid in qids:
        if task == 'basic':
            pred, ref = get_basic_result(qid, pred_result, ref_result)
        elif task == 'yesno':
            pred, ref = get_yesno_result(qid, pred_result, ref_result)
        elif task == 'all':
            pred, ref = get_all_result(qid, pred_result, ref_result)
        elif task == 'entity':
            pred, ref = get_entity_result(qid, pred_result, ref_result)
        else:
            raise ValueError("Illegal task name: {}".format(task))
        if pred and ref:
            pred_list += pred
            ref_list += ref
    pred_dict = dict(pred_list)
    ref_dict = dict(ref_list)
    for qid, ans in ref_dict.items():
        if not ans or ans == [EMPTY]:
            if pred_dict[qid] == [EMPTY]:
                del ref_dict[qid]
                del pred_dict[qid]

    for k, v in pred_dict.items():
        assert len(v) == 1, \
            "There should be only one predict answer. query_id: {}".format(k)
    return pred_dict, ref_dict


def get_basic_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'basic'.

    Args:
        qid: query_id.
        pred_result: A dict include all query_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all query_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same query_id. Each list has
        elements of tuple (query_id, answers), 'answers' is a list of strings.
    """
    ref_ans = normalize(ref_result[qid]['answers'])
    if not ref_ans:
        ref_ans = [EMPTY]
    pred_ans = normalize(pred_result.get(qid, {}).get('answers', [])[:1])
    if not pred_ans:
        pred_ans = [EMPTY]

    return [(qid, pred_ans)], [(qid, ref_ans)]


def get_entity_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'entity'.

    Args:
        qid: query_id.
        pred_result: A dict include all query_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all query_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same query_id. Each list has
        elements of tuple (query_id, answers), 'answers' is a list of strings.
    """
    if ref_result[qid]['query_type'] != 'YES_NO':
        return None, None
    return get_basic_result(qid, pred_result, ref_result)


def get_yesno_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'yesno'.

    Args:
        qid: query_id.
        pred_result: A dict include all query_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all query_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same query_id. Each list has
        elements of tuple (query_id, answers), 'answers' is a list of strings.
    """
    def _uniq(li, is_ref):
        uniq_li = []
        left = []
        keys = set()
        for k, v in li:
            if k not in keys:
                uniq_li.append((k, v))
                keys.add(k)
            else:
                left.append((k, v))

        if is_ref:
            dict_li = dict(uniq_li)
            for k, v in left:
                dict_li[k] += v
            uniq_li = [(k, v) for k, v in dict_li.items()]
        return uniq_li

    def _expand_result(uniq_li):
        expanded = uniq_li[:]
        keys = set([x[0] for x in uniq_li])
        for k in YESNO_LABELS - keys:
            expanded.append((k, [EMPTY]))
        return expanded

    def _get_yesno_ans(qid, result_dict, is_ref=False):
        if qid not in result_dict:
            return [(str(qid) + '_' + k, v) for k, v in _expand_result([])]
        yesno_answers = result_dict[qid]['yesno_answers']
        answers = normalize(result_dict[qid]['answers'])
        lbl_ans = _uniq([(k, [v]) for k, v in zip(yesno_answers, answers)], is_ref)
        ret = [(str(qid) + '_' + k, v) for k, v in _expand_result(lbl_ans)]
        return ret

    if ref_result[qid]['query_type'] != 'YES_NO':
        return None, None

    ref_ans = _get_yesno_ans(qid, ref_result, is_ref=True)
    pred_ans = _get_yesno_ans(qid, pred_result)
    return pred_ans, ref_ans


def get_all_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'all'.

    Args:
        qid: query_id.
        pred_result: A dict include all query_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all query_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same query_id. Each list has
        elements of tuple (query_id, answers), 'answers' is a list of strings.
    """
    if ref_result[qid]['query_type'] == 'YES_NO':
        return get_yesno_result(qid, pred_result, ref_result)
    return get_basic_result(qid, pred_result, ref_result)


def format_metrics(metrics, task, err_msg):
    """
    Format metrics. 'err' field returns any error occured during evaluation.

    Args:
        metrics: A dict object contains metrics for different tasks.
        task: Task name.
        err_msg: Exception raised during evaluation.
    Returns:
        Formatted result. If task is 'entity', the returned result have 4
        fields, for example:
            >>> {'precision': '0.9', 'recall': '0.9', 'f1-score': '0.9',
            ...  'err': None}

        If the task is one of 'basic', 'yesno', or 'all', the result should
        be like:
            >>> {'bleu_4': '0.25', 'rouge_l': '0.28', 'err': None}

        All metrics will be cast into strings intead of float numbers.

    """
    result = {}
    sources = ['both', 'search', 'zhidao']
    if err_msg is not None:
        return {'errorMsg': str(err_msg), 'errorCode': 1, 'data': []}
    data = []
    if task == 'entity':
        for src in sources:
            data.append((src, 'F1', round(metrics[src].get('f1', 0) * 100, 2)))
            data.append(
                (src, 'Precision',
                 round(metrics[src].get('precision', 0) * 100, 2)))
            data.append(
                (src, 'Recall', round(metrics[src].get('recall', 0) * 100, 2)))
            data.append(
                (src, 'Bleu-4', round(metrics[src].get('bleu_4', 0) * 100, 2)))
            data.append(
                (src, 'Rouge-L', round(metrics[src].get('rouge_l', 0) * 100, 2)))
        result['data'] = data
        result['errorCode'] = 0
        result['errorMsg'] = 'success'
    else:
        for src in sources:
            data.append(
                (src, 'Bleu-4', round(metrics[src].get('bleu_4', 0) * 100, 2)))
            data.append(
                (src, 'Rouge-L', round(metrics[src].get('rouge_l', 0) * 100, 2)))
        result['data'] = data
        result['errorCode'] = 0
        result['errorMsg'] = 'success'

    return result


def main(args):
    """
    Do evaluation.
    """
    err = None
    metrics = {}
    try:
        pred_result = read_file(args.pred_file)
        ref_result = read_file(args.ref_file, type='reference')
        for source in ['both', 'zhidao', 'search']:
            metrics[source] = get_metrics(
                    pred_result, ref_result, args.task, source)
    except ValueError as ve:
        err = ve
    except AssertionError as ae:
        err = ae
    print format_metrics(metrics, args.task, err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file', help='predict file')
    parser.add_argument('ref_file', help='reference file')
    parser.add_argument('task', help='task name: basic|yesno|all|entity')

    args = parser.parse_args()
    main(args)
