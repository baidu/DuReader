# -*- coding:utf8 -*-
###############################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
###############################################################################
"""
This module computes evaluation metrics for DuReader dataset.

Authors: liuyuan(liuyuan04@baidu.com)
Data: 2017/10/26 12:00:00
"""

import hashlib
import json
import os
import sys
import random
from brc_eval import compute_metrics_from_list

EMPTY = ''
YESNO_LABELS = {
        'None': 0,
        'Yes': 1,
        'No': 2,
        'Depends': 3}

def getid(query):
    """
    compute id.
    """
    m = hashlib.md5()
    #print >> sys.stderr, '===', query.encode('utf8'), type(query)
    m.update(query.encode('utf8'))
    return m.hexdigest()

def build_yesno_golden(obj):
    """
    Get golden reference of query_id.
    """
    ret_list = []
    if obj['query_type'] != 'YES_NO':
        return ret_list
    answers = obj['answers']
    yesno_answers = obj['yesno_answers']
    assert len(answers) == len(yesno_answers), \
            "num_answers != num_yesno_answers"
    qid = getid(obj['query'])
    for ans, lbl in zip(answers, yesno_answers):
        key = qid + '_' + lbl
        result = {}
        result[key] = result.get(key, [])
        result[key].append(ans)
        ret_list.append(result)

    for lbl in set(YESNO_LABELS.keys()) - set(yesno_answers):
        key = qid + '_' + lbl
        result = {}
        result[key] = result.get(key, [])
        result[key].append(EMPTY)
        ret_list.append(result)

    return ret_list

def build_yesno_human(obj):
    """
    Get human result.
    """
    ret_list = []
    if obj['query_type'] != 'YES_NO':
        return ret_list
    answers = obj['answers_by_annotator_2']
    labels = obj['yesno_answers_by_annotator_2']
    qid = getid(obj['query'])
    for ans, lbl in zip(answers, labels):
        key = qid + '_' + lbl
        result = {}
        result[key] = result.get(key, [])
        result[key].append(ans)
        ret_list.append(result)

    for lbl in set(YESNO_LABELS.keys()) - set(labels):
        key = qid + '_' + lbl
        result = {}
        result[key] = result.get(key, [])
        result[key].append(EMPTY)
        ret_list.append(result)

    keys = set()
    no_dup_list = []
    for result in ret_list:
        if result.keys()[0] not in keys:
            no_dup_list.append(result)
            keys.add(result.keys()[0])

    return no_dup_list

def build_yesno_ctrl(obj):
    """
    Get control experiment result.
    """
    ret_list = []
    if obj['query_type'] != 'YES_NO':
        return ret_list
    answers = obj['pred_answers']
    labels = YESNO_LABELS.keys()
    answers_exp = answers * len(labels)
    qid = getid(obj['query'])
    for ans, lbl in zip(answers_exp, labels):
        result = {}
        key = qid + '_' + lbl
        result[key] = result.get(key, [])
        result[key].append(ans)
        ret_list.append(result)

    return ret_list

def build_yesno_random(obj):
    """
    Get results of random model.
    """
    ret_list = []
    if obj['query_type'] != 'YES_NO':
        return ret_list

    reverse_dict = {v: k for k, v in YESNO_LABELS.items()}
    answers = obj['pred_answers']
    labels = [random.choice(YESNO_LABELS.keys())]
    qid = getid(obj['query'])
    for ans, lbl in zip(answers, labels):
        key = qid + '_' + lbl
        result = {key: [ans]}
        ret_list.append(result)

    for lbl in set(YESNO_LABELS.keys()) - set(labels):
        key = qid + '_' + lbl
        result = {key: [EMPTY]}
        ret_list.append(result)

    return ret_list

def build_yesno_exp(obj):
    """
    Get results of experiment.
    """
    ret_list = []
    if obj['query_type'] != 'YES_NO':
        return ret_list

    reverse_dict = {v: k for k, v in YESNO_LABELS.items()}
    answers = obj['pred_answers']
    labels_raw = obj['yesno_answers_pred']
    labels = [reverse_dict[i] for i in labels_raw]
    qid = getid(obj['query'])
    for ans, lbl in zip(answers, labels):
        key = qid + '_' + lbl
        result = {key: [ans]}
        ret_list.append(result)

    for lbl in set(YESNO_LABELS.keys()) - set(labels):
        key = qid + '_' + lbl
        result = {key: [EMPTY]}
        ret_list.append(result)

    return ret_list

def build_normal(obj):
    """
    Normal answer result.
    """
    ret_list = []
    answers = obj['pred_answers']
    qid = getid(obj['query'])
    result = {qid: answers}
    ret_list.append(result)
    return ret_list

def build_normal_golden(obj):
    """
    build normal golden
    """
    ret_list = []
    answers = obj['answers']
    qid = getid(obj['query'])
    result = {qid: answers}
    ret_list.append(result)
    return ret_list

def merge_dict(li_pred, li_ref):
    """
    merge results.
    """
    def merge_one(a, b):
        for k, li in b.items():
            a[k] = a.get(k, [])
            a[k] += li
            a[k] = list(set(a[k]))
        return a

    pred_list, ref_list = [], []

    pred_dict = reduce(merge_one, li_pred)
    ref_dict = reduce(merge_one, li_ref)
    assert set(pred_dict.keys()) == set(ref_dict.keys())
    for qid in set(pred_dict.keys()):
        if (len(ref_dict[qid]) == 0 and len(pred_dict[qid]) == 0) \
                or (ref_dict[qid] == [EMPTY] and pred_dict[qid] == [EMPTY]):
            continue
        assert len(pred_dict[qid]) <= 1, 'qid: {}'.format(qid)
        obj_pred = {'query': qid, 'answer': pred_dict[qid]}
        obj_ref = {'query': qid, 'answer': ref_dict[qid]}

        pred_list.append(obj_pred)
        ref_list.append(obj_ref)
    return pred_list, ref_list

def get_metrics(pred_results, ref_results):
    """
    compute metric.
    """
    pred_list, ref_list = merge_dict(pred_results, ref_results)
    metrics = compute_metrics_from_list(pred_list, ref_list, 4)
    return metrics

def main():
    """
    The main logic.
    """
    normal_results = []
    normal_golden_results = []
    yesno_exp_results = []
    yesno_random_results = []
    yesno_ctrl_results = []
    yesno_human_results = []
    yesno_golden_results = []
    for line in sys.stdin:
        obj = json.loads(line.strip())
        normal_results += build_normal(obj)
        normal_golden_results += build_normal_golden(obj)
        yesno_exp_results += build_yesno_exp(obj)
        yesno_random_results += build_yesno_random(obj)
        yesno_ctrl_results += build_yesno_ctrl(obj)
        yesno_human_results += build_yesno_human(obj)
        yesno_golden_results += build_yesno_golden(obj)

    yesno_ctrl_metric = get_metrics(yesno_ctrl_results,
            yesno_golden_results)
    yesno_exp_metric = get_metrics(yesno_exp_results,
            yesno_golden_results)
    yesno_random_metric = get_metrics(yesno_random_results,
            yesno_golden_results)
    yesno_human_metric = get_metrics(yesno_human_results,
            yesno_golden_results)
    normal_metric = get_metrics(normal_results, normal_golden_results)

    print 'ctrl: ', yesno_ctrl_metric
    print 'human:', yesno_human_metric
    print 'random:', yesno_random_metric
    print 'exp:', yesno_exp_metric
    print 'normal: ', normal_metric


if __name__ == '__main__':
    main()
