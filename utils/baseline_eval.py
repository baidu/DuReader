# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module computes evaluation metrics for DuReader dataset.
"""


import hashlib
import json
import os
import sys
import random
from brc_eval import compute_metrics_from_list
from find_answer import find_best_question_match

EMPTY = ''
YESNO_LABELS = {
        'None': 0,
        'Yes': 1,
        'No': 2,
        'Depends': 3}


def getid(question):
    """
    compute id.
    """
    m = hashlib.md5()
    m.update(question.encode('utf8'))
    return m.hexdigest()


def build_yesno_subtype_golden(obj, subtype):
    """
    Get golden reference of question_id.
    """
    ret_list = []
    if obj['question_type'] != 'YES_NO':
        return ret_list

    if subtype != 0:
        if obj['yesno_type'] != subtype:
            return ret_list

    answers = obj['answers']
    yesno_answers = obj['yesno_answers']
    assert len(answers) == len(yesno_answers), \
            "num_answers != num_yesno_answers"
    qid = getid(obj['question'])
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


def build_yesno_subtype_human(obj, subtype):
    """
    Get human result.
    """
    ret_list = []
    if obj['question_type'] != 'YES_NO':
        return ret_list
    if subtype != 0:
        if obj['yesno_type'] != subtype:
            return ret_list

    answers = obj['answers_by_annotator_2']
    labels = obj['yesno_answers_by_annotator_2']
    qid = getid(obj['question'])
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


def build_yesno_subtype_ctrl(obj, subtype):
    """
    Get control experiment result.
    """
    ret_list = []
    if obj['question_type'] != 'YES_NO':
        return ret_list
    if subtype != 0:
        if obj['yesno_type'] != subtype:
            return ret_list

    answers = obj['pred_answers']
    labels = YESNO_LABELS.keys()
    answers_exp = answers * len(labels)
    qid = getid(obj['question'])
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
    if obj['question_type'] != 'YES_NO':
        return ret_list

    reverse_dict = {v: k for k, v in YESNO_LABELS.items()}
    answers = obj['pred_answers']
    labels = [random.choice(YESNO_LABELS.keys())]
    qid = getid(obj['question'])
    for ans, lbl in zip(answers, labels):
        key = qid + '_' + lbl
        result = {key: [ans]}
        ret_list.append(result)

    for lbl in set(YESNO_LABELS.keys()) - set(labels):
        key = qid + '_' + lbl
        result = {key: [EMPTY]}
        ret_list.append(result)

    return ret_list


def build_yesno_subtype_exp(obj, subtype):
    """
    Get results of experiment.
    """
    ret_list = []
    if obj['question_type'] != 'YES_NO':
        return ret_list

    if subtype != 0:
        if obj['yesno_type'] != subtype:
            return ret_list

    reverse_dict = {v: k for k, v in YESNO_LABELS.items()}
    answers = obj['pred_answers']
    labels_raw = [
            x[1]
            for x in sorted(obj['yesno_answers_pred'], key=lambda x: x[0])]
    labels = [reverse_dict[i] for i in labels_raw]
    qid = getid(obj['question'])
    for ans, lbl in zip(answers, labels):
        key = qid + '_' + lbl
        result = {key: [ans]}
        ret_list.append(result)

    keys = set()
    no_dup_list = []
    for result in ret_list:
        if result.keys()[0] not in keys:
            no_dup_list.append(result)
            keys.add(result.keys()[0])

    for lbl in set(YESNO_LABELS.keys()) - set(labels):
        key = qid + '_' + lbl
        result = {key: [EMPTY]}
        no_dup_list.append(result)

    return no_dup_list


def build_yesno_selected(obj):
    """
    A control exp on selected passage, which is selected according to
    recall of question tokens.
    """
    ret_list = []
    if obj['question_type'] != 'YES_NO':
        return ret_list

    docs = obj['documents']
    selected_paras = []
    for d in docs:
        most_related_id, score = find_best_question_match(d,
                obj['segmented_question'],
                with_score=True)
        selected_paras.append(
                (d['segmented_paragraphs'][most_related_id], score))

    answer = ''.join(
            sorted(selected_paras, key=lambda x: x[1], reverse=True)[0][0])
    labels = YESNO_LABELS.keys()
    answers = [answer] * len(labels)
    qid = getid(obj['question'])
    for ans, lbl in zip(answers, labels):
        ret_list.append({qid + '_' + lbl: [ans]})

    return ret_list


def build_normal(obj):
    """
    Normal answer result.
    """
    ret_list = []
    if obj['question_type'] == 'YES_NO':
        return build_yesno_subtype_exp(obj, 0)
    answers = obj['pred_answers'][:1]
    qid = getid(obj['question'])
    result = {qid: answers}
    ret_list.append(result)
    return ret_list


def build_normal_golden(obj):
    """
    build normal golden
    """
    ret_list = []
    if obj['question_type'] == 'YES_NO':
        return build_yesno_subtype_golden(obj, 0)
    answers = obj['answers']
    qid = getid(obj['question'])
    result = {qid: answers}
    ret_list.append(result)
    return ret_list


def build_normal_human(obj):
    """
    build normal human results.
    """
    ret_list = []
    if obj['question_type'] == 'YES_NO':
        return build_yesno_subtype_human(obj, 0)

    answer = obj['answers_by_annotator_2'][0] \
            if len(obj['answers_by_annotator_2']) > 0 \
            else EMPTY
    qid = getid(obj['question'])
    ret_list.append({qid: [answer]})
    return ret_list


def build_normal_human_golden(obj):
    """
    build golden results for human normal.
    """
    if obj['question_type'] == 'YES_NO':
        return build_yesno_subtype_golden(obj, 0)
    return build_normal_golden(obj)


def merge_dict(li_pred, li_ref):
    """
    merge results.
    """
    def __merge_one(a, b):
        for k, li in b.items():
            a[k] = a.get(k, [])
            a[k] += li
            a[k] = list(set(a[k]))
        return a

    pred_list, ref_list = [], []

    pred_dict = reduce(__merge_one, li_pred)
    ref_dict = reduce(__merge_one, li_ref)
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            '{} vs {}'.format(pred_dict.keys(), ref_dict.keys())
    for qid in set(pred_dict.keys()):
        if (len(ref_dict[qid]) == 0 and len(pred_dict[qid]) == 0) \
                or (ref_dict[qid] == [EMPTY] and pred_dict[qid] == [EMPTY]):
            continue
        assert len(pred_dict[qid]) <= 1, 'qid: {}'.format(qid)
        obj_pred = {'question': qid, 'answer': pred_dict[qid]}
        obj_ref = {'question': qid, 'answer': ref_dict[qid]}

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
    normal_results = []
    normal_golden_results = []

    yesno_exp_results = []
    yesno_type1_exp_results = []
    yesno_type2_exp_results = []

    yesno_random_results = []

    yesno_ctrl_results = []
    yesno_type1_ctrl_results = []
    yesno_type2_ctrl_results = []

    yesno_human_results = []
    yesno_type1_human_results = []
    yesno_type2_human_results = []

    yesno_golden_results = []
    yesno_type1_golden_results = []
    yesno_type2_golden_results = []

    yesno_selected_results = []
    normal_human_results = []
    normal_human_golden_results = []
    for line in sys.stdin:
        obj = json.loads(line.strip())
        normal_results += build_normal(obj)
        normal_golden_results += build_normal_golden(obj)

        yesno_exp_results += build_yesno_subtype_exp(obj, 0)
        yesno_type1_exp_results += build_yesno_subtype_exp(obj, 1)
        yesno_type2_exp_results += build_yesno_subtype_exp(obj, 2)

        yesno_random_results += build_yesno_random(obj)

        yesno_ctrl_results += build_yesno_subtype_ctrl(obj, 0)
        yesno_type1_ctrl_results += build_yesno_subtype_ctrl(obj, 1)
        yesno_type2_ctrl_results += build_yesno_subtype_ctrl(obj, 2)

        yesno_human_results += build_yesno_subtype_human(obj, 0)
        yesno_type1_human_results += build_yesno_subtype_human(obj, 1)
        yesno_type2_human_results += build_yesno_subtype_human(obj, 2)

        normal_human_results += build_normal_human(obj)
        normal_human_golden_results += build_normal_human_golden(obj)

        yesno_golden_results += build_yesno_subtype_golden(obj, 0)
        yesno_type1_golden_results += build_yesno_subtype_golden(obj, 1)
        yesno_type2_golden_results += build_yesno_subtype_golden(obj, 2)

        yesno_selected_results += build_yesno_selected(obj)

    yesno_ctrl_metric = get_metrics(yesno_ctrl_results,
            yesno_golden_results)
    yesno_type1_ctrl_metric = get_metrics(yesno_type1_ctrl_results,
            yesno_type1_golden_results)
    yesno_type2_ctrl_metric = get_metrics(yesno_type2_ctrl_results,
            yesno_type2_golden_results)

    yesno_exp_metric = get_metrics(yesno_exp_results,
            yesno_golden_results)
    yesno_type1_exp_metric = get_metrics(yesno_type1_exp_results,
            yesno_type1_golden_results)
    yesno_type2_exp_metric = get_metrics(yesno_type2_exp_results,
            yesno_type2_golden_results)

    yesno_random_metric = get_metrics(yesno_random_results,
            yesno_golden_results)

    yesno_human_metric = get_metrics(yesno_human_results,
            yesno_golden_results)
    yesno_type1_human_metric = get_metrics(yesno_type1_human_results,
            yesno_type1_golden_results)
    yesno_type2_human_metric = get_metrics(yesno_type2_human_results,
            yesno_type2_golden_results)

    normal_metric = get_metrics(normal_results, normal_golden_results)
    yesno_selected_metric = get_metrics(yesno_selected_results,
            yesno_golden_results)

    normal_human_metric = get_metrics(normal_human_results,
            normal_human_golden_results)

    print 'ctrl: ', yesno_ctrl_metric
    print 'yesno_type1_ctrl', yesno_type1_ctrl_metric
    print 'yesno_type2_ctrl', yesno_type2_ctrl_metric

    print 'human:', yesno_human_metric
    print 'yesno_type1_human:', yesno_type1_human_metric
    print 'yesno_type2_human:', yesno_type2_human_metric

    print 'random:', yesno_random_metric

    print 'exp:', yesno_exp_metric
    print 'yesno_type1_exp', yesno_type1_exp_metric
    print 'yesno_type2_exp', yesno_type2_exp_metric

    print 'normal: ', normal_metric
    print 'selected: ', yesno_selected_metric
    print 'normal_human2: ', normal_human_metric


if __name__ == '__main__':
    main()
