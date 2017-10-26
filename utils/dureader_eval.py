# -*- coding:utf8 -*-
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
import json
import sys

from bleu_metric.bleu import Bleu
from rouge_metric.rouge import Rouge


def read_file(file_name):
    """
    Args:
        file_name(str): The file name of the file containing precdict results.

    Input Example:
        A json object that at least covers the following fields:

        {"query_id", "5werfsiuerjs",
        "answer": ["God Father"],
        "answer_neg": ["No, it's not."],
        "answer_pos": ["Yes, it is."],
        "answer_dep": ["If you want this, then yes, otherwise no"],
        "answer_none": ["I have no idea."],
        "entities": ["God Father", "Taxi Driver"]}

        - query_id: A uniq id of query.
        - answer: Answer predicted by your model.
        - answer_neg: If query_type is "YES_NO", and your model finds
          an answer with negative opinion, the answer should be filled in
          this field. By default the value should be [].
        - answer_pos: Like `answer_neg`, if the answer expresses positive
          opinion, the answer should be filled in this field. By default
          the value should be [].
        - answer_dep: Like `answer_neg`, if the answer expresses an opinion
          that depends on various conditions, the answer should be filled in
          this field. By default the value should be [].
        - answer_none: If query_type is "YES_NO" and the model predicts an
          answer that have no obvious opinion, the answer should be filled in
          this field. By default the value should be [].
        - entities: If query_type is ENTITY, and the model finds entities
          in the predicted answer, the entities should be in this list.

    Returns:
        7 dicts:
        1. answers for questions of type DESCRIPTION;
        2. answers for questions of type YES_NO;
        3. the rest four dicts respectively contain answers for
        quesitons of type YES_NO.Y, YESNO.N, YES_NO.D, YES_NO.NONE
        4. entity dict contains tagged entities for questions of type
        ENTITY.

        The dicts are like:

            >>> {'query_id': ["answer string"]}
    """
    basic_answer_dict = {}
    yes_no_answer_dict = {}
    pos_answer_dict = {}
    neg_answer_dict = {}
    dep_answer_dict = {}
    none_answer_dict = {}
    entity_dict = {}
    with open(file_name, 'r') as fn:
        for line in fn:
            obj = json.loads(line.strip())
            if not check(obj):
                continue
            q_id = obj['query_id']
            q_id_pos = q_id + '_pos'
            q_id_neg = q_id + '_neg'
            q_id_dep = q_id + '_dep'
            q_id_none = q_id + '_none'
            basic_answer_dict[q_id] = basic_answer_dict.get(q_id, [])
            yes_no_answer_dict[q_id_pos] = yes_no_answer_dict.get(q_id_pos, [])
            yes_no_answer_dict[q_id_neg] = yes_no_answer_dict.get(q_id_neg, [])
            yes_no_answer_dict[q_id_dep] = yes_no_answer_dict.get(q_id_dep, [])
            yes_no_answer_dict[q_id_none] = yes_no_answer_dict.get(q_id_none, [])
            entity_dict[q_id] = entity_dict.get(q_id, [])
            basic_answer_dict[q_id] += obj['answer']
            yes_no_answer_dict[q_id_pos] += obj['answer_pos']
            yes_no_answer_dict[q_id_neg] += obj['answer_neg']
            yes_no_answer_dict[q_id_dep] += obj['answer_dep']
            yes_no_answer_dict[q_id_none] += obj['answer_none']
            entity_dict[q_id] += obj['entities']
    results = [basic_answer_dict, yes_no_answer_dict, pos_answer_dict,
            neg_answer_dict, dep_answer_dict, none_answer_dict,
            entit_dict]
    return results


def compute_bleu_rouge(pred_dict, ref_dict, bleu_order):
    """
    Compute bleu and rouge scores.
    """
    scores = {}
    bleu_scores, _ = Blue(bleu_order).compute_score(ref_dict, pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['bleu_%d' % (i + 1)] = bleu_score
    rouge_score, _ = Rouge().compute_score(ref_dict, pred_dict)
    scores['rouge_l'] = rouge_score
    return scores


def compute_prf(pred_dict, ref_dict):
    """
    Compute precision recall and f1-score.
    """
    pass


def main(pred_file, ref_file):
    """
    Do evaluation.
    """
    pred_list = read_file(pred_file)
    ref_list = read_file(ref_file)
