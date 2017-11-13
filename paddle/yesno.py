# -*- coding:utf-8 -*-
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
This module implements an opinion classification model to classify a
question answer pair into 3 categories: Yes(positive opinion),
No(negative opinion), Depends(depends on conditions).
"""

import logging
import json
import sys
import paddle.v2.layer as layer
import paddle.v2.attr as Attr
import paddle.v2.activation as Act
import paddle.v2.data_type as data_type
import paddle.v2 as paddle

from match_lstm import MatchLstm

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


class OpinionClassifier(MatchLstm):
    """
    Implements a opinion classifer model, replace the pointer net of MatchLstm
    with a one-layered classifer. Inherits from `MatchLstm`.
    """
    def __init__(self, name, inputs, *args, **kwargs):
        self.name = name
        self.inputs = inputs
        self.emb_dim = kwargs['emb_dim']
        self.vocab_size = kwargs['vocab_size']
        self.is_infer = kwargs['is_infer']
        self.label_dim = 3
        self.static_emb = kwargs.get('static_emb', False)
        self.labels = ['Yes', 'No', 'Depends']
        self.label_dict = {v: idx for idx, v in enumerate(self.labels)}
        super(OpinionClassifier, self).__init__(name, inputs, *args, **kwargs)

    def check_and_create_data(self):
        """
        Checks if the input data is legal and creates the data layers
        according to the input fields.
        """
        if self.is_infer:
            expected = ['q_ids', 'a_ids']
            if len(self.inputs) < 2:
                raise ValueError('''Input schema: expected vs given:
                         {} vs {}'''.format(expected, self.inputs))
        else:
            expected = ['q_ids', 'a_ids', 'label']
            if len(self.inputs) < 3:
                raise ValueError('''Input schema: expected vs given:
                         {} vs {}'''.format(expected, self.inputs))
            self.label = layer.data(name=self.inputs[2],
                               type=data_type.integer_value(4))

        self.q_ids = layer.data(
                name=self.inputs[0],
                type=data_type.integer_value_sequence(self.vocab_size))

        self.a_ids = layer.data(
                name=self.inputs[1],
                type=data_type.integer_value_sequence(self.vocab_size))

    def network(self):
        """
        Implements the detail of the model.
        """
        self.check_and_create_data()
        self.create_shared_params()
        q_enc = self.get_enc(self.q_ids, type='q')
        a_enc = self.get_enc(self.a_ids, type='q')

        q_proj_left = layer.fc(size=self.emb_dim * 2,
                bias_attr=False,
                param_attr=Attr.Param(self.name + '_left.wq'),
                input=q_enc)
        q_proj_right = layer.fc(size=self.emb_dim * 2,
                bias_attr=False,
                param_attr=Attr.Param(self.name + '_right.wq'),
                input=q_enc)
        left_match = self.recurrent_group(self.name + '_left',
                [layer.StaticInput(q_enc),
                    layer.StaticInput(q_proj_left), a_enc],
                reverse=False)
        right_match = self.recurrent_group(self.name + '_right',
                [layer.StaticInput(q_enc),
                    layer.StaticInput(q_proj_right), a_enc],
                reverse=True)
        match_seq = layer.concat(input=[left_match, right_match])
        with layer.mixed(size=match_seq.size,
                act=Act.Identity(),
                layer_attr=Attr.ExtraLayerAttribute(drop_rate=0.2),
                bias_attr=False) as dropped:
            dropped += layer.identity_projection(match_seq)
        match_result = layer.pooling(input=dropped,
                pooling_type=paddle.pooling.Max())
        cls = layer.fc(input=match_result,
                act=Act.Softmax(),
                size=self.label_dim)
        return cls

    def train(self):
        """
        Trains the model.
        """
        cls = self.network()
        loss = layer.cross_entropy_cost(input=cls,
                label=self.label,
                name=self.name + '_cost')
        evaluator_0 = paddle.evaluator.precision_recall(
                input=cls, name='label0', label=self.label, positive_label=0)
        evaluator_1 = paddle.evaluator.precision_recall(
                input=cls, name='label1', label=self.label, positive_label=1)
        evaluator_2 = paddle.evaluator.precision_recall(
                input=cls, name='label2', label=self.label, positive_label=2)
        evaluator_all = paddle.evaluator.precision_recall(
                input=cls, name='label_all', label=self.label)
        return loss

    def infer(self):
        """
        Infers with the trained models.
        """
        cls = self.network()
        return cls

    def evaluate(self,
            infer_file,
            ret=None,
            from_file=False):
        """
        Processes and evaluates the inferred result of one batch.
        """
        results, stored_objs = self._parse_infer_ret(ret)
        with open(infer_file, 'w') as inf:
            for obj in stored_objs:
                sorted_ans = sorted(obj['yesno_answers'], key=lambda x: x[0])
                obj['yesno_answers'] = [x[1] for x in sorted_ans]
                print >> inf, json.dumps(obj, ensure_ascii=False).encode('utf8')
        self._calc_pr(results)

    def _parse_infer_ret(self, infer_ret):
        results = []
        stored_objs = []
        if not infer_ret:
            return results, stored_objs
        for batch_input, batch_output in infer_ret:
            pred_labels = map(int, batch_output[0].argmax(axis=1))
            for ins, pred in zip(batch_input, pred_labels):
                obj = ins[-1]
                obj['yesno_answers'] = [(obj['answer_idx'], self.labels[pred])]
                stored_objs.append(obj)
        return results, self._merge_objs(stored_objs)

    def _merge_objs(self, obj_list):
        merged_objs = []
        last_id = None

        for obj in obj_list:
            qid = obj['question_id']
            if last_id != qid:
                merged_objs.append(obj)
                last_id = qid
                continue
            merged_objs[-1]['yesno_answers'].append(obj['yesno_answers'][0])

        return merged_objs

    def _calc_pr(self, results):
        # {label: [true, pred, real]}
        labels = {}
        if len(results) > 0:
            acc = 1.0 * len([(x, y) for x, y in results if x == y]) / len(results)
        else:
            acc = 0.0
        for label, pred in results:
            labels[label] = labels.get(label, [0, 0, 0])
            labels[label][2] += 1
            if label == pred:
                labels[label][0] += 1
            labels[pred] = labels.get(pred, [0, 0, 0])
            labels[pred][1] += 1

        eval_result = {}
        for label, counts in labels.items():
            true, pred, real = counts
            recall = 1.0 * true / real if real > 0 else 0.0
            precision = 1.0 * true / pred if pred > 0 else 0.0
            f1 = 2 * recall * precision / (recall + precision) \
                    if recall + precision > 0 \
                    else 0.0
            eval_result['label_{}_recall'.format(label)] = recall
            eval_result['label_{}_precision'.format(label)] = precision
            eval_result['label_{}_f1'.format(label)] = f1
        eval_result['accuracy'] = acc
        logger.info('eval resulsts: {}'.format(
            ' '.join(["{}={}".format(x, y) for x, y in eval_result.items()])))

    def __call__(self):
        if self.is_infer:
            return self.infer()
        return self.train()
