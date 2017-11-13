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
This module implements the basic common functions of the Match-LSTM and BiDAF
networks.
"""

import copy
import math
import json
import logging
import paddle.v2.layer as layer
import paddle.v2.attr as Attr
import paddle.v2.activation as Act
import paddle.v2.data_type as data_type
import paddle.v2 as paddle

from utils import compute_bleu_rouge
from utils import normalize

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)

class RCModel(object):
    """
    This is the base class of Match-LSTM and BiDAF models.
    """
    def __init__(self, name, inputs, *args, **kwargs):
        self.name = name
        self.inputs = inputs
        self.emb_dim = kwargs['emb_dim']
        self.vocab_size = kwargs['vocab_size']
        self.is_infer = kwargs['is_infer']
        self.doc_num = kwargs.get('doc_num', 5)
        self.static_emb = kwargs.get('static_emb', False)
        self.max_a_len = kwargs.get('max_a_len', 200)

    def check_and_create_data(self):
        """
        Checks if the input data is legal and creates the data layers
        according to the input fields.
        """
        if self.is_infer:
            expected = ['q_ids', 'p_ids', 'para_length',
                        '[start_label, end_label, ...]']
            if len(self.inputs) < 2 * self.doc_num + 1:
                raise ValueError(r'''Input schema: expected vs given:
                         {} vs {}'''.format(expected, self.inputs))
        else:
            expected = ['q_ids', 'p_ids', 'para_length',
                        'start_label', 'end_label', '...']
            if len(self.inputs) < 4 * self.doc_num + 1:
                raise ValueError(r'''Input schema: expected vs given:
                         {} vs {}'''.format(expected, self.inputs))
            self.start_labels = []
            for i in range(1 + 2 * self.doc_num, 1 + 3 * self.doc_num):
                self.start_labels.append(
                        layer.data(name=self.inputs[i],
                            type=data_type.dense_vector_sequence(1)))
            self.start_label = reduce(
                    lambda x, y: layer.seq_concat(a=x, b=y),
                    self.start_labels)
            self.end_labels = []
            for i in range(1 + 3 * self.doc_num, 1 + 4 * self.doc_num):
                self.end_labels.append(
                        layer.data(name=self.inputs[i],
                            type=data_type.dense_vector_sequence(1)))
            self.end_label = reduce(
                    lambda x, y: layer.seq_concat(a=x, b=y),
                    self.end_labels)
        self.q_ids = layer.data(
                name=self.inputs[0],
                type=data_type.integer_value_sequence(self.vocab_size))
        self.p_ids = []
        for i in range(1, 1 + self.doc_num):
            self.p_ids.append(
                    layer.data(name=self.inputs[i],
                        type=data_type.integer_value_sequence(self.vocab_size)))
        self.para_lens = []
        for i in range(1 + self.doc_num, 1 + 2 * self.doc_num):
            self.para_lens.append(
                    layer.data(name=self.inputs[i],
                        type=data_type.dense_vector_sequence(1)))
        self.para_len = reduce(lambda x, y: layer.seq_concat(a=x, b=y),
                self.para_lens)

    def create_shared_params(self):
        """
        Creates parameter objects that shared by multiple layers.
        """
        # embedding parameter, shared by question and paragraph.
        self.emb_param = Attr.Param(name=self.name + '.embs',
                                    is_static=self.static_emb,
                                    initial_std=math.sqrt(1. / self.emb_dim))

    def get_embs(self, input):
        """
        Get embeddings of token sequence.
        Args:
            - input: input sequence of tokens. Should be of type
                     paddle.v2.data_type.integer_value_sequence
        Returns:
            The sequence of embeddings.
        """
        embs = layer.embedding(input=input,
                               size=self.emb_dim,
                               param_attr=self.emb_param)
        return embs

    def network(self):
        """
        Implements the detail of the model. Should be implemented by subclasses.
        """
        raise NotImplementedError

    def get_loss(self, start_prob, end_prob, start_label, end_label):
        """
        Compute the loss: $l_{\theta} = -logP(start)\cdotP(end|start)$

        Returns:
            A LayerOutput object containing loss.
        """
        probs = layer.seq_concat(a=start_prob, b=end_prob)
        labels = layer.seq_concat(a=start_label, b=end_label)

        log_probs = layer.mixed(
                    size=probs.size,
                    act=Act.Log(),
                    bias_attr=False,
                    input=paddle.layer.identity_projection(probs))

        neg_log_probs = layer.slope_intercept(
                        input=log_probs,
                        slope=-1,
                        intercept=0)

        loss = paddle.layer.mixed(
               size=1,
               input=paddle.layer.dotmul_operator(a=neg_log_probs, b=labels))

        sum_val = paddle.layer.pooling(input=loss,
                                       pooling_type=paddle.pooling.Sum())
        cost = paddle.layer.sum_cost(input=sum_val)
        return cost

    def train(self):
        """
        The training interface.

        Returns:
            A LayerOutput object containing loss.
        """
        start, end = self.network()
        cost = self.get_loss(start, end, self.start_label, self.end_label)
        return cost

    def infer(self):
        """
        The inferring interface.

        Returns:
            start_end: A sequence of concatenated start and end probabilities.
            para_len: A sequence of the lengths of every paragraph, which is
                      used for parse the inferring output.
        """
        start, end = self.network()
        start_end = layer.seq_concat(name='start_end', a=start, b=end)
        return start_end, self.para_len

    def decode(self, name, input):
        """
        Implements the answer pointer part of the model.

        Args:
            name: name prefix of the layers defined in this method.
            input: the encoding of the paragraph.

        Returns:
            A probability distribution over temporal axis.
        """
        latent = layer.fc(size=input.size / 2,
                          input=input,
                          act=Act.Tanh(),
                          bias_attr=False)
        probs = layer.fc(
                name=name,
                size=1,
                input=latent,
                act=Act.SequenceSoftmax())
        return probs

    def _search_boundry(self, start_probs, end_probs):
        assert len(start_probs) == len(end_probs)
        boundries = []
        for start_idx, start_prob in enumerate(start_probs):
            max_idx = min(start_idx + self.max_a_len + 1, len(end_probs))
            for end_idx in range(start_idx, max_idx):
                end_prob = end_probs[end_idx]
                boundries.append(
                    ((start_idx, end_idx), start_prob * end_prob))
        max_boundry = sorted(boundries, key=lambda x: x[1], reverse=True)[0][0]
        return max_boundry

    def _parse_infer_ret(self, infer_ret):
        pred_list = []
        ref_list = []
        objs = []
        ins_cnt = 0
        for batch_input, batch_output in infer_ret:
            lens, probs = [x.flatten() for x in batch_output]
            len_sum = int(sum(lens))
            assert len(probs) == 2 * len_sum
            idx_len = 0
            idx_prob = 0

            for ins in batch_input:
                ins = ins[-1]
                len_slice = lens[idx_len:idx_len + self.doc_num]
                prob_len = int(sum(len_slice))
                start_prob_slice = probs[idx_prob:idx_prob + prob_len]
                end_prob_slice = probs[idx_prob + prob_len:idx_prob + 2 * prob_len]
                start_idx, end_idx = self._search_boundry(start_prob_slice,
                        end_prob_slice)
                pred_tokens = [] if start_idx > end_idx \
                        else ins['tokens'][start_idx:end_idx + 1]

                pred = [' '.join(pred_tokens)]
                ref = [' '.join(s) for s in ins['answers_ref']]

                idx_len += self.doc_num
                idx_prob += prob_len * 2
                pred_obj = {ins['question_id']: pred}
                ref_obj = {ins['question_id']: ref}
                stored_obj = copy.deepcopy(ins)
                stored_obj['answers'] = pred
                objs.append(stored_obj)
                pred_list.append(pred_obj)
                ref_list.append(ref_obj)
                ins_cnt += 1
        return ref_list, pred_list, objs

    def _read_list(self, infer_file):
        ref_list = []
        pred_list = []
        with open(infer_file, 'r') as inf:
            for line in inf:
                obj = json.loads(line.strip())
                ref_obj = {obj['question_id']: obj['answers_ref']}
                pred_obj = {obj['question_id']: obj['answers']}
                ref_list.append(ref_obj)
                pred_list.append(pred_obj)
        return ref_list, pred_list

    def drop_out(self, input, drop_rate=0.5):
        """
        Implements drop out.

        Args:
            input: the LayerOutput needs to apply drop out.
            drop_rate: drop out rate.

        Returns:
            The layer output after applying drop out.
        """
        with layer.mixed(
                layer_attr=Attr.ExtraLayerAttribute(
                    drop_rate=drop_rate),
                bias_attr=False) as dropped:
            dropped += layer.identity_projection(input)
        return dropped

    def fusion_layer(self, input1, input2):
        """
        Combine input1 and input2 by concat(input1 .* input2, input1 - input2,
        input1, input2)
        """
        # fusion layer
        neg_input2 = layer.slope_intercept(input=input2,
                slope=-1.0,
                intercept=0.0)
        diff1 = layer.addto(input=[input1, neg_input2],
                act=Act.Identity(),
                bias_attr=False)
        diff2 = layer.mixed(bias_attr=False,
                input=layer.dotmul_operator(a=input1, b=input2))

        fused = layer.concat(input=[input1, input2, diff1, diff2])
        return fused

    def evaluate(self,
            infer_file,
            ret=None,
            from_file=False):
        """
        Processes and evaluates the inferred result.

        Args:
            infer_file: A file name to store or read from the inferred results.
            ret: The information returned by the inferring operation, which
                 contains the batch-level input and the the batch-level
                 inferring result.
            from_file: If True, the time consuming inferring process will be
                       skipped, and this method takes the content of infer_file
                       as input for evaluation. If False, this method takes
                       the ret as input for evaluation.

        """
        def _merge_and_normalize(obj_list):
            ret = {}
            for obj in obj_list:
                normalized = {k: normalize(v) for k, v in obj.items()}
                ret.update(normalized)
            return ret

        pred_list = []
        ref_list = []
        objs = []

        if from_file:
            ref_list, pred_list = self._read_list(infer_file)
        else:
            ref_list, pred_list, objs = self._parse_infer_ret(ret)
            with open(infer_file, 'w') as of:
                for o in objs:
                    print >> of, json.dumps(o, ensure_ascii=False).encode('utf8')
        metrics = compute_bleu_rouge(
                _merge_and_normalize(pred_list),
                _merge_and_normalize(ref_list))
        res_str = '{} {}'.format(infer_file,
                ' '.join('{}={}'.format(k, v) for k, v in metrics.items()))
        logger.info(res_str)

    def __call__(self):
        if self.is_infer:
            return self.infer()
        return self.train()
