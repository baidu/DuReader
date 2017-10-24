# -*- coding:utf8 -*-
###############################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
###############################################################################
"""
This module implements the shared functions of Match-LSTM and BiDAF algorithm

Authors: liuyuan(liuyuan04@baidu.com)
Data: 2017/09/20 12:00:00
"""
import paddle.v2.layer as layer
import paddle.v2.attr as Attr
import paddle.v2.activation as Act
import paddle.v2.data_type as data_type
import paddle.v2 as paddle


class QAModel(object):
    def __init__(self, name, inputs, *args, **kwargs):
        self.name = name
        self.inputs = inputs
        self.emb_dim = kwargs['emb_dim']
        self.vocab_size = kwargs['vocab_size']
        self.is_infer = kwargs['is_infer']
        self.doc_num = 1

        # check and create input data
        if self.is_infer:
            expected = ['q_ids', 'p_ids', 'para_length',
                        '[start_label, end_label, ...]']
            if len(inputs) < 3:
                raise ValueError('''Input schema: expected vs given:
                         {} vs {}'''.format(expected, self.inputs))
        else:
            expected = ['q_ids', 'p_ids', 'para_length',
                        'start_label', 'end_label', '...']
            if len(inputs) < 5:
                raise ValueError('''Input schema: expected vs given:
                         {} vs {}'''.format(expected, self.inputs))
            self.start_labels = [
                    layer.data(name=inputs[i],
                               type=data_type.dense_vector_sequence(1))
                    for i in range(1+2*self.doc_num, 1+3*self.doc_num)
                    ]
            self.start_label = reduce(
                    lambda x, y: layer.seq_concat(a=x, b=y),
                    self.start_labels)

            self.end_labels = [
                    layer.data(name=inputs[i],
                               type=data_type.dense_vector_sequence(1))
                    for i in range(1+3*self.doc_num, 1+4*self.doc_num)
                    ]

            self.end_label = reduce(
                    lambda x, y: layer.seq_concat(a=x, b=y),
                    self.end_labels)

        self.q_ids = layer.data(
                     name=inputs[0],
                     type=data_type.integer_value_sequence(self.vocab_size))

        self.p_ids = [
                layer.data(
                    name=inputs[i],
                    type=data_type.integer_value_sequence(self.vocab_size))
                for i in range(1, 1+self.doc_num)
                ]

        self.para_lens = [
                layer.data(
                    name=inputs[i],
                    type=data_type.dense_vector_sequence(1))
                for i in range(1+self.doc_num, 1+2*self.doc_num)
                ]

        self.para_len = reduce(
                lambda x, y: layer.seq_concat(a=x, b=y),
                self.para_lens)

        self.create_shared_params()

    def create_shared_params(self):
        """
        Creates parameters shared by multiple layers.
        """
        # create parameters
        # embedding parameter, shared by question and paragraph.
        is_static = True if self.pre_emb.strip() != '' else False
        self.emb_param = Attr.Param(name=self.name + '.embs',
                                    is_static=is_static)

    def get_embs(self, input):
        """Get embeddings of token sequence.

        Args:
            input: input sequence of tokens.

        Returns:
            An embedding layer.

        """
        embs = layer.embedding(input=input,
                               size=self.emb_dim,
                               param_attr=self.emb_param)
        return embs

    def network(self):
        """
        The implementation of the algorithm.

        Returns:
            start: A LayerOutput object of start postion sequence.
            end: A layerOutput object of end position sequence.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def get_loss(self, start_prob, end_prob, start_label, end_label):
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
        start, end = self.network()
        cost = self.get_loss(start, end, self.start_label, self.end_label)
        return cost

    def infer(self):
        start, end = self.network()
        start_end = layer.seq_concat(name='start_end', a=start, b=end)
        return start_end, self.para_len

    def decode(self, name, input, fix_latent_param=False):
        latent_param = None
        if fix_latent_param:
            latent_param = Attr.Param(self.name + '_pointer_lat.w')
        latent = layer.fc(size=input.size / 2,
                          input=input,
                          act=Act.Tanh(),
                          bias_attr=False,
                          param_attr=latent_param)
        probs = layer.fc(
                name=name,
                size=1,
                input=latent,
                act=Act.SequenceSoftmax())
        return probs

    def __call__(self):
        if self.is_infer:
            return self.infer()
        return self.train()
