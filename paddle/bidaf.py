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
This module implements the BiDAF algorithm described in
https://arxiv.org/abs/1611.01603
"""

import paddle.v2.layer as layer
import paddle.v2.attr as Attr
import paddle.v2.activation as Act
import paddle.v2 as paddle
import paddle.v2.networks as networks

from rc_model import RCModel

class BiDAF(RCModel):
    """
    Implements BiDAF.
    """
    def _get_enc(self, input, type='q'):
        embs = self.get_embs(input)
        enc = networks.bidirectional_lstm(
              input=embs,
              size=self.emb_dim,
              fwd_mat_param_attr=Attr.Param(self.name + '_f_enc_mat.w' + type),
              fwd_bias_param_attr=Attr.Param(self.name + '_f_enc.bias' + type,
                  initial_std=0.),
              fwd_inner_param_attr=Attr.Param(self.name + '_f_enc_inn.w' + type),
              bwd_mat_param_attr=Attr.Param(self.name + '_b_enc_mat.w' + type),
              bwd_bias_param_attr=Attr.Param(self.name + '_b_enc.bias' + type,
                  initial_std=0.),
              bwd_inner_param_attr=Attr.Param(self.name + '_b_enc_inn.w' + type),
              return_seq=True)
        enc_dropped = self.drop_out(enc, drop_rate=0.25)
        return enc_dropped

    def _step_basic(self, h_cur, u):
        expanded_h = layer.expand(input=h_cur, expand_as=u)
        hu = layer.concat(input=[expanded_h, u])
        with layer.mixed(bias_attr=False) as dot_hu:
            dot_hu += layer.dotmul_operator(a=expanded_h, b=u)
        cat_all = layer.concat(input=[hu, dot_hu])
        s = layer.fc(size=1,
                     bias_attr=False,
                     param_attr=Attr.Param(self.name + '.ws'),
                     input=cat_all)
        return s

    def _h_step(self, h_cur, u):
        s = self._step_basic(h_cur, u)
        step_max = layer.pooling(input=s, pooling_type=paddle.pooling.Max())
        return step_max

    def _u_step(self, h_cur, u):
        s = self._step_basic(h_cur, u)
        with layer.mixed(size=1,
                         bias_attr=False,
                         act=Act.SequenceSoftmax()) as h_weights:
            h_weights += layer.identity_projection(s)
        applied_weights = layer.scaling(input=u, weight=h_weights)
        u_ctx = layer.pooling(input=applied_weights,
                              pooling_type=paddle.pooling.Sum())
        return u_ctx

    def _beta(self, h, u_expr, h_expr):
        with layer.mixed(bias_attr=False) as dot_h_u_expr:
            dot_h_u_expr += layer.dotmul_operator(a=h, b=u_expr)
        with layer.mixed(bias_attr=False) as dot_h_h_expr:
            dot_h_h_expr += layer.dotmul_operator(a=h, b=h_expr)
        cat_all = layer.concat(input=[h, u_expr, dot_h_u_expr, dot_h_h_expr])
        return cat_all

    def _attention_flow(self, h, u):
        bs = layer.recurrent_group(
             input=[h, layer.StaticInput(u)],
             step=self._h_step,
             reverse=False)
        b_weights = layer.mixed(act=Act.SequenceSoftmax(),
                    bias_attr=False,
                    input=layer.identity_projection(bs))
        h_step_scaled = layer.scaling(input=h, weight=b_weights)
        h_step = layer.pooling(input=h_step_scaled,
                               pooling_type=paddle.pooling.Sum())
        h_expr = layer.expand(input=h_step, expand_as=h)
        u_expr = layer.recurrent_group(
                 input=[h, layer.StaticInput(u)],
                 step=self._u_step,
                 reverse=False)
        g = self._beta(h, u_expr, h_expr)
        return g

    def network(self):
        """
        Implements the whole network.

        Returns:
            A tuple of LayerOutput objects containing the start and end
            probability distributions respectively.
        """
        self.check_and_create_data()
        self.create_shared_params()
        u = self._get_enc(self.q_ids, type='q')
        m1s = []
        m2s = []
        for p in self.p_ids:
            h = self._get_enc(p, type='q')
            g = self._attention_flow(h, u)
            m1 = networks.bidirectional_lstm(
                 fwd_mat_param_attr=Attr.Param('_f_m1_mat.w'),
                 fwd_bias_param_attr=Attr.Param('_f_m1.bias',
                     initial_std=0.),
                 fwd_inner_param_attr=Attr.Param('_f_m1_inn.w'),
                 bwd_mat_param_attr=Attr.Param('_b_m1_mat.w'),
                 bwd_bias_param_attr=Attr.Param('_b_m1.bias',
                     initial_std=0.),
                 bwd_inner_param_attr=Attr.Param('_b_m1_inn.w'),
                 input=g,
                 size=self.emb_dim,
                 return_seq=True)
            m1_dropped = self.drop_out(m1, drop_rate=0.)
            cat_g_m1 = layer.concat(input=[g, m1_dropped])

            m2 = networks.bidirectional_lstm(
                 fwd_mat_param_attr=Attr.Param('_f_m2_mat.w'),
                 fwd_bias_param_attr=Attr.Param('_f_m2.bias',
                     initial_std=0.),
                 fwd_inner_param_attr=Attr.Param('_f_m2_inn.w'),
                 bwd_mat_param_attr=Attr.Param('_b_m2_mat.w'),
                 bwd_bias_param_attr=Attr.Param('_b_m2.bias',
                     initial_std=0.),
                 bwd_inner_param_attr=Attr.Param('_b_m2_inn.w'),
                 input=m1,
                 size=self.emb_dim,
                 return_seq=True)
            m2_dropped = self.drop_out(m2, drop_rate=0.)
            cat_g_m2 = layer.concat(input=[g, m2_dropped])
            m1s.append(cat_g_m1)
            m2s.append(cat_g_m2)

        all_m1 = reduce(lambda x, y: layer.seq_concat(a=x, b=y), m1s)
        all_m2 = reduce(lambda x, y: layer.seq_concat(a=x, b=y), m2s)

        start = self.decode('start', all_m1)
        end = self.decode('end', all_m2)
        return start, end
