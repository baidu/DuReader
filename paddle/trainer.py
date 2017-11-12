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
This module prepares and runs a training process.
"""


import argparse
import numpy as np
import logging

import gzip

import paddle.v2 as paddle
import paddle.v2.evaluator as evaluator
import paddle.v2.optimizer as opt


import os

class Trainer(object):
    """
    The Trainer class prepares and runs a training process.
    """
    def __init__(self,
                 cmd_args,
                 model=None,
                 datasets=None,
                 *args,
                 **kwargs):
        self.logger = logging.getLogger("paddle")
        self.logger.setLevel(logging.INFO)
        self.args = cmd_args
        self.model = model
        self.train_reader = datasets[0]
        self.test_reader = datasets[1]
        self.feeding = datasets[0].feeding
        self.costs = []
        self._prepare()

    def save_model(self, event):
        """
        Saves model. This method is called by the trainer object.

        Args:
            event: a paddle.v2.event object.

        Raises:
            Raises TypeError if the event type is either EndIteration nor
            EndPass.
        """
        if not self.args.save_dir.strip():
            return None
        if isinstance(event, paddle.event.EndIteration):
            file_name = "{}_pass_{:05d}_batch_{:05d}.tar.gz".format(
                        self.model.name,
                        event.pass_id,
                        event.batch_id)
        elif isinstance(event, paddle.event.EndPass):
            file_name = "{}_pass_{:05d}.tar.gz".format(
                        self.model.name,
                        event.pass_id)
        else:
            raise TypeError('Unexpected event type: {}'.format(event))

        with gzip.open(os.path.join(self.args.save_dir, file_name), "w") as f:
            self.parameters.to_tar(f)

        self.logger.info("Saved model to {}".format(file_name))

    def do_test(self, event):
        """
        Do test on devset during training, logs the test result.

        Args:
            event: a paddle.v2.event object.
        """
        test_info = "Test result: Pass={} Batch={} Cost={:.5f} Metrics={}"
        test_result = self.trainer.test(reader=self.test_reader,
                                        feeding=self.feeding)
        batch_id = len(self.costs) if isinstance(event, paddle.event.EndPass) \
                   else event.batch_id

        self.logger.info(test_info.format(event.pass_id,
                                         batch_id,
                                         test_result.cost,
                                         test_result.metrics))

    def stat_params(self):
        """
        Collects and logs parameter statisitics.
        """
        param_names = self.parameters.keys()
        param_info = "name={} shape={} val_mean={:.5f} val_max={:.5f} val_std={:.5f}"
        for p in param_names:
            self.logger.info(param_info.format(p,
                self.parameters.get_shape(p),
                np.absolute(self.parameters.get(p)).mean(),
                self.parameters.get(p).max(),
                self.parameters.get(p).std(),
                ))

    def _event_handler(self, event):
        if isinstance(event, paddle.event.EndIteration):
            self.costs.append(event.cost)
            if event.batch_id and self.args.saving_period \
                    and not event.batch_id % self.args.saving_period:
                # save model
                self.save_model(event)

            if event.batch_id \
                    and self.args.test_period \
                    and not event.batch_id % self.args.test_period:
                # test on devset and log out the test result.
                self.do_test(event)

            if event.batch_id \
                    and self.args.log_period \
                    and not event.batch_id % self.args.log_period:
                info = "Pass={} Batch={} Cost={:.5f} AvgCost={:.5f} Metrics={}"
                self.logger.info(info.format(
                        event.pass_id,
                        event.batch_id,
                        event.cost,
                        sum(self.costs) / len(self.costs),
                        event.metrics))
                self.stat_params()

        if isinstance(event, paddle.event.EndPass):
            self.save_model(event)
            self.do_test(event)
            info = 'Pass={} Batches={} AvgCost={} Metrics={}'
            self.logger.info(info.format(
                event.pass_id,
                len(self.costs),
                sum(self.costs) / len(self.costs),
                event.metrics))
            self.costs = []

    def _prepare(self):
        # prepare reader
        self.train_reader = paddle.batch(
                            reader=self.train_reader.create_reader(),
                            batch_size=self.args.batch_size)
        self.test_reader = paddle.batch(
                           reader=self.test_reader.create_reader(),
                           batch_size=self.args.batch_size)

        # init paddle
        paddle.init(use_gpu=self.args.use_gpu,
                    trainer_count=self.args.trainer_count)

        # create optimizer
        optimizer = paddle.optimizer.RMSProp(
                    learning_rate=self.args.learning_rate,
                    regularization=opt.L2Regularization(rate=self.args.l2))

        # create parameters and trainer
        model_out = self.model()
        if self.args.init_from:
            self.parameters = paddle.parameters.Parameters.from_tar(
                              gzip.open(self.args.init_from, 'r'))
        else:
            self.parameters = paddle.parameters.create(model_out)
        if self.args.pre_emb.strip() != '':
            embeddings = np.loadtxt(self.args.pre_emb,
                    dtype=float)[:self.args.vocab_size]
            self.parameters.set(self.model.name + '.embs', embeddings)
            self.logger.info('init emb from {} to {}'.format(
                self.args.pre_emb,
                self.model.name + '.embs'))

        self.trainer = paddle.trainer.SGD(cost=model_out,
                                          parameters=self.parameters,
                                          update_equation=optimizer)

    def start(self):
        """
        Runs the whole training process.
        """
        self.logger.info("start training...")
        self.trainer.train(reader=self.train_reader,
                           event_handler=self._event_handler,
                           num_passes=self.args.num_passes,
                           feeding=self.feeding)
