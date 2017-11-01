# -*- coding:utf8 -*-
###############################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
###############################################################################
"""
This module implements the BiDAF algorithm described in
https://arxiv.org/abs/1611.01603

Authors: liuyuan(liuyuan04@baidu.com)
Date: 2017/09/20 12:00:00
"""
import argparse
import collections
import numpy as np
import logging
import itertools

import gzip

import paddle.v2 as paddle
import paddle.v2.evaluator as evaluator
import paddle.v2.optimizer as opt

import os

class Inferer(object):
    """
    The Inferer class prepares and runs the inferring process.
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
        self.test_reader = datasets[1]
        self.feeding = datasets[1].feeding
        self.costs = []
        self.__prepare()

    def __prepare(self):
        # prepare reader
        self.test_reader = paddle.batch(
                           reader=self.test_reader.create_reader(),
                           batch_size=self.args.batch_size)

        # init paddle
        paddle.init(use_gpu=self.args.use_gpu,
                    trainer_count=self.args.trainer_count)

        # create parameters and trainer
        model_out = self.model()
        out_names = [x.name for x in model_out] \
                if isinstance(model_out, collections.Iterable) \
                else model_out.name
        self.logger.info("out type: {}".format(model_out))
        self.logger.info("out names: {}".format(out_names))
        try:
            self.parameters = paddle.parameters.Parameters.from_tar(
                              gzip.open(self.args.model_file, 'r'))
        except IOError:
            raise IOError('can not find: {}'.format(self.args.model_file))
        self.inferer = paddle.inference.Inference(
                       output_layer=model_out,
                       parameters=self.parameters)

    def get_infer_file(self):
        """
        Decides the infer file name. An infer file is a file storing the parsed
        inferring result for later analysis.

        Returns:
            is_exist: True if the file is already exists, otherwise False.
            infer_file: the infer file name.
        """
        paths = self.args.model_file.split('/')
        model_name = paths[-1]
        paths[-2] = 'infer'
        paths[-1] = model_name + '.json'
        infer_file = '/'.join(paths)
        is_exist = os.path.isfile(infer_file)
        return is_exist, infer_file

    def run(self):
        """
        Runs the whole inferring process.
        """
        self.logger.info("start inferring...")
        is_exist, infer_file = self.get_infer_file()
        if is_exist:
            self.logger.info('file {} exists, skipping.'.format(infer_file))
            self.model.evaluate(infer_file, from_file=True)
            return None

        all_res = []
        for i, batch in enumerate(self.test_reader()):
            self.logger.info('Inferring batch...{}'.format(i))
            res = self.inferer.infer(input=batch,
                                     flatten_result=False,
                                     feeding=self.feeding)
            all_res.append((batch, res))
        self.model.evaluate(infer_file, ret=all_res)
