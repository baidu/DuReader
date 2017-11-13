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
Prepares and runs the whole system.
"""

import argparse
import logging
import os
import sys
import json
import dataset

from bidaf import BiDAF
from match_lstm import MatchLstm
from yesno import OpinionClassifier

from trainer import Trainer
from inferer import Inferer

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


class Algos(object):
    """
    Enumerates algorithms that the system supports.
    """
    BIDAF = 'bidaf'
    MLSTM = 'mlstm'
    YESNO = 'yesno'


class Env(object):
    """
    Prepares data and model.
    """
    def __init__(self, args):
        self.args = args
        if self.args.is_infer:
            logger.info("infer mode")
        else:
            logger.info("train mode")
        self._prepare()

    def _prepare(self):
        if self.args.algo == Algos.BIDAF:
            self._create_qa_data()
            self.model = BiDAF(
                         Algos.BIDAF,
                         self.datasets[1].schema,
                         is_infer=self.args.is_infer,
                         vocab_size=self.args.vocab_size,
                         doc_num=self.datasets[1].doc_num,
                         static_emb=(self.args.pre_emb.strip() != ''),
                         emb_dim=self.args.emb_dim,
                         max_a_len=self.args.max_a_len)
        elif self.args.algo == Algos.MLSTM:
            self._create_qa_data()
            self.model = MatchLstm(
                         Algos.MLSTM,
                         self.datasets[1].schema,
                         is_infer=self.args.is_infer,
                         vocab_size=self.args.vocab_size,
                         doc_num=self.datasets[1].doc_num,
                         static_emb=(self.args.pre_emb.strip() != ''),
                         emb_dim=self.args.emb_dim,
                         max_a_len=self.args.max_a_len)
        elif self.args.algo == Algos.YESNO:
            self._create_yesno_data()
            self.model = OpinionClassifier(
                         Algos.YESNO,
                         self.datasets[1].schema,
                         is_infer=self.args.is_infer,
                         vocab_size=self.args.vocab_size,
                         static_emb=(self.args.pre_emb.strip() != ''),
                         doc_num=1,
                         emb_dim=self.args.emb_dim)
        else:
            raise ValueError('Illegal algo: {}'.format(self.args.algo))

    def _create_qa_data(self):
        if self.args.is_infer:
            train_reader = None
        else:
            train_reader = dataset.DuReaderQA(
                           file_names=self.args.trainset,
                           vocab_file=self.args.vocab_file,
                           vocab_size=self.args.vocab_size,
                           max_p_len=self.args.max_p_len,
                           shuffle=(not self.args.is_infer),
                           preload=(not self.args.is_infer))
        test_reader = dataset.DuReaderQA(
                      file_names=self.args.testset,
                      vocab_file=self.args.vocab_file,
                      vocab_size=self.args.vocab_size,
                      max_p_len=self.args.max_p_len,
                      shuffle=False,
                      is_infer=self.args.is_infer,
                      preload=(not self.args.is_infer))
        self.datasets = [train_reader, test_reader]

    def _create_yesno_data(self):
        if self.args.is_infer:
            train_reader = None
        else:
            train_reader = dataset.DuReaderYesNo(
                           file_names=self.args.trainset,
                           vocab_file=self.args.vocab_file,
                           vocab_size=self.args.vocab_size,
                           preload=True,
                           shuffle=True)
        test_reader = dataset.DuReaderYesNo(
                      file_names=self.args.testset,
                      vocab_file=self.args.vocab_file,
                      vocab_size=self.args.vocab_size,
                      is_infer=self.args.is_infer,
                      preload=(not self.args.is_infer),
                      shuffle=False)
        self.datasets = [train_reader, test_reader]


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', nargs='+', help='train dataset')
    parser.add_argument('--testset', nargs='+', help='test dataset')
    parser.add_argument('--test_period', type=int, default=10)
    parser.add_argument('--vocab_file', help='dict')
    parser.add_argument('--batch_size', help='batch size',
                        default=32, type=int)
    parser.add_argument('--num_passes', type=int, default=30)
    parser.add_argument('--emb_dim', help='dim of word vector',
                        default=300, type=int)
    parser.add_argument('--vocab_size', help='vocab size',
                        default=-1, type=int)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--save_dir', help='save dir', default='')
    parser.add_argument('--trainer_count', type=int, default=1)
    parser.add_argument('--saving_period', type=int, default=100)
    parser.add_argument('--pre_emb', default='')
    parser.add_argument('--algo', default='bidaf', help='bidaf|mlstm|yesno')
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--log_period', default=10, type=int)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--is_infer', default=False, action='store_true')
    parser.add_argument('--model_file', default='')
    parser.add_argument('--init_from', default='')
    parser.add_argument('--max_p_len', type=int, default=500)
    parser.add_argument('--max_a_len', type=int, default=200)

    args = parser.parse_args()
    return args


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()
    logger.info('Args are: {}'.format(args))
    env = Env(args)
    model = env.model
    datasets = env.datasets
    worker = Trainer(args, model=model, datasets=datasets) \
             if not args.is_infer else \
             Inferer(args, model=model, datasets=datasets)
    worker.start()


if __name__ == '__main__':
    run()
