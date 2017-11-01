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
import logging
import os
import sys
import json
import dataset

from bidaf import BiDAF
from match_lstm import MatchLstm
from dssm import DSSM
from yesno import TypeCls

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
    RANK = 'rank'
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
        self.__prepare()

    def __prepare(self):
        if self.args.algo == Algos.BIDAF:
            self.__create_qa_data()
            self.model = BiDAF(
                         Algos.BIDAF,
                         self.datasets[1].schema,
                         is_infer=self.args.is_infer,
                         vocab_size=self.args.vocab_size,
                         doc_num=self.datasets[1].doc_num,
                         static_emb=(self.args.pre_emb.strip() != ''),
                         emb_dim=self.args.emb_dim)
        elif self.args.algo == Algos.MLSTM:
            self.__create_qa_data()
            self.model = MatchLstm(
                         Algos.MLSTM,
                         self.datasets[1].schema,
                         is_infer=self.args.is_infer,
                         vocab_size=self.args.vocab_size,
                         doc_num=self.datasets[1].doc_num,
                         static_emb=(self.args.pre_emb.strip() != ''),
                         emb_dim=self.args.emb_dim)
        elif self.args.algo == Algos.RANK:
            self.__create_ranking_data()
            self.model = DSSM(
                         Algos.RANK,
                         train_reader.schema,
                         is_infer=self.args.is_infer,
                         vocab_size=self.args.vocab_size,
                         emb_dim=self.args.emb_dim)
        elif self.args.algo == Algos.YESNO:
            self.__create_yesno_data()
            self.model = TypeCls(
                         Algos.YESNO,
                         train_reader.schema,
                         is_infer=self.args.is_infer,
                         vocab_size=self.args.vocab_size,
                         static_emb=(self.args.pre_emb.strip() != ''),
                         doc_num=1,
                         emb_dim=self.args.emb_dim)
        else:
            raise ValueError('Illegal algo: {}'.format(self.args.algo))

    def __create_qa_data(self):
        if self.args.is_infer:
            train_reader = None
        else:
            train_reader = dataset.DuReaderQA(
                           file_name=self.args.trainset,
                           vocab_file=self.args.vocab_file,
                           vocab_size=self.args.vocab_size,
                           max_p_len=self.args.max_p_len,
                           shuffle=(not self.args.is_infer),
                           preload=(not self.args.is_infer))
        test_reader = dataset.DuReaderQA(
                      file_name=self.args.testset,
                      vocab_file=self.args.vocab_file,
                      vocab_size=self.args.vocab_size,
                      max_p_len=self.args.max_p_len,
                      shuffle=False,
                      is_infer=self.args.is_infer,
                      preload=(not self.args.is_infer))
        self.datasets = [train_reader, test_reader]

    def __create_ranking_data(self):
        if self.args.is_infer:
            train_reader = None
        else:
            train_reader = dataset.BaiduNlpRanking(
                           file_name=self.args.trainset,
                           vocab_file=self.args.vocab_file,
                           vocab_size=self.args.vocab_size,
                           keep_raw=False,
                           preload=False)
        test_reader = dataset.BaiduNlpRanking(
                      file_name=self.args.testset,
                      vocab_file=self.args.vocab_file,
                      vocab_size=self.args.vocab_size,
                      is_infer=self.args.is_infer,
                      preload=(not self.args.is_infer))
        self.datasets = [train_reader, test_reader]

    def __create_yesno_data(self):
        if self.args.is_infer:
            train_reader = None
        else:
            train_reader = dataset.BaiduYesNo(
                           file_name=self.args.trainset,
                           vocab_file=self.args.vocab_file,
                           vocab_size=self.args.vocab_size,
                           keep_raw=False,
                           preload=True,
                           shuffle=True)
        test_reader = dataset.BaiduYesNo(
                      file_name=self.args.testset,
                      vocab_file=self.args.vocab_file,
                      vocab_size=self.args.vocab_size,
                      keep_raw=False,
                      is_infer=self.args.is_infer,
                      preload=(not self.args.is_infer),
                      shuffle=False)
        self.datasets = [train_reader, test_reader]


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', help='train dataset')
    parser.add_argument('--testset', help='test dataset')
    parser.add_argument('--test_period', type=int, default=10)
    parser.add_argument('--vocab_file', help='dict')
    parser.add_argument('--batch_size', help='batch size',
                        default=30, type=int)
    parser.add_argument('--num_passes', type=int, default=100)
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
    parser.add_argument('--task', default='train')
    parser.add_argument('--algo', default='bidaf', help='bidaf|mlstm|rank')
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--log_period', default=10, type=int)
    parser.add_argument('--l2', default=2e-4, type=float)
    parser.add_argument('--is_infer', default=False, action='store_true')
    parser.add_argument('--model_file', default='')
    parser.add_argument('--init_from', default='')
    parser.add_argument('--max_p_len', type=int, default=300)

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
    worker.run()


if __name__ == '__main__':
    run()
