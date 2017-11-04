# -*- coding:utf8 -*-
###############################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
###############################################################################
"""
This module prepares and runs the whole system.

Authors: Yizhong Wang(wangyizhong01@baidu.com)
Date: 2017/09/20 12:00:00
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import BRCDataset
from vocab import Vocab
from rc_model import RCModel


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers with trained model')
    parser.add_argument('--task', type=str, default='both',
                        help='reading comprehension on search/zhidao/both dataset')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=10,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('Checking the data for {} task...'.format(args.task))
    for suffix in ['train.json', 'dev.json', 'test.json']:
        data_path = os.path.join(args.brc_dir, args.task + '.' + suffix)
        assert os.path.exists(data_path), '{} file does not exist.'
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    brc_data = BRCDataset(args.brc_dir, args.task, args.max_p_num, args.max_p_len, args.max_q_len)
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))

    logger.info('Assigning embeddings...')
    vocab.randomly_init_embeddings(args.embed_size)

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, args.task + '.' + 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, args.task + '.' + 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.brc_dir, args.task, args.max_p_num, args.max_p_len, args.max_q_len)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')
    rc_model = RCModel(vocab, args)
    logger.info('Training the model...')
    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   save_prefix=args.task + '.' + args.algo,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def predict(args):
    """
    predicts answers for dev set and test set
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, args.task + '.' + 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.brc_dir, args.task,
                          args.max_p_num, args.max_p_len, args.max_q_len, train=False)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.task + '.' + args.algo)
    logger.info('Predicting answers for dev set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(dev_batches,
                      result_dir=args.result_dir, result_prefix=args.task + '.dev.predicted')
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix=args.task + '.test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()
