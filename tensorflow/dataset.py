# -*- coding:utf8 -*-
###############################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
###############################################################################
"""
This module implements the BiDAF algorithm described in
https://arxiv.org/abs/1611.01603

Authors: Yizhong Wang(wangyizhong01@baidu.com)
Date: 2017/09/20 12:00:00
"""

import os
import json
import numpy as np
from collections import Counter


class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, data_dir, task, max_p_num, max_p_len, max_q_len,
                 train=True, dev=True, test=True):
        self.data_dir = data_dir
        self.task = task
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set = self._load_dataset(self.task + '.train') if train else None
        if train:
            print('Train set size: {} questions.'.format(len(self.train_set)))

        self.dev_set = self._load_dataset(self.task + '.dev') if dev else None
        if dev:
            print('Dev set size: {} questions.'.format(len(self.dev_set)))

        self.test_set = self._load_dataset(self.task + '.test') if test else None
        if test:
            print('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, prefix):
        """
        Loads the dataset
        Args:
            prefix: task + 'train/dev/test' indicating the data file to load
        """
        with open(os.path.join(self.data_dir, prefix + '.json')) as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                if 'train' in prefix:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue
                    sample['answer_passages'] = sample['answer_docs']

                sample['query_tokens'] = sample['segmented_query']

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    if 'train' in prefix:
                        most_related_para = doc['most_related_para']
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected']}
                        )
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            query_tokens = sample['segmented_query']
                            common_with_query = sum(Counter(para_tokens) & Counter(query_tokens))
                            recall_wrt_query = float(common_with_query) / len(query_tokens)
                            para_infos.append((para_tokens, recall_wrt_query, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]
                        sample['passages'].append({'passage_tokens': fake_passage_tokens})
                data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'query_token_ids': [],
                      'query_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['query_token_ids'].append(sample['query_token_ids'])
                    batch_data['query_length'].append(len(sample['query_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['query_token_ids'].append([])
                    batch_data['query_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_passage_len, padded_query_len = self._dynamic_padding(batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_passage_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid while evaluating
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['query_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['query_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                         for ids in batch_data['query_token_ids']]
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['query_tokens']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the query and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is not None:
                for sample in data_set:
                    sample['query_token_ids'] = vocab.convert_to_ids(sample['query_tokens'])
                    for passage in sample['passages']:
                        passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)