# coding:utf8
###############################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
###############################################################################
"""
This package implements some utility functions shared by PaddlePaddle
and Tensorflow model implementations.

Authors: liuyuan(liuyuan04@baidu.com)
Date:    2017/10/06 18:23:06
"""

from .dureader_eval import compute_bleu_rouge
from .dureader_eval import normalize
from .preprocess import find_fake_answer
from .preprocess import find_best_query_match

__all__ = [
    'compute_bleu_rouge',
    'normalize',
    'find_fake_answer',
    'find_best_query_match',
    ]
