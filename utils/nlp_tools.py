# -*- coding:utf8 -*-
###############################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
###############################################################################
"""
Basic tools.

Authors: zhaoximo(zhaoximo@baidu.com)
Data: 2017/09/20 12:00:00
"""

import urllib
import urllib2
import sys
import json
import ssl

__all__ = [
    'get_emb',
    'get_wordseg',
    ]

lexer = "https://aip.baidubce.com/rpc/2.0/nlp/v1/lexer"
ember = "https://aip.baidubce.com/rpc/2.0/nlp/v2/word_emb_vec"
access_token = "24.a456bde3693f4b2c6743011dec33ec29.2592000.1508494458.282335-10163834"

lex_url = lexer + "?access_token=%s" % (access_token)
emb_url = ember + "?access_token=%s" % (access_token)

def get_emb(word, dim=1024):
    """
    Gets embedding of the word.

    Args:
        word: The token id to look up with.
        dim: the dim of the embedding. By default the returned embedding is of
             length 1024, if the specified `dim` is less than 1024, the embedding
             vector will be truncated to length of `dim`.

    Returns:
        The embedding of the token. When `dim` is bigger than 1024, returns
        None. If `word` is out of vocabulary, the returned embedding is a
        vector of zeros.
    """
    if dim > 1024:
        return None
    post_data = "{\"word\":\"%s\"}" % (word)
    request = urllib2.Request(emb_url, post_data)
    request.add_header('Content-Type', 'application/json')
    response = urllib2.urlopen(request)
    content = response.read()

    js = json.loads(content.decode("gb18030"))
    if "vec" not in js:
        return [0.0 for i in range(dim)]
    return js["vec"]


def get_wordseg(text):
    """
    Gets segmented text.

    Args:
        text: The text string to be segmented.

    Returns:
        A dict object contains segmented results. For example:

            >>> {"basic_word": [u"word1", u"word2", u"word3"],
                    "ne_word": u"word1+word2+word3",
                    "pos": u"pos",
                    "ne": u"ne"}

        The value of "pos" and "ne" will be empty string by default.
    """
    post_data = "{\"text\":\"%s\"}" % (text)
    request = urllib2.Request(lex_url, post_data)
    request.add_header('Content-Type', 'application/json')
    response = urllib2.urlopen(request)
    content = response.read()

    js = json.loads(content.decode("gb18030"))

    res = []
    for itm in js["items"]:
        dct = {}
        dct["basic_word"] = itm["basic_words"]
        dct["pos"] = itm["pos"]
        dct["ne"] = itm["ne"]
        dct["ne_word"] = ''.join(itm["basic_words"])
        res.append(dct)

    return res


def print_wordseg(seg_res):
    """
    Formats and print the word segment results.

    Args:
        seg_res: the dict returned by `get_wordseg`.

    """
    for dct in seg_res:
        neword = dct["ne_word"]
        bsword = '|||'.join(dct["basic_word"])
        pos = dct["pos"]
        ne = dct["ne"]
        l = [neword, bsword, pos, ne]
        print '\t'.join(l)
