import random
import time

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.tokenize import word_tokenize

import json
from urllib.request import urlopen
import spotlight
import requests
import re
import krovetz
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import math
import datetime
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
outputTRECFile = open("BM25_origin2", "w")

#this function can build a basic bm25 model
if __name__ == "__main__":
    searcher = SimpleSearcher('/Users/wangsiwei/PycharmProjects/IR_BM/venv/anserini/indexes/msmarco-passage/lucene'
                               '-index-msmarco/')
    index_reader = IndexReader('/Users/wangsiwei/PycharmProjects/IR_BM/venv/anserini/indexes/msmarco-passage/lucene'
                               '-index-msmarco/')
    print(index_reader.stats())
    with open('/Users/wangsiwei/PycharmProjects/IR_BM/msmarco-test2019-queries.tsv', 'rt', encoding="utf-8") as f:
        dict_query = []
        list_query = []
        for jsonstr in f.readlines():
            # Convert josn string to dict dictionary
            jsonstr = jsonstr.split("\t")
            jsonstr[0] = int(jsonstr[0])
            dict_query.append(jsonstr)
        dict_query2 = dict(dict_query)
        print(len(dict_query2))
    for id, query in dict_query2.items():
        true_passage_list = []
        hits = searcher.search(query, 1000)
        for i in range(0,len(hits)):
            if i == len(hits) - 1:
                if len(hits) == 1000:
                    true_passage_list.append(hits[i].docid)
                    print("{} Q0 {} {} {:.6f} Anserini\n".format(id, hits[i].docid, i + 1, hits[i].score))
                    outputTRECFile.write(
                        "{} Q0 {} {} {:.6f} Anserini\n".format(id, hits[i].docid, i + 1, hits[i].score))
                else:
                    #if there are not 1000 passages, add the other one
                    while i != 1000:
                        fake_passage_id = random.randint(0, 8841822)
                        while fake_passage_id in true_passage_list:
                            fake_passage_id = random.randint(0, 8841822)

                        print("{} Q0 {} {} {:.6f} Anserini\n".format(id, fake_passage_id, i + 1, 0))
                        outputTRECFile.write(
                            "{} Q0 {} {} {:.6f} Anserini\n".format(id, fake_passage_id, i + 1, 0))
                        i += 1
            else:
                true_passage_list.append(hits[i].docid)
                print("{} Q0 {} {} {:.6f} Anserini\n".format(id, hits[i].docid, i + 1, hits[i].score))
                outputTRECFile.write(
                    "{} Q0 {} {} {:.6f} Anserini\n".format(id, hits[i].docid, i + 1, hits[i].score))
