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
from collections import Counter, defaultdict
import math
import datetime
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
outputTRECFile = open("improvement2", "w")
ks = krovetz.PyKrovetzStemmer()
stop_words = set(stopwords.words('english'))  # | set(string.punctuation)


def preprocess(text):
    word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9]', ' ', text.lower()))
    processed = [ks.stem(w) for w in word_tokens if w not in stop_words]
    return processed

#to extract the relevance 
def relevance_extract():
    with open('qrels.dev.tsv', 'rt', encoding="utf-8") as f:
        dict_query = defaultdict(list)
        list_passage={}
        for jsonstr in f.readlines():
            # Convert josn string to dict dictionary
            jsonstr = jsonstr.split("\t")
            dict_query[jsonstr[0]].append(jsonstr[2])
        return dict_query

#extract seven features.
def feature_extract():
    searcher = SimpleSearcher.from_prebuilt_index('msmarco-passage')
    searcher.set_qld(1000)
    relevance_label = relevance_extract()
    index_reader = IndexReader(
        '/Users/wangsiwei/PycharmProjects/IR_BM/venv/anserini/indexes/msmarco-passage/lucene-index-msmarco/')
    with open('/Users/wangsiwei/PycharmProjects/IR_BM/venv/anserini/collections/msmarco-passage/queries.dev.tsv',
              'rt', encoding="utf-8") as f:
        dict_query = []
        i = 0
        for jsonstr in f.readlines():
            i = i+1
            # Convert josn string to dict dictionary
            jsonstr = jsonstr.split("\t")
            jsonstr[0] = int(jsonstr[0])
            if i>9000:
                dict_query.append(jsonstr)
        dict_query2 = dict(dict_query)
    # topics = get_topics('msmarco-passage-dev-subset')
    for id, query in dict_query2.items():
        query_tf_idf_score = []
        true_idf =0
        hits = searcher.search(query, 1000)
        length_query = len(preprocess(query))
        labels = relevance_label[str(id)]
        new_query = preprocess(query)
        for i in range(0, len(hits)):
            true_tf_idf_idf = {}
            true_tf_idf_tf = {}
            true_tf_idf = {}
            true_df = {}
            sum_tf = 0
            score_tf = 0
            score_idf = 0
            score_tf_dtf = 0
            sum_of_square = 0
            doc = searcher.doc(hits[i].docid)
            json_doc = json.loads(doc.raw())
            length_passage = len(preprocess(json_doc['contents']))
            score = index_reader.compute_query_document_score(hits[i].docid, query)
            tf = index_reader.get_document_vector(hits[i].docid)
            for b in tf:
                sum_tf += tf[b]
            print(new_query)
            for q in new_query:
                if q in tf:
                    true_tf = tf[q] / sum_tf
                else:
                    true_tf = 0
                true_df = {q: (index_reader.get_term_counts(q, analyzer=None))[0]}
                # print(true_df)
                if true_df[q] != 0:
                    true_idf = 1 + math.log(8841822 / float(true_df[q]))
                else:
                    true_idf == 1
                # print(true_idf)
                true_tf_idf_tf[q] = true_tf
                true_tf_idf[q] = true_tf * true_idf
                true_tf_idf_idf[q] = true_idf
            for a in true_tf_idf:
                score_tf_dtf += true_tf_idf[a]
            for b in true_tf_idf_idf:
                print(true_tf_idf_idf[b])
                score_idf += true_tf_idf_idf[b]
            for c in true_tf_idf_tf:
                score_tf += true_tf_idf_tf[c]

            # postings_list = index_reader.get_postings_list(tf_idf_query)
            # for posting in postings_list:
            #     print(f'docid={posting.docid}, tf={posting.tf}, pos={posting.positions}')
            if hits[i].docid in labels:
                relevance_score = 1
                # print("hh")
            else:
                relevance_score = 0
            # print(("{} qid:{} 1:{} 2:{}  #docid = {}\n".format(1000 - i, id, hits[i].score, score,
            #                                                     hits[i].docid)))
            if i == len(hits) - 1:
                if len(hits) == 1000:
                    outputTRECFile.write(
                        "{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} #docid = {}\n".format(relevance_score, id,
                                                                                            hits[i].score, score,
                                                                                            length_query,
                                                                                            length_passage,
                                                                                            score_tf_dtf, score_tf,
                                                                                            score_idf, hits[i].docid))
                    print(("{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} #docid = {}\n".format(relevance_score, id,
                                                                                               hits[i].score, score,
                                                                                               length_query,
                                                                                               length_passage,
                                                                                               score_tf_dtf, score_tf,
                                                                                               score_idf,
                                                                                               hits[i].docid)))
                else:
                    while i != 1000:
                        print(("{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} #docid = {}\n".format(relevance_score, id,
                                                                                                   0, 0, length_query,
                                                                                                   0, 0, 0, 0,
                                                                                                   "unknown")))
                        outputTRECFile.write(
                            "{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} #docid = {}\n".format(relevance_score, id, 0,
                                                                                                0, length_query, 0, 0,
                                                                                                0, 0,
                                                                                                "unknown"))
                        i += 1
            else:
                outputTRECFile.write(
                    "{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} #docid = {}\n".format(relevance_score, id,
                                                                                        hits[i].score, score,
                                                                                        length_query, length_passage,
                                                                                        score_tf_dtf, score_tf,
                                                                                        score_idf, hits[i].docid))
                print(("{} qid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} #docid = {}\n".format(relevance_score, id,
                                                                                           hits[i].score, score,
                                                                                           length_query, length_passage,
                                                                                           score_tf_dtf,
                                                                                           score_tf, score_idf,
                                                                                           hits[i].docid)))


if __name__ == '__main__':
    feature_extract()
