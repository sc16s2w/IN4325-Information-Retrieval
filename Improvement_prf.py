import random
import time
from collections import defaultdict
from math import log, sqrt
from decimal import Decimal
import krovetz
import nltk
import itertools

from pyserini.dsearch import TCTColBERTQueryEncoder, SimpleDenseSearcher
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher, querybuilder, get_topics
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import json

tf_idf = {}
dict_doc = {}
dict_query = {}
outputTRECFile = open("improvement1", "w")
ks = krovetz.PyKrovetzStemmer()
stop_words = set(stopwords.words('english'))  # | set(string.punctuation)

#define the preprocess of the query
def preprocess(text):
    word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9]', ' ', text.lower()))
    processed = [ks.stem(w) for w in word_tokens if w not in stop_words]
    return processed

#to find the relevant words
def do_task(qtext, top100_result, expansion_limit):
    percent_rel = 0.35
    rel_docs_ct = int(percent_rel * 100)
    qtext_copy = qtext
    qtext = preprocess(qtext)
    qtext_set = set(qtext)
    df_rel_doc_set, df_all_doc_set = {}, {}
    doc_body_all = []
    i = 0
    tot_doc_len = 0
    for i in range(len(top100_result)):
        for k, v in top100_result[i].items():
            doc_data = v
            doc_body = doc_data.rstrip('\n').split('\t')[-1]
            doc_body = preprocess(doc_body)
            tot_doc_len += len(doc_body)
            doc_body_all.append(doc_body)
            doc_body_set = set(doc_body)
            for word in doc_body_set:
                df_all_doc_set[word] = df_all_doc_set.get(word, 0) + 1
                if word in qtext_set:
                    continue
                if i < rel_docs_ct:
                    df_rel_doc_set[word] = df_rel_doc_set.get(word, 0) + 1
            i += 1
    vocab_words_df = df_all_doc_set
    # Define variables as in paper
    N = 100  # the number of documents in the collectio
    R = rel_docs_ct  # the number of known relevant document for a request = 100
    # r = the number of known relevant documents term t(i) occurs in
    # n = the number of documents term t(i) occurs in = df(i)
    #calculate the score
    for word, r in df_rel_doc_set.items():
        n = vocab_words_df[word]
        score = r * log(
            ((r + 0.5) * (3 * N - n - R + r + 0.5)) / ((n - r + 0.5) * (R - r + 0.5)))  # just update the values
        df_rel_doc_set[word] = score

    # add query terms
    new_queries = sorted(df_rel_doc_set.items(), key=lambda x: x[1], reverse=True)[:expansion_limit]
    for qw_ in new_queries:
        qtext_copy = qtext_copy.rstrip('\n') + ' ' + qw_[0]
    return qtext_copy

#to call the function
def getmorequries():
    searcher = SimpleSearcher.from_prebuilt_index('msmarco-passage')
    with open('/Users/wangsiwei/PycharmProjects/IR_BM/msmarco-test2019-queries.tsv', 'rt', encoding="utf-8") as f:
        # read all lines, all lines can be a string
        dict_query = []
        for jsonstr in f.readlines():
            #exchange the strrng to the dictionary
            jsonstr = jsonstr.split("\t")
            jsonstr[0] = int(jsonstr[0])
            dict_query.append(jsonstr)
            #search top 1000, and to get the relevant words
            hits = searcher.search(jsonstr[1], 100)
            list_store = []
            for i in range(0, len(hits)):
                dict_store = {}
                doc = searcher.doc(hits[i].docid)
                json_doc = json.loads(doc.raw())
                dict_store[hits[i].docid] = json_doc['contents']
                list_store.append(dict_store)
            query_new = do_task(jsonstr[1], list_store, 5)
            jsonstr[1] = query_new
        dict_query2 = dict(dict_query)
    return dict_query2

if __name__ == "__main__":
    searcher = SimpleSearcher('/Users/wangsiwei/PycharmProjects/IR_BM/venv/anserini/indexes/msmarco-passage/lucene'
                               '-index-msmarco/')
    index_reader = IndexReader('/Users/wangsiwei/PycharmProjects/IR_BM/venv/anserini/indexes/msmarco-passage/lucene'
                               '-index-msmarco/')
    dict_query2 = getmorequries()
    for id, query in dict_query2.items():
        true_passage_list = []
        # expand_query = query_expand(query)
        #searcher.set_rm3(10, 100, 0.7)
        hits = searcher.search(query, 1000)
        for i in range(0,len(hits)):
            if i == len(hits) - 1:
                if len(hits) == 1000:
                    true_passage_list.append(hits[i].docid)
                    print("{} Q0 {} {} {:.6f} Anserini\n".format(id, hits[i].docid, i + 1, hits[i].score))
                    outputTRECFile.write(
                        "{} Q0 {} {} {:.6f} Anserini\n".format(id, hits[i].docid, i + 1, hits[i].score))
                else:
                    #if length is not equal to 1000, try to add it to 1000
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
