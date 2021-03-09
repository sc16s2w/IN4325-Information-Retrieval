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
outputTRECFile = open("improvement2", "w")
ks = krovetz.PyKrovetzStemmer()
stop_words = set(stopwords.words('english'))  # | set(string.punctuation)
language = "en"
rel = "RelatedTo" #more info at: https://github.com/commonsense/conceptnet5/wiki/Relations
origin = "chaos"
destiny = "order"


#this function is to preprocess the whole text
def preprocess(text):
    word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9]', ' ', text.lower()))
    processed = [ks.stem(w) for w in word_tokens if w not in stop_words]
    return processed


#here is for the wiki-pedia score calculating
#to merge the dictionary
def merge_dict(x, y):
    X, Y = Counter(x), Counter(y)
    z = dict(X + Y)
    return z

#to get the document frequency
def get_df(text):
    df_single = {}
    for subs in text:
        df_single[subs] = 1
    return df_single

#this function is to get the term frequency
def get_tf(string):
    word_frequency = {}
    tf = {}
    length = len(string)
    string_set = set(string)
    for single in string_set:
        word_frequency[single] = string.count(single)
        tf[single] = word_frequency[single] / length
    return tf


#the function for the calculate the socre for entities in wiki-pedia
def score(tf, df, len_text, rank, len_passage):
    sum_score = {}
    for passage in range(len_passage):
        for key, value in tf[passage].items():
            tf[passage][key] = (tf[passage][key] / len_text[passage]) * rank[passage] * math.log(len_passage / (df[key]))
    for x in range(len_passage):
        sum_score = merge_dict(sum_score, tf[x])
        print(len(sum_score))
    return sum_score

#to get the n-gram for the concept net processing
def extract_ngram(query):
    sentences = sent_tokenize(query)
    sentences = [word_tokenize(sent) for sent in sentences]
    sentences = [pos_tag(sent) for sent in sentences]
    tuple_get = sentences[0]
    get_result = []
    for i in range(0, len(tuple_get)):
        if tuple_get[i][1] == 'NN':
            get_result.append(tuple_get[i][0])
        if tuple_get[i][1] == 'ADJ':
            get_result.append(tuple_get[i][0])
    return get_result

#the basic setting of conceptnet
class conceptNet:

    def __init__(self):
        self.url = "http://api.conceptnet.io/"

    #search for the specific word
    def lookup(self, lang, term, verbose):
        url_to_search = self.url + "c/" + lang + "/" + term
        data = urlopen(url_to_search)
        json_data = json.load(data)
        relative_edge = []
        if verbose:
            for i in json_data["edges"]:
                relative_edge.append(i["start"]["label"])
        return relative_edge

    #search for the relation
    def relation(self, rel, concept, verbose):
        url_to_search = self.url + "search?rel=/r/" + rel +"&end=/c/en/" + concept
        print(url_to_search)
        data = urlopen(url_to_search)
        json_data = json.load(data)
        if verbose:
            print(url_to_search)
            for i in json_data["edges"]:
                print("----------------")
                print(i["surfaceText"])
                print("weight:", i["weight"])


#the function is to call the dbpedia
def query_expand(query):
    expand_query = query
    #first judge whether there is entities linking with the query
    try:
        annotations = spotlight.annotate('https://api.dbpedia-spotlight.org/en/annotate',
                                     query,
                                     confidence=0.4, support=20)
        df = {}
        true_df = {}
        tf = []
        rank = []
        len_text = []
        for i in range(len(annotations)):
            url = annotations[i][u'URI']
            r = requests.get(url)
            #crawl the lead page
            text = re.findall('<p class="lead">(.*?)</p>', r.text)
            if text:
                pass
            else:
                text = re.findall('<p class="lead">(.*?)</p>', r.text, re.S)
                if text:
                    pass
                else:
                    query_expand_conceptnet(query)
                    continue
            print(text[0])
            text_string = preprocess(text[0])
            print(len(text_string))
            rank.append(annotations[i][u'similarityScore'])
            tf.append(get_tf(text_string))
            df = get_df(text_string)
            true_df = merge_dict(true_df, df)
            len_text.append(len(text_string))
        final_score = score(tf, true_df, len_text, rank, len(annotations))
        final_score = sorted(final_score.items(), key=lambda x:x[1], reverse = True)
        if len(final_score) > 1:
            for qw_ in final_score[:0]:
                expand_query = expand_query + ' ' + qw_[0]
        else:
            for qw_ in final_score:
                expand_query = expand_query + ' ' + qw_[0]
        return expand_query
    # if no entity here, to call the spotlight to enrich the edges.
    except spotlight.SpotlightException:
        print("there is no entity in this query")
        expand_query = query_expand_conceptnet(query)
        return expand_query


#the function to expand the query by conceptnet
def query_expand_conceptnet(query):
    nltk.download('averaged_perceptron_tagger')
    cn = conceptNet()
    relative_content = []
    get_result = extract_ngram(query)
    process_text = []
    nltk.download('words')
    words = set(nltk.corpus.words.words())
    for i in get_result:
        result_initial = cn.lookup(language, i, True)
        if result_initial:
            for i in range(0,len(result_initial)):
                relative_content.append(result_initial[i])
        else:
            query = query
    #to remove all other language words
    for i in relative_content:
        if i.lower() in words or not i.isalpha():
            process_text.append(i)
    if len(process_text) > 1:
        for i in range(0,0):
            query = query + ' ' + process_text[i]
    else:
        for i in range(0, len(process_text)):
            query = query + ' ' + process_text[i]
    return query


if __name__ == "__main__":
    starttime =time.time()
    searcher = SimpleSearcher('/Users/wangsiwei/PycharmProjects/IR_BM/venv/anserini/indexes/msmarco-passage/lucene'
                               '-index-msmarco/')
    index_reader = IndexReader('/Users/wangsiwei/PycharmProjects/IR_BM/venv/anserini/indexes/msmarco-passage/lucene'
                               '-index-msmarco/')
    print(index_reader.stats())
    with open('/Users/wangsiwei/PycharmProjects/IR_BM/msmarco-test2019-queries.tsv', 'rt', encoding="utf-8") as f:
        dict_query = []
        list_query = []
        for jsonstr in f.readlines():
            # 将josn字符串转化为dict字典
            jsonstr = jsonstr.split("\t")
            jsonstr[0] = int(jsonstr[0])
            dict_query.append(jsonstr)
        dict_query2 = dict(dict_query)
        print(len(dict_query2))
    for id, query in dict_query2.items():
        true_passage_list = []
        expand_query = query_expand(query)
        #searcher.set_rm3(10, 100, 0.7)
        hits = searcher.search(expand_query, 1000)
        for i in range(0,len(hits)):
            if i == len(hits) - 1:
                if len(hits) == 1000:
                    true_passage_list.append(hits[i].docid)
                    print("{} Q0 {} {} {:.6f} Anserini\n".format(id, hits[i].docid, i + 1, hits[i].score))
                    outputTRECFile.write(
                        "{} Q0 {} {} {:.6f} Anserini\n".format(id, hits[i].docid, i + 1, hits[i].score))
                else:
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
    endtime = time.time()
    print(endtime - starttime)