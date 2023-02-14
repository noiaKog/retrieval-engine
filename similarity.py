# import pyspark
# import sys
from collections import Counter, OrderedDict, defaultdict
# import itertools
# from itertools import islice, count, groupby
# import pandas as pd
# import os
# import re
# from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
# from time import time
# from pathlib import Path
import pickle

from google.cloud import storage
import numpy as np

import hashlib


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


client = storage.Client()
bucket = client.get_bucket('207653312_313689804')


blob = bucket.blob('Body_index.pkl')
pkl = blob.download_as_string()
Body_index = pickle.loads(pkl)
#
blob = bucket.blob('body_idf.pkl')
pkl = blob.download_as_string()
body_idf = pickle.loads(pkl)
#


nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)


def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = [token for token in tokens if (token not in all_stopwords)]
    return tokens

def get_candidate_documents_and_scores(query_to_search, index,blob_name):
    words_in_index = index.df.keys()
#     print(words_in_index)
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words_in_index:
            pls = index.read_posting_list(w=term, blob_name=blob_name)
            for doc_id, score in pls:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + score
    return candidates

def get_candidate_documents_and_scores_search(query_to_search, index, blob_name):
    words_in_index = index.df.keys()
    #     print(words_in_index)
    candidates = []
    for term in np.unique(query_to_search):
        if term in words_in_index:
            pls = index.read_posting_list(w=term, blob_name=blob_name)
            doc_pls = [x[0] for x in pls]
            candidates = candidates + doc_pls
    return candidates


def tfidf_candidates(candidate, body_dl, body_idf):
    tfidf_candidates = {}
    for doc_id_term in candidate.keys():
        tfidf_candidates[doc_id_term] = np.multiply(np.divide(candidate[doc_id_term], body_dl[doc_id_term[0]]), body_idf[doc_id_term[1]])
    return tfidf_candidates

def tfidf_query(query_to_search):
    tf_query_to_search = Counter(query_to_search)
    words_in_index = Body_index.df.keys()
    for term in list(tf_query_to_search.keys()):
        if term in words_in_index:
            tf_query_to_search[term] = np.multiply(np.divide(tf_query_to_search[term], len(query_to_search)), body_idf[term])
        else:
            tf_query_to_search[term] = 0
    return tf_query_to_search


def binary_search(candidates):
    docs_terms_exist = Counter(candidates)
    similarity = sorted(docs_terms_exist, key=docs_terms_exist.get, reverse=True)
    return similarity

def cosine_similarity(query_to_search, candidates, doc_norm):
    cosine_similarity_dict = defaultdict(float)
    tf_query_to_search = Counter(query_to_search)
    for (doc_id, token), tfidf in candidates.items():
        cosine_similarity_dict[doc_id] += (tfidf * tf_query_to_search[token]) / doc_norm[doc_id]
    similarity = sorted(cosine_similarity_dict, key=cosine_similarity_dict.get, reverse=True)
    return similarity