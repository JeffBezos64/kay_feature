import string
import difflib
from collections import Counter
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

from language_checkers.spell_checker import SpellChecker
from language_checkers.grammar_checker import GrammarChecker
from language_checkers.pos_tagger import POSTagger
from scipy import sparse

from tqdm import tqdm
import nltk
#TODO Tweak so that we can vary the vocab size!
class NonGenSimMeanTfidfEmbeddingVectorizer(object):
    def __init__(self, embedder):
        self.embedder = embedder
        self.word2weight = None
        self.dim = len(self.embedder.vec('the'))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def fit_transform(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x, max_features=8192)
        tfidf_data = tfidf.fit_transform(X)
        tfidf_names = tfidf.get_feature_names_out()
        max_idf=max(tfidf.idf_)
        for row in tqdm(range(0, tfidf_data.shape[0]), desc='processing row - column level is supressed'):
                for col in tfidf_data[row].indices:
                    tfidf_data[row,col] = np.mean(tfidf_data[row,col] * self.embedder.vec(tfidf_names[col]))
        tfidf_data.eliminate_zeros()
        return tfidf_data

class GensimEmbed(object):
    def __init__(self, model, OOVRandom=False):
        self.model = model
        self.OOVRandom = OOVRandom
        self.dim = model['the'].shape[0]

    def vec(self, word):
        try:
            return self.model[word]
        except KeyError:
            if self.OOVRandom == True:
                return np.random(self.dim,)
            else:
                return np.zeros(self.dim,)