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
    def __init__(self, embedder, vectorizer=None):
        self.embedder = embedder
        self.word2weight = None
        self.vectorizer = vectorizer
        self.dim = len(self.embedder.vec('the'))

    def fit(self, X, y):
        #jank workaround so I can save and use the vectorizer for data transforms for my OOD experiment.
        tfidf = TfidfVectorizer(analyzer=lambda x: x, max_features=8192)
        tfidf.fit(X)
        return tfidf

    def fit_transform(self, X, y):
        if self.vectorizer == None:
            tfidf = TfidfVectorizer(analyzer=lambda x: x, max_features=8192)
            tfidf_data = tfidf.fit_transform(X)
        else:
            tfidf_data = self.vectorizer.transform(X)
        tfidf_names = self.vectorizer.get_feature_names_out()
        max_idf=max(self.vectorizer.idf_)
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
                return np.random.rand(self.dim,)
            else:
                return np.zeros(self.dim,)