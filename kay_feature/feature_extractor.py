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
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf_data = tfidf.fit_transform(X)
        tfidf_names = tfidf.get_feature_names_out()
        max_idf=max(tfidf.idf_)
        for row in tqdm(range(0, tfidf_data.shape[0]), desc='processing row - column level is supressed'):
                for col in tfidf_data[row].indices:
                    tfidf_data[row,col] = np.mean(tfidf_data[row,col] * self.embedder.vec(tfidf_names[col]))
        tfidf_data.eliminate_zeros()
        return tfidf_data


class GenSimMeanTfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

#, remove_stopwords=True make sure to include stopwords!
class EmbeddingFeatureExtractor():
    def __init__(self, vectorizer=None, spell_check_flag=False, max_edit_distance=2):
        self.vectorizer = vectorizer
        self.spell_check_flag = spell_check_flag
        self._spell_checker = SpellChecker(max_edit_distance=max_edit_distance)

    def transform_spelling(self, X):
        X = [[self._spell_checker.get_close_correction(z) for z in y] for y in X]
        return X
    
    def spell_check(self, X):
        if self.spell_check_flag == True:
            return self.transform_spelling(self, X)
        else:
            return X

    def preprocess(self, X):
        X = [self._clean_text(x) for x in X]
        X = [self._remove_punctuation(x) for x in X]
        X = [nltk.word_tokenize(x) for x in X]
        return X

    def _clean_text(self, text: str) -> str:
        return text.replace('\n', ' ')

    def _remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))

    def transform(self, X):
        X = self.preprocess(X)
        if self.spell_check_flag == True:
            X = self.spell_check(X)
        return self._vectorizer.transform(X)
        