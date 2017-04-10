#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import nltk
import numpy as np
import pandas as pd
from time import time
import re

from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from nltk.corpus import stopwords
import pymorphy2
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# import matplotlib.pyplot as plt
# from sklearn.manifold import MDS
# from sklearn.metrics.pairwise import cosine_similarity


def prepare_dfs(gen_file_path, in_file_path, feature):
    print "Load to DataFrame gen_file_path"
    gen_df = pd.read_csv(gen_file_path,
                         encoding='utf-8',
                         sep='\t',
                         skipinitialspace=True,
                         usecols=['gender', 'age', 'uid']
                         )
    gen_train_df = gen_df[~((gen_df['gender'] == '-') & (gen_df['age'] == '-'))]
    gen_test_df = gen_df[(gen_df['gender'] == '-') & (gen_df['age'] == '-')]
    gen_test_df_uids = gen_test_df['uid'].unique()

    print "Load to DataFrame meta data"
    meta_df = pd.read_csv(in_file_path,
                          encoding='utf-8',
                          sep='\t',
                          skipinitialspace=True,
                          )

    print "Preparing Train DataFrame"
    meta_train_df = pd.merge(meta_df, gen_train_df, on='uid', sort=False)
    meta_train_series = meta_train_df.groupby(feature)['meta'] \
        .apply(lambda x: u' '.join([unicode(ss).replace('&nbsp', ' ').replace('&quot', '"')
                                   .replace('&laquo', '"').replace('&raquo', '"') for ss in x.tolist()]))
    meta_train_df = pd.DataFrame(meta_train_series, index=meta_train_series.index, columns=['meta'])

    print "Preparing Test DataFrame"
    meta_test_df = meta_df[meta_df['uid'].isin(gen_test_df_uids.tolist())]
    meta_test_series = meta_test_df.groupby('uid')['meta'] \
        .apply(lambda x: u' '.join([unicode(ss).replace('&nbsp', ' ').replace('&quot', '"')
                                   .replace('&laquo', '"').replace('&raquo', '"') for ss in x.tolist()]))
    meta_test_df = pd.DataFrame(meta_test_series, index=meta_test_series.index, columns=['meta'])

    # return gen_train_df, gen_test_df, meta_train_df, meta_test_df
    return meta_train_df, meta_test_df


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.morph = pymorphy2.MorphAnalyzer()

    def __call__(self, text):
        tokenz = TfidfVectorizer(ngram_range=(1, 2)).build_tokenizer()(text)
        lemmas = []
        for t in tokenz:
            if len(t) > 2:
                p = self.morph.parse(t)
                if 'LATN' in p[0].tag:  # and re.search('!\d+', p[0].normal_form)
                    lemmas.append(self.wnl.lemmatize(t))
                elif 'NUMB' in p[0].tag:
                    continue
                elif 'UNKN' in p[0].tag:
                    continue
                elif 'ROMN' in p[0].tag:
                    continue
                else:
                    lemmas.append(p[0].normal_form)
        return lemmas


def vectorizing(meta_train_df, meta_test_df):
    print '_' * 80
    print "Vectorizing"
    print "Train"
    t0 = time()

    y_train = meta_train_df.index.values
    corpus_train = meta_train_df['meta'].tolist()
    stop_words = stopwords.words('english') + stopwords.words('russian') + ['www', 'com', 'ru']
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words,
                                 ngram_range=(1, 3), max_features=15000)  # ngram_range=(1, 2)
    x_train = vectorizer.fit_transform(corpus_train)
    feature_names = np.asarray(vectorizer.get_feature_names())

    command_time = time() - t0
    print "vectorizing train time: %0.3fs" % command_time

    # print x_train.shape
    # print x_train.toarray()
    #
    # ch2 = SelectKBest(chi2, k=10)
    # x_train = ch2.fit_transform(x_train, y_train)

    print "Test"
    t0 = time()

    y_test = meta_test_df.index.values
    corpus_test = meta_test_df['meta'].tolist()
    x_test = vectorizer.transform(corpus_test)

    command_time = time() - t0
    print "vectorizing test time: %0.3fs" % command_time
    # x_test = ch2.transform(x_test)

    # Print some info x_train, x_test
    # print x_train.toarray()
    print x_train.shape

    # print x_test.toarray()
    print x_test.shape

    return x_train, y_train, x_test, y_test, feature_names


def predicting(clf, clf_name, x_train, y_train, x_test, y_test):
    # clf = MultinomialNB(alpha=.01)  # BernoulliNB(alpha=.01)
    print '_' * 80
    print "%s" % clf_name
    print "Training:"
    t0 = time()
    clf.fit(x_train, y_train)
    train_time = time() - t0
    print "train time: %0.3fs" % train_time

    print "Predicting: "
    t0 = time()
    pred = clf.predict(x_test)
    test_time = time() - t0
    print pred
    print "test time:  %0.3fs" % test_time

    score = metrics.accuracy_score(y_test, pred)
    print "accuracy:   %0.3f" % score
    # print pred

    print "classification report:"
    # print uids_test[0]
    print metrics.classification_report(y_test, pred)  # , target_names=uids_test

    print "confusion matrix:"
    print metrics.confusion_matrix(y_test, pred)


def main(gen_file_path, in_file_path, feature):
    # Preparing data info
    print '_' * 80
    print "Cooking data for feature '%s'" % feature
    t0 = time()

    meta_train_df, meta_test_df = prepare_dfs(gen_file_path, in_file_path, feature)

    command_time = time() - t0
    print "cooking data time: %0.3fs" % command_time

    # Vectorizing
    x_train, y_train, x_test, y_test, feature_names = vectorizing(meta_train_df, meta_test_df)

    # Classification
    #     clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    clf = MultinomialNB(alpha=.01)  # BernoulliNB(alpha=.01)
    predicting(clf, 'MultinomialNB', x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main('/Users/usual/PycharmProjects/npl_project01/data/gender_age_dataset.txt',
         '/Users/usual/PycharmProjects/npl_project01/data/csv/uid_meta_fixed.csv', 'age')
