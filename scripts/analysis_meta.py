#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import nltk
# import numpy as np
import pandas as pd
from time import time
import pickle
# import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# from nltk.corpus import stopwords # as nltk_stopwords
import stopwords as stopwords
import pymorphy2
from nltk.stem import WordNetLemmatizer, SnowballStemmer
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2


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


# def prepare_dfs_new(gen_file_path, in_file_path, feature):
def prepare_dfs_new(gen_df, meta_df, feature):
    print '_' * 80
    print "Cooking data for feature '%s'" % feature
    t0 = time()

    gen_train_df = gen_df[~((gen_df['gender'] == '-') & (gen_df['age'] == '-'))]
    gen_test_df = gen_df[(gen_df['gender'] == '-') & (gen_df['age'] == '-')]
    # gen_test_df_uids = gen_test_df['uid'].unique()

    feature_fac_name = 'gender' if feature == 'age' else 'age'

    print "Preparing Train DataFrame"
    meta_series = meta_df.groupby('uid')['meta']\
        .apply(lambda x: u' '.join([unicode(ss).replace('&nbsp', ' ').replace('&quot', '"')
                                   .replace('&laquo', '"').replace('&raquo', '"') for ss in x.tolist()]))

    meta_base_df = pd.DataFrame(meta_series, index=meta_series.index, columns=['meta'])
    meta_base_df['uid'] = meta_base_df.index

    meta_train_df = pd.merge(meta_base_df, gen_train_df, on='uid', sort=False)
    # meta_train_df = meta_train_df.drop(feature_to_del, axis=1)
    feature_fac = 0
    if feature_fac_name == 'age':
        meta_train_df = pd.get_dummies(meta_train_df, columns=[feature_fac_name])
    else:
        feature_fac = pd.factorize(meta_train_df[feature_fac_name])
        meta_train_df[feature_fac_name] = feature_fac[0]

    target_names_features = pd.factorize(meta_train_df[feature])
    meta_train_df[feature] = target_names_features[0]
    uids_train = meta_train_df['uid']
    meta_train_df = meta_train_df.drop('uid', axis=1)

    print "Preparing Test DataFrame"
    meta_test_df = pd.merge(meta_base_df, gen_test_df, on='uid', sort=False)
    uids_test = meta_test_df['uid']
    meta_test_df = meta_test_df.drop(['uid', 'gender', 'age'], axis=1)

    command_time = time() - t0
    print "cooking data time: %0.3fs" % command_time

    return meta_train_df, uids_train, meta_test_df, uids_test, target_names_features, feature_fac, feature_fac_name


class LemmaTokenizer(object):
    def __init__(self):
        # self.wnl = WordNetLemmatizer()
        self.wnl = SnowballStemmer('english')
        self.morph = pymorphy2.MorphAnalyzer()

    def __call__(self, text):
        tokenz = CountVectorizer().build_tokenizer()(text)
        lemmas = []
        for t in tokenz:
            if len(t) > 2:
                p = self.morph.parse(t)
                if 'LATN' in p[0].tag:  # and re.search('!\d+', p[0].normal_form)
                    # lemmas.append(self.wnl.lemmatize(t))
                    lemmas.append(self.wnl.stem(t))
                elif 'NUMB' in p[0].tag:
                    continue
                elif 'UNKN' in p[0].tag:
                    continue
                elif 'ROMN' in p[0].tag:
                    continue
                else:
                    lemmas.append(p[0].normal_form)
        return lemmas


def vectorizing(meta_train_df, meta_test_df, max_features=20):
    print '_' * 80
    print "Vectorizing max_feartures=%d" % max_features
    print "Train"
    t0 = time()

    # y_target = meta_train_df[feature]
    corpus_train = meta_train_df['meta'].tolist()
    stop_words = stopwords.get_stopwords('english') + stopwords.get_stopwords('russian') + ['www', 'com', 'ru']
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words,
                                 ngram_range=(1, 2), max_features=max_features)  # ngram_range=(1, 2)
    fit_train = vectorizer.fit_transform(corpus_train).toarray()
    # feature_names = vectorizer.get_feature_names().toarray()
    words = [x[0] for x in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])]
    X_train = pd.DataFrame(fit_train, columns=words)

    command_time = time() - t0
    print "vectorizing train time: %0.3fs" % command_time
    print X_train.shape
    print "Test"
    t0 = time()

    # y_test = meta_test_df.index.values
    corpus_test = meta_test_df['meta'].tolist()
    fit_test = vectorizer.transform(corpus_test).toarray()
    words = [x[0] for x in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])]
    X_test = pd.DataFrame(fit_test, columns=words)

    command_time = time() - t0
    print "vectorizing test time: %0.3fs" % command_time
    print X_test.shape

    return X_train, X_test


def f_pred(clf, clf_name, x_train, y_train, x_test):
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
    pred = clf.predict_proba(x_test)
    test_time = time() - t0
    print "predicting time:  %0.3fs" % test_time
    print "Predicting length: %d" % len(pred)

    return pred


def main(gen_file_path, in_file_path, n_estimators=2000, max_features=8000):
    print "Load to DataFrame gen_file_path"
    gen_df = pd.read_csv(gen_file_path,
                         encoding='utf-8',
                         sep='\t',
                         skipinitialspace=True,
                         usecols=['gender', 'age', 'uid']
                         )

    print "Load to DataFrame meta data"
    meta_df = pd.read_csv(in_file_path,
                          encoding='utf-8',
                          sep='\t',
                          skipinitialspace=True,
                          )

    # Preparing data info
    meta_train_df_a, uids_train_a, meta_test_df_a, uids_test_a, \
        target_names_features_a, feature_fac_a, feature_fac_name_a = \
        prepare_dfs_new(gen_df, meta_df, 'age')

    meta_train_df_g, uids_train_g, meta_test_df_g, uids_test_g,\
        target_names_features_g, feature_fac_g, feature_fac_name_g = \
        prepare_dfs_new(gen_df, meta_df, 'gender')

    # Vectorizing
    X_train, X_test = vectorizing(meta_train_df_a, meta_test_df_a, max_features)

    print "After adding additional features"
    X_train_a, X_test_a = X_train, X_test
    y_train_a = meta_train_df_a['age']
    X_train_a['gender'] = meta_train_df_a['gender']
    X_test_a['gender'] = meta_train_df_a['gender']
    print X_train_a.shape, y_train_a.shape, X_test_a.shape

    X_train_g, X_test_g = X_train, X_test
    y_train_g = meta_train_df_g['gender']
    X_train_g['age_18-24'] = meta_train_df_g['age_18-24']
    X_train_g['age_25-34'] = meta_train_df_g['age_25-34']
    X_train_g['age_35-44'] = meta_train_df_g['age_35-44']
    X_train_g['age_45-54'] = meta_train_df_g['age_45-54']
    X_train_g['age_>=55'] = meta_train_df_g['age_>=55']
    X_test_g['age_18-24'] = meta_train_df_g['age_18-24']
    X_test_g['age_25-34'] = meta_train_df_g['age_25-34']
    X_test_g['age_35-44'] = meta_train_df_g['age_35-44']
    X_test_g['age_45-54'] = meta_train_df_g['age_45-54']
    X_test_g['age_>=55'] = meta_train_df_g['age_>=55']
    print X_train_g.shape, y_train_g.shape, X_test_g.shape

    # pickle.dump(X_train, open('/Users/usual/PycharmProjects/npl_project01/data/X_train.pckl', 'w'), protocol=-1)
    # pickle.dump(y_train, open('/Users/usual/PycharmProjects/npl_project01/data/y_train.pckl', 'w'), protocol=-1)
    # pickle.dump(X_test, open('/Users/usual/PycharmProjects/npl_project01/data/X_test.pckl', 'w'), protocol=-1)

    # Classification
    clf_name = 'RandomForestClassifier'
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=10)
    # clf = GradientBoostingClassifier(n_estimators=1000)
    print "Predict for 'age'"
    prd_age = f_pred(clf, clf_name, X_train_a, y_train_a, X_test_a)
    print "Predict for 'gender'"
    prd_gender = f_pred(clf, clf_name, X_train_g, y_train_g, X_test_g)

    return prd_age, prd_gender, uids_test_a, target_names_features_a, target_names_features_g


if __name__ == "__main__":
    gen_file_path = '/Users/usual/PycharmProjects/npl_project01/data/gender_age_dataset.txt'
    in_file_path = '/Users/usual/PycharmProjects/npl_project01/data/csv/uid_meta_fixed.csv'
    project01_gender_age_file_path = '/Users/usual/PycharmProjects/npl_project01/data/csv/project01_gender-age.csv'
    project01_gender_age_dom_file_path = '/Users/usual/PycharmProjects/npl_project01/data/csv/project01_gender-age_dom.csv'

    predict_age, predict_gender, uids, target_names_a, target_names_g = main(gen_file_path, in_file_path,
                                                                             max_features=10000)

    pred_for_test_age = [target_names_a[1][i.argmax()] for i in predict_age]
    pred_for_test_gender = [target_names_g[1][i.argmax()] for i in predict_gender]

    sum_pred = pd.DataFrame({'gender': pred_for_test_gender, 'age': pred_for_test_age}, index=uids)
    sum_pred = sum_pred.sort_index()
    sum_pred.to_csv(project01_gender_age_file_path, sep='\t')
    print "File ready %s" % project01_gender_age_file_path

    # gen_df = pd.read_csv(project01_gender_age_dom_file_path,
    #                      encoding='utf-8',
    #                      sep='\t',
    #                      skipinitialspace=True,
    #                      # usecols=['gender', 'age', 'uid']
    #                      )
