#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import nltk
# import scipy
import numpy as np
import pandas as pd
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2

# import matplotlib.pyplot as plt
# from sklearn.manifold import MDS
# from sklearn.metrics.pairwise import cosine_similarity


def prepare_dfs(gen_file_path, in_file_path):
    print "Load to DataFrame gen_file_path"
    gen_df = pd.read_csv(gen_file_path,
                         encoding='utf-8',
                         sep='\t',
                         skipinitialspace=True,
                         usecols=['gender', 'age', 'uid']
                         )
    gen_train_df = gen_df[~((gen_df['gender'] == '-') & (gen_df['age'] == '-'))]
    gen_train_df_uids = gen_train_df['uid'].unique()

    gen_test_df = gen_df[(gen_df['gender'] == '-') & (gen_df['age'] == '-')]
    gen_test_df_uids = gen_test_df['uid'].unique()

    print "Load to DataFrame meta data"
    meta_df = pd.read_csv(in_file_path,
                          encoding='utf-8',
                          sep='\t',
                          skipinitialspace=True,
                          )

    meta_train_df = meta_df[meta_df['uid'].isin(gen_train_df_uids.tolist())]
    meta_test_df = meta_df[meta_df['uid'].isin(gen_test_df_uids.tolist())]

    t0 = time()
    print "Transform Train DataFrame"
    # Transform Train/Test DataFrame
    tmeta_train_df = pd.DataFrame(columns=('uid', 'meta'))
    tmeta_test_df = pd.DataFrame(columns=('uid', 'meta'))
    i = 0
    for u_uid in gen_train_df_uids:
        df1 = meta_train_df.loc[meta_train_df['uid'] == u_uid, 'meta']
        mstr = u' '.join([unicode(ss) for ss in df1.tolist()])
        mstr = mstr.replace('&nbsp', ' ')
        mstr = mstr.replace('&quot', '"')
        tmeta_train_df.loc[i] = [u_uid, mstr]
        print u'TRAIN UID: %s\tIter: %d' % (u_uid, i)
        i += 1
    tmeta_train_df['meta'].replace('', np.nan, inplace=True)
    tmeta_train_df.dropna(subset=['meta'], inplace=True)

    print "Transform Test DataFrame"
    i = 0
    for u_uid in gen_test_df_uids:
        df1 = meta_test_df.loc[meta_test_df['uid'] == u_uid, 'meta']
        mstr = u' '.join([unicode(ss) for ss in df1.tolist()])
        mstr = mstr.replace('&nbsp', ' ')
        mstr = mstr.replace('&quot', '"')
        tmeta_test_df.loc[i] = [u_uid, mstr]
        print u'TEST UID: %s\tIter: %d' % (u_uid, i)
        i += 1
    tmeta_test_df['meta'].replace('', np.nan, inplace=True)
    tmeta_test_df.dropna(subset=['meta'], inplace=True)

    tmeta_train_df.sort_values(by=['uid'], inplace=True)
    tmeta_test_df.sort_values(by=['uid'], inplace=True)

    train_time = time() - t0
    print "Transform time: %0.3fs" % train_time

    return gen_train_df, gen_test_df, meta_train_df, meta_test_df, tmeta_train_df, tmeta_test_df


def main(gen_file_path, in_file_path):
    print '_' * 80
    print "Cooking data "
    t0 = time()
    gen_train_df, gen_test_df, meta_train_df, meta_test_df, tmeta_train_df, tmeta_test_df = \
        prepare_dfs(gen_file_path, in_file_path)
    train_time = time() - t0
    print "cooking time: %0.3fs" % train_time
    print "Data ready"
# Print out prepared data info
#     print u'len(gen_train_df) = %d\nlen(gen_test_df) = %d\nlen(meta_train_df) = %d\n' \
#           u'len(meta_test_df) = %d\nlen(tmeta_train_df) = %d\nlen(tmeta_test_df) = %d' % \
#           (len(gen_train_df),
#            len(gen_test_df),
#            len(meta_train_df),
#            len(meta_test_df),
#            len(tmeta_train_df),
#            len(tmeta_test_df))

    # for _, u, tm in tmeta_train_df.itertuples():
    #     print (u'%s\t%s' % (u, tm)).encode('utf-8')

    uids_train = tmeta_train_df['uid'].tolist()
    y_train = tmeta_train_df.index.values
    # print y_train
    # print type(y_train)
    corpus_train = tmeta_train_df['meta'].tolist()
    vectorizer = TfidfVectorizer() # ngram_range=(1, 2)
    x_train = vectorizer.fit_transform(corpus_train)
    feature_names = np.array(vectorizer.get_feature_names())

    uids_test = tmeta_test_df['uid'].tolist()
    y_test = tmeta_test_df.index.values
    corpus_test = tmeta_test_df['meta'].tolist()
    x_test = vectorizer.transform(corpus_test)

# Print out x_train
    # print X_train
    print x_train.shape
    print x_test.shape
    # print y_test, uids_test
    # row = 0
    # line = u'-\t'
    # for fn in feature_names:
    #     line += u'%s\n' % fn
    # print line.encode('utf-8')

    # for uid in uids:
    #     line = u'%s\t' % uid
    #     col = 0
    #     for _ in feature_names:
    #         line += u'%s\t' % x_train[row, col]
    #         col += 1
    #     print line.encode('utf-8')
    #     row += 1

    # print np.count_nonzero(x_train)

# Classification
    # x_train = x_train.toarray()
    # ch2 = SelectKBest(chi2, k=10)
    # X_train = ch2.fit_transform(x_train)
    clf = RandomForestClassifier(n_estimators=100)
    print '_' * 80
    print "Training: "
    print "RandomForestClassifier"
    t0 = time()
    clf.fit(x_train, y_train)
    train_time = time() - t0
    print "train time: %0.3fs" % train_time

    print "Predicting: "
    t0 = time()
    pred = clf.predict(x_test)
    test_time = time() - t0
    print "test time:  %0.3fs" % test_time

    score = metrics.accuracy_score(y_test, pred)
    print "accuracy:   %0.3f" % score
    # print pred

    print "classification report:"
    # print uids_test[0]
    print metrics.classification_report(y_test, pred) # , target_names=uids_test

    # dtm = vectorizer.fit_transform(raw_meta)
    # vocab = np.array(vectorizer.get_feature_names())
    # dtm = dtm.toarray()
    # dist = 1 - cosine_similarity(dtm)

    # mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    # pos = mds.fit_transform(dist)
    # xs, ys = pos[:, 0], pos[:, 1]
    #
    # for x, y in zip(xs, ys):
    #     # color = 'orange' if "Austen" in name else 'skyblue'
    #     plt.scatter(x, y)
    #     plt.text(x, y)
    #
    # for x, y in zip(xs, ys):
    #     # color = 'orange' if "Austen" in name else 'skyblue'
    #     plt.scatter(x, y)
    #     # plt.text(x, y)
    # plt.show()
    #
    # linkage_matrix = ward(dist)
    # dendrogram(linkage_matrix, orientation="right")

if __name__ == "__main__":
    main('/Users/usual/PycharmProjects/npl_project01/data/gender_age_dataset.txt',
         '/Users/usual/PycharmProjects/npl_project01/data/csv/uid_meta_fixed.csv')
