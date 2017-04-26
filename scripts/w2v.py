#!/usr/bin/env python

import pandas as pd
import numpy as np
import logging
import urlparse
import urllib
import json
import re
import os


from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier


def url2domain(url):
    a = urlparse.urlparse(urllib.unquote(url.strip()))
    if a.scheme not in ('http', 'https'):
        return
    s_res = re.search("(?:www\.)?(.*)", a.netloc)
    if s_res is None:
        return
    b = s_res.group(1)
    if b is None:
        return
    return str(b).strip()


def create_uid_gender_age_domains_csv(gender_age_dataset_txt='../data/gender_age_dataset.txt',
                                      out_file='../data/csv/uid_gender_age_domains.csv'):
    raw_df = pd.read_csv(gender_age_dataset_txt,
                         encoding='utf-8',
                         sep='\t',
                         skipinitialspace=True,
    )
    raw_df['domains'] = [
        json.dumps(
            [url2domain(url.encode('utf-8'))
             for ts, url in sorted([(i['timestamp'], i['url']) for i in json.loads(x)['visits']],
                                   key=lambda x: x[0])]
        )
        for x in tqdm(raw_df.loc[:, 'user_json'])]

    raw_df.drop('user_json', axis=1, inplace=True)
    raw_df.to_csv(out_file, sep='\t', index=False)


# create_uid_gender_age_domains_csv()


def get_df(in_file='../data/csv/uid_gender_age_domains.csv'):
    df = pd.read_csv('../data/csv/uid_gender_age_domains.csv', sep='\t', encoding='utf-8')
    df.domains = df.domains.apply(lambda r: json.loads(r))
    df['text'] = df.domains.apply(lambda r: ' '.join(r))
    return df


def make_feature_vec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    feature_vec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec, model[word])
    #
    # Divide the result by the number of words to get the average
    if nwords:
        return np.divide(feature_vec, nwords)
    return feature_vec


def get_avg_feature_vecs(sentences, model, num_features):
    # Given a set of sentences (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    feature_vecs = np.zeros((len(sentences), num_features), dtype="float32")
    #
    # Loop through the sentences
    for sentence in sentences:
        #
        # Print a status message every 1000th sentence
        if counter % 1000. == 0.:
            log("sentence %d of %d" % (counter, len(sentences)))
        #
        # Call the function (defined above) that makes average feature vectors
        feature_vecs[int(counter)] = make_feature_vec(sentence, model, num_features)
        #
        # Increment the counter
        counter = counter + 1.
    return feature_vecs


def create_w2v(sentences, num_features, min_word_count, num_workers, context, downsampling, seed):
    log("Training Word2Vec model...")
    w2v = Word2Vec(sentences,
                   workers=num_workers,
                   size=num_features,
                   min_count=min_word_count,
                   window=context,
                   sample=downsampling,
                   seed=1)
    w2v.init_sims(replace=True)
    w2v.save(
        os.path.join(
            os.getcwd(),
            "%sfeatures_%sminwords_%scontext" % (num_features, min_word_count, context)))
    return w2v


def log(msg):
    logging.info(msg)
    print msg


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO,
                    filename=os.path.join(os.getcwd(),
                    'logs',
                    'w2v.log'))


df = get_df()
train_df = df[~(df['gender'] == '-')]
test_df = df[df['gender'] == '-']

num_features = 500   # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 4      # Number of threads to run in parallel
context = 10         # Context window size
downsampling = 1e-3  # Downsample setting for frequent words


w2v = create_w2v(df.domains,
                 num_features=num_features,
                 min_word_count=min_word_count,
                 num_workers=num_workers,
                 context=context,
                 downsampling=downsampling,
                 seed=1)

log("Creating average feature vecs for training reviews")
train_data_vecs = get_avg_feature_vecs(train_df.domains, w2v, num_features)

log("Creating average feature vecs for test reviews")
test_data_vecs = get_avg_feature_vecs(test_df.domains, w2v, num_features)

rfc = RandomForestClassifier(n_estimators=1000, n_jobs=40)

log("Fitting a random forest to age training data...")
rfc.fit(train_data_vecs, train_df.age)
age_predict = rfc.predict_proba(test_data_vecs)

log("Fitting a random forest to gender training data...")
rfc.fit(train_data_vecs, train_df.gender)
gender_predict = rfc.predict_proba(test_data_vecs)

test_df.gender = gender_predict
test_df.age = age_predict

test_df.sort('uid', axis=0, inplace=True)

test_df.to_csv(os.path.join(os.getcwd(),
                            'project01_gender-age_pred_w2v.csv'),
               sep='\t',
               index=False,
               columns=['uid', 'gender_p', 'age_p'])