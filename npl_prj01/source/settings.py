#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SEP = '\t'
DATA_DIR = '../data'
MAIN_FILENAME = 'gender_age_dataset.txt'
UID_META_FILENAME = 'uid_meta.txt'
UID_DOMAIN_URL_TIMESTAMP_FILENAME = 'uid_domain_url_timestamp.txt'
UID_LOST_URLS_FILENAME = 'uid_lost_url.txt'


MAIN_FILE_PATH = '/'.join([DATA_DIR, MAIN_FILENAME])
UID_META_FILE_PATH = '/'.join([DATA_DIR, UID_META_FILENAME])
UID_LOST_URLS_FILE_PATH = '/'.join([DATA_DIR, UID_LOST_URLS_FILENAME])
UID_DOMAIN_URL_TIMESTAMP_FILE_PATH = '/'.join([DATA_DIR, UID_DOMAIN_URL_TIMESTAMP_FILENAME])


META_NAMES = ['description',
              'og:description',
              'twitter:description'
              'keywords',
              'title',
              'og:title',
              'twitter:title']
