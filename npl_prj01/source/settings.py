#!/usr/bin/env python
# -*- coding: utf-8 -*-

SEP = '\t'
DATA_DIR = '../data'
MAIN_FILENAME = 'gender_age_dataset.txt'
UID_META_FILENAME = 'uid_meta.txt'
MAIN_FILE_PATH = '/'.join([DATA_DIR, MAIN_FILENAME])
UID_META_FILE_PATH = '/'.join([DATA_DIR, UID_META_FILENAME])

META_NAMES = ['description',
              'og:description',
              'twitter:description'
              'keywords',
              'title',
              'og:title',
              'twitter:title']
