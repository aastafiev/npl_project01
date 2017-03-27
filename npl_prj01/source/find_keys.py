#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Он содержит данные о посещении сайтов ~40 000 пользователей, при этом по некоторым из них (~ 35 000) известны их пол
#  и возрастная категория, а по 5 000 - эта информация не известна. В файле есть 4 поля:

# gender - пол, принимающий значения M (male - мужчина), F (female - женщина), - (пол неизвестен);
# age - возраст, представленный в виде диапазона x-y (строковый тип), или - (возрастная категория неизвестна);
# uid - идентификатор пользователя, строковая переменная;
# user_json - поле json, в котором содержатся записи о посещении сайтов этим пользователем (url, timestamp).


import re
import json
import urllib2
import argparse
import grequests

import pandas as pd
from urlparse import urlparse
from bs4 import BeautifulSoup
from datetime import datetime


META_NAMES = ['description',
              'og:description',
              'twitter:description'
              'keywords',
              'title',
              'og:title',
              'twitter:title']


def print_time_deltas(now, step_time, start_time):
    l = 'step duration: %s, delta from start: %s' % (now - step_time, now - start_time)
    print '\n'.join(('-' * len(l), l, '-' * len(l)))


def url2domain(url):
    url = re.sub('(http(s)*://)+', 'http://', url)
    parsed_url = urlparse(urllib2.unquote(url.strip()))

    if parsed_url.scheme not in ['http', 'https']:
        return None
    netloc = re.search("(?:www\.)?(.*)", parsed_url.netloc).group(1)

    if netloc is not None:
        return netloc.strip(), parsed_url.scheme


def get_meta_content(html):
    return [tag["content"].strip()
            for tag in BeautifulSoup(html, "lxml").find_all("meta")
            if tag.has_attr("name")
            if tag["name"].lower() in META_NAMES
            if tag["content"].strip()]


def get_all_meta_content(urls_ts, count, timeout=None):
    out = []
    lost_urls = []
    domain_url = []
    urls = []

    for url, timestamp in urls_ts:
        domain, url_scheme = url2domain(url)
        domain_url.extend([(domain, url, timestamp)])
        new_url = u'://'.join([url_scheme, domain])
        if not url in urls:
            urls.append(url)
        if not new_url in urls:
            urls.append(new_url)

    requests = (grequests.get(u) for u in urls)
    responses = grequests.map(requests=requests, gtimeout=timeout)

    for i, response in enumerate(responses):
        try:
            count += 1
            meta_info = get_meta_content(response._content)
            if meta_info:
                out.extend(meta_info)
                print "%s\tGot meta\t%s" % (count, urls[i])
            else:
                lost_urls.append((urls[i], 'empty_meta'))
                print "%s\tEmpty meta\t%s" % (count, urls[i])
        except Exception as ex:
            lost_urls.append((urls[i], 'bad_url'))
            print '%d\tERROR:\t%s - %s' % (count, urls[i], ex)
    return list(set(out)), domain_url, lost_urls, count


def parse_to_files(in_file_path,
                   meta_file_path,
                   domain_urls_timestamp_file_path,
                   lost_urls_file_path,
                   timeout=None,
                   nrows=None):
    df = pd.read_csv(in_file_path,
                     encoding='utf-8',
                     sep='\t',
                     skipinitialspace=True,
                     usecols=['uid', 'user_json'],
                     nrows=nrows,
    )
    start_time = datetime.now()

    with open(meta_file_path, 'w') as um_file, \
            open(lost_urls_file_path, 'w') as lu_file, \
            open(domain_urls_timestamp_file_path, 'w') as dut_file:
        count = 0
        for _, uid, user_json in df.itertuples():
            print "\nUser %s\n------------------" % uid

            user_json = json.loads(user_json)
            urls_ts = [(v['url'], v['timestamp']) for v in user_json['visits']]
            step_time = datetime.now()
            meta_data, domains_urls, lost_urls, count = get_all_meta_content(urls_ts, count, timeout)
            print_time_deltas(datetime.now(), step_time, start_time)

            for md in meta_data:
                um_file.write((u'%s\t%s\n' % (uid, u' '.join(md.split()))).encode('utf-8'))
            for (dom, url, ts) in domains_urls:
                dut_file.write((u'%s\t%s\t%s\t%s\n' % (uid, dom, url, ts)).encode('utf-8'))
            for (url, err) in lost_urls:
                lu_file.write((u'%s\t%s\t%s\n' % (uid, url, err)).encode('utf-8'))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('--in-file-path',
                            default='gender_age_dataset.txt',
                            help='input file path')
    arg_parser.add_argument('--meta-file-path',
                            default='uid_meta.txt',
                            help='meta file path')
    arg_parser.add_argument('--lost-urls-file-path',
                            default='uid_lost_url.txt',
                            help='lost urls file path')
    arg_parser.add_argument('--domain-urls-timestamp-file-path',
                            default='uid_domain_url_timestamp.txt',
                            help='domain urls timestamp file path')
    arg_parser.add_argument('--nrows',
                            type=int,
                            default=None,
                            help='rows count to get from in file')
    arg_parser.add_argument('--timeout',
                            type=int,
                            default=None,
                            help='timeout for all requests of one uid')

    args = arg_parser.parse_args()
    parse_to_files(**args.__dict__)
