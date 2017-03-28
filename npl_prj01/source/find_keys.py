#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Он содержит данные о посещении сайтов ~40 000 пользователей, при этом по некоторым из них (~ 35 000) известны их пол
#  и возрастная категория, а по 5 000 - эта информация не известна. В файле есть 4 поля:

# gender - пол, принимающий значения M (male - мужчина), F (female - женщина), - (пол неизвестен);
# age - возраст, представленный в виде диапазона x-y (строковый тип), или - (возрастная категория неизвестна);
# uid - идентификатор пользователя, строковая переменная;
# user_json - поле json, в котором содержатся записи о посещении сайтов этим пользователем (url, timestamp).


import re
import sys
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

log_file = None


def log(msg, _bounder=None):
    if _bounder:
        msg = '\n'.join((_bounder * len(msg), msg, _bounder * len(msg)))
    msg += '\n'
    if log_file:
        log_file.write(msg.encode('utf-8'))
    sys.stdout.write(msg.encode('utf-8'))


def delta2str(delta):
    return ('%s' % delta)[:7]


def size2str(size):
    for m in 'bKMG':
        size /= 1024.
        if size < 1:
            return '%d%s' % (1024 * size, m)


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


def get_all_meta_content(urls_ts, count, requests_count=128, timeout=None):
    out = []
    lost_urls = []
    domain_url = []
    urls = []
    total_size = 0

    for url, timestamp in urls_ts:
        domain, url_scheme = url2domain(url)
        domain_url.extend([(domain, url, timestamp)])
        new_url = u'://'.join([url_scheme, domain])
        if not url in urls:
            urls.append(url)
        if not new_url in urls:
            urls.append(new_url)

    for _urls in (urls[i:i+requests_count] for i in xrange(0, len(urls), requests_count)):
        requests = (grequests.get(u) for u in _urls)
        responses = grequests.map(requests=requests, gtimeout=timeout)

        for i, response in enumerate(responses):
            try:
                count += 1
                meta_info = get_meta_content(response._content)
                response_time = response.elapsed
                response_size = len(response._content)
                total_size += response_size
                response.close()

                if meta_info:
                    out.extend(meta_info)
                    log(u'%s\tGot meta\t%s\t%s\t%s' % (
                        count,
                        delta2str(response_time),
                        size2str(len(response._content)),
                        _urls[i]))
                else:
                    lost_urls.append((_urls[i], 'empty_meta'))
                    log(u'%s\tEmpty meta\t%s\t%s\t%s' % (
                        count,
                        delta2str(response_time),
                        size2str(len(response._content)),
                        _urls[i]))
            except Exception as ex:
                lost_urls.append((_urls[i], 'bad_url'))
                log(u'%d\tERROR:\t%s - %s' % (count, _urls[i], ex))
    return list(set(out)), domain_url, lost_urls, total_size, count


def parse_to_files(in_file_path,
                   meta_file_path,
                   domain_urls_timestamp_file_path,
                   lost_urls_file_path,
                   requests_count=128,
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
    total_size = 0

    with open(meta_file_path, 'w') as um_file, \
            open(lost_urls_file_path, 'w') as lu_file, \
            open(domain_urls_timestamp_file_path, 'w') as dut_file:
        count = 0
        for _, uid, user_json in df.itertuples():
            user_json = json.loads(user_json)
            urls_ts = [(v['url'], v['timestamp']) for v in user_json['visits']]
            step_time = datetime.now()
            meta_data, domains_urls, lost_urls, user_total_size, count = get_all_meta_content(
                urls_ts,
                count,
                requests_count,
                timeout
            )

            total_size += user_total_size
            now = datetime.now()
            log('%s | step duration: %s, delta from start: %s, size: %s, total size: %s' % \
                    (uid,
                     delta2str(now - step_time),
                     delta2str(now - start_time),
                     size2str(user_total_size),
                     size2str(total_size)),
                _bounder='-'
            )

            for md in meta_data:
                um_file.write((u'%s\t%s\n' % (uid, u' '.join(md.split()))).encode('utf-8'))
            for (dom, url, ts) in domains_urls:
                dut_file.write((u'%s\t%s\t%s\t%s\n' % (uid, dom, url, ts)).encode('utf-8'))
            for (url, err) in lost_urls:
                lu_file.write((u'%s\t%s\t%s\n' % (uid, url, err)).encode('utf-8'))

    log('total duration: %s, total size: %s' % \
            (delta2str(datetime.now() - start_time),
             size2str(user_total_size)),
        _bounder='='
    )


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
    arg_parser.add_argument('--log-file',
                            default=None,
                            help='log file')
    arg_parser.add_argument('--nrows',
                            type=int,
                            default=None,
                            help='rows count to get from in file')
    arg_parser.add_argument('--requests-count',
                            type=int,
                            default=128,
                            help='requests count per pool')
    arg_parser.add_argument('--timeout',
                            type=int,
                            default=None,
                            help='timeout for all requests of one uid')

    args = arg_parser.parse_args()
    if args.log_file:
        log_file = open(args.log_file, 'w')
    del args.log_file

    parse_to_files(**args.__dict__)
