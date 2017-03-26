#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Он содержит данные о посещении сайтов ~40 000 пользователей, при этом по некоторым из них (~ 35 000) известны их пол
#  и возрастная категория, а по 5 000 - эта информация не известна. В файле есть 4 поля:

# gender - пол, принимающий значения M (male - мужчина), F (female - женщина), - (пол неизвестен);
# age - возраст, представленный в виде диапазона x-y (строковый тип), или - (возрастная категория неизвестна);
# uid - идентификатор пользователя, строковая переменная;
# user_json - поле json, в котором содержатся записи о посещении сайтов этим пользователем (url, timestamp).


import settings as st
import pandas as pd
import json
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from urlparse import urlparse
# from urllib import unquote
import re
from urlparse import urlparse
from bs4 import BeautifulSoup
import urllib2


def url2domain(url):
    url = re.sub('(http(s)*://)+', 'http://', url)
    parsed_url = urlparse(urllib2.unquote(url.strip()))

    if parsed_url.scheme not in ['http', 'https']:
        return None
    netloc = re.search("(?:www\.)?(.*)", parsed_url.netloc).group(1)

    if netloc is not None:
        return str(netloc.encode('utf8')).strip(), parsed_url.scheme
    return None


def get_meta_content(html):
    out = list()
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all("meta"):
        if tag.has_attr("name") and tag["name"].lower() in st.META_NAMES and tag["content"].strip():
            out.append(tag["content"])
    return out


def get_all_meta_content(urls_ts, count):
    out = list()
    lost_urls = list()
    domain_url = list()

    buffer_urls = list()

    for (url, timestamp) in urls_ts:
        try:
            count += 1
            domain, url_scheme = url2domain(url)
            domain_url.extend([(domain, url, timestamp)])
            url_new = '://'.join([url_scheme, domain])
            if url_new not in buffer_urls:
                # Сначала по домену
                buffer_urls.extend([url_new])
                print "%d: Trying NEW URL %s" % (count, url_new)
                response = urllib2.urlopen(url_new)
                meta_info = get_meta_content(response)
                if meta_info:
                    out.extend(meta_info)
                    print "Got meta"
                else:
                    lost_urls.extend([(url_new, 'empty_meta')])
                    print "Empty meta"

            if url not in buffer_urls:
                # Теперь по основной ссылке
                buffer_urls.extend([url])
                print "%d: Trying %s" % (count, url)
                response = urllib2.urlopen(url)
                meta_info = get_meta_content(response)
                if meta_info:
                    out.extend(meta_info)
                    print "Got meta"
                else:
                    lost_urls.extend([(url, 'empty_meta')])
                    print "Empty meta"
        # except urllib2.HTTPError:
        # except urllib2.URLError, e:
        #     print 'URLError = ' + str(e.reason)
        except Exception:
            lost_urls.extend([(url, 'bad_url')])
            print "Bad url"
    return list(set(out)), domain_url, lost_urls, count


def parse_to_files(nrows=None):
    fields = ['uid', 'user_json']
    df = pd.read_csv(st.MAIN_FILE_PATH, sep='\t', skipinitialspace=True, usecols=fields, nrows=nrows)
    urls_ts = list()

    with open(st.UID_META_FILE_PATH, 'w') as um_file,\
            open(st.UID_LOST_URLS_FILE_PATH, 'w') as lu_file,\
            open(st.UID_DOMAIN_URL_TIMESTAMP_FILE_PATH, 'w') as dut_file:
        count = 0
        for row in df.itertuples():
            uid = row[1]
            uid_visits = json.loads(row[2])
            for v in uid_visits['visits']:
                timestamp = v['timestamp']
                urls_ts.extend([(v['url'], timestamp)])

            print "\nUser %s\n------------------" % uid
            meta_data, domains_urls, lost_urls, count = get_all_meta_content(urls_ts, count)
            for md in meta_data:
                um_file.write("%s\t%s\n" % (uid, md.encode('utf-8')))
            for (dom, url, ts) in domains_urls:
                dut_file.write("%s\t%s\t%s\t%s\n" % (uid, dom.encode('utf-8'), url.encode('utf-8'), ts))
            for (url, err) in lost_urls:
                lu_file.write("%s\t%s\t%s\n" % (uid, url.encode('utf-8'), err))


def main():
    parse_to_files(2)


if __name__ == "__main__":
    main()
