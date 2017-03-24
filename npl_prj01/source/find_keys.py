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
    out = []
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all("meta"):
        if tag.has_attr("name") and tag["name"] in st.META_NAMES:
            out.append(tag["content"])
    return out


def get_all_meta_content(urls):
    out = []

    for url in urls:
        try:
            no_exception = True
            response = urllib2.urlopen(url)
        # except urllib2.HTTPError:
        # except urllib2.URLError, e:
        #     print 'URLError = ' + str(e.reason)
        except Exception:
            url, url_scheme = url2domain(url)
            response = urllib2.urlopen("://".join([url_scheme, url]))
            no_exception = False

        out.extend(get_meta_content(response))

        if no_exception:
            url, url_scheme = url2domain(url)
            response = urllib2.urlopen("://".join([url_scheme, url]))
            out.extend(get_meta_content(response))
    return list(set(out))


def gen_uid_meta_file():
    fields = ['uid', 'user_json']
    df = pd.read_csv(st.MAIN_FILE_PATH, sep='\t', skipinitialspace=True, usecols=fields, nrows=5)
    # urls = list()

    with open(st.UID_META_FILE_PATH, 'w') as um_file:
        for row in df.itertuples():
            uid_visits = json.loads(row[2])
            for v in uid_visits['visits']:
                um_file.write("%s\t%s\n" % (row[1], v['url'].encode('utf-8'))) # urls.extend(v['url'])


def main():
    # res = get_all_meta_content(urls)

    # df = pd.read_csv(st.FILE_PATH, sep='\t')
    gen_uid_meta_file()

    # for i in res:
    #     print i

if __name__ == "__main__":
    main()
