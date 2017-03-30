#!/usr/bin/env python
# -*- coding: utf-8 -*-

import trollius as asyncio
import concurrent.futures
import grequests
from bs4 import BeautifulSoup
import pandas as pd
import json
import logging
import sys
import multiprocessing

META_NAMES = ['description',
              'og:description',
              'twitter:description'
              'keywords',
              'title',
              'og:title',
              'twitter:title']


def get_meta_content(html):
    return [tag["content"].strip()
            for tag in BeautifulSoup(html, "lxml").find_all("meta")
            if tag.has_attr("name")
            if tag["name"].lower() in META_NAMES
            if tag["content"].strip()]


def req(dt, timeout=None, requests_count=128):
    (urls, n) = dt
    log = logging.getLogger('blocks(%d)' % n)
    log.info('running')

    for _urls in (urls[i:i + requests_count] for i in xrange(0, len(urls), requests_count)):
        requests = (grequests.get(u) for u in _urls)
        responses = grequests.map(requests=requests, gtimeout=timeout)

        for i, response in enumerate(responses):
            try:
                meta_info = get_meta_content(response._content)
                response.close()
                for mi in meta_info:
                    print (u'URL %s: %s' % (_urls[i], mi)).encode('utf-8')
            except Exception as ex:
                print (u'ERROR:\t%s - %s' % (_urls[i], ex)).encode('utf-8')

    log.info('done')


@asyncio.coroutine
def run_blocking_tasks(ex):
    log = logging.getLogger('run_blocking_tasks')
    log.info('starting')
    log.info('creating executor tasks')

    u_urls = []
    df = pd.read_csv('/Users/usual/PycharmProjects/npl_project01/npl_prj01/data/gender_age_dataset1.txt',
                     encoding='utf-8',
                     sep='\t',
                     skipinitialspace=True,
                     usecols=['uid', 'user_json'],
                     nrows=None,
                     )
    user_json = json.loads(df.iloc[0]['user_json'])
    urls = [v['url'] for v in user_json['visits']]
    batch = 15
    step = len(urls)/batch
    start = 0
    for pos in xrange(step, step*(batch+1), step):
        u_urls.append(urls[start:pos])
        start = pos

    if len(urls)/(len(u_urls)*len(u_urls[0])):
        u_urls[len(u_urls)-1].extend(urls[start:len(urls)-1])

    loop = asyncio.get_event_loop()
    blocking_tasks = []
    n = 0
    for u in u_urls:
        n += 1
        dd = (u, n)
        blocking_tasks.append(loop.run_in_executor(ex, req, dd))

    log.info('waiting for executor tasks')
    completed, pending = yield asyncio.wait(blocking_tasks)
    results = [t.result() for t in completed]
    log.info("RESULTS: '%s'" % results)
    log.info('exiting')


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='PID %(process)5s %(name)18s: %(message)s',
        stream=sys.stderr,
    )

    # Create a limited thread pool.
    max_workers = multiprocessing.cpu_count()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    event_loop = asyncio.get_event_loop()
    try:
        event_loop.run_until_complete(run_blocking_tasks(executor))
    finally:
        event_loop.close()

# max_workers=multiprocessing.cpu_count()

