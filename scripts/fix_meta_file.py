#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, with_statement, division
from codecs import open


def main(in_file_path, out_file_path):
    with open(in_file_path, 'r', encoding='utf-8') as in_ff,\
            open(out_file_path, 'w', encoding='utf-8') as out_ff:
        while True:
            try:
                out_ff.write(next(in_ff))
            except StopIteration:
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue


if __name__ == "__main__":
    main('/Users/usual/PycharmProjects/npl_project01/data/csv/uid_meta.csv',
         '/Users/usual/PycharmProjects/npl_project01/data/csv/uid_meta_fixed.csv')
