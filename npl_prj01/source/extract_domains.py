#!/usr/bin/env python


import json
import argparse

import pandas as pd
from find_keys import url2domain


def extract_uid_domains(in_file_path, out_file_path):
    df = pd.read_csv(in_file_path,
                     encoding='utf-8',
                     sep='\t',
                     skipinitialspace=True,
                     usecols=['uid', 'gender', 'age', 'user_json'])
    with open(out_file_path, 'w') as fd:
        fd.write('uid\tgender\tage\tdomains\n')
        for l in ('\t'.join((uid, gender, age, json.dumps(list(set([url2domain(v['url'])[0]
                                                                    for v in json.loads(user_json)['visits']])))))
                  for _, gender, age, uid, user_json in df.itertuples()):
            fd.write(l + '\n')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('--in-file-path',
                            default='gender_age_dataset.txt',
                            help='input file path')
    arg_parser.add_argument('--out-file-path',
                            default='uid_domains.csv',
                            help='out file path')

    args = arg_parser.parse_args()
    extract_uid_domains(**args.__dict__)
