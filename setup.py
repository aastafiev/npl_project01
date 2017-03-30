#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(name='npl_project01',
      version='1.0',
      entry_points={
          'console_scripts': [
              'find_keys = scripts.find_keys:main',
              'extract_domains = scripts.extract_domains:main',
          ]
      },
      packages=find_packages())
