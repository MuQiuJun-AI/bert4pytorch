#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='bert4pytorch',
    version='0.1.0',
    description='an elegant bert4pytorch',
    long_description='bert4pytorch: ',
    license='Apache License 2.0',
    url='https://github.com/MuQiuJun-AI/bert4pytorch',
    author='MuQiuJun',
    install_requires=['pytorch<=1.8.1'],
    packages=find_packages()
)