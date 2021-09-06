#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='bert4pytorch',
    version='0.1.3',
    description='an elegant bert4pytorch',
    long_description='bert4pytorch: ',
    license='Apache License 2.0',
    url='https://github.com/MuQiuJun-AI/bert4pytorch',
    author='MuQiuJun',
    install_requires=['torch>1.0', 'numpy>=1.17'],
    packages=find_packages()
)