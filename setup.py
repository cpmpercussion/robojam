# -*- coding: utf-8 -*-

# Setup for RoboJam

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='robojam',
    version='1.1',
    description='A Recurrent Neural Network for generating musical touchscreen data.',
    long_description=readme,
    author='Charles P. Martin',
    author_email='cpm@charlesmartin.com.au',
    url='https://github.com/cpmpercussion/robojam',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
