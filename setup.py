# -*- coding: utf-8 -*-

# Setup for RoboJam

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='robojam',
    version='0.1.0',
    description='A Recurrent Neural Network for generating musical touchscreen data.',
    long_description=readme,
    author='Charles P. Martin',
    author_email='charlepm@ifi.uio.no',
    url='https://github.com/cpmpercussion/robojam',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
