#!/bin/sh

mkdir datasets
cd datasets
#wget http://folk.uio.no/charlepm/datasets/TinyPerformanceCorpus.h5
#wget http://folk.uio.no/charlepm/datasets/MetatoneTinyPerformanceRecords.h5
wget http://folk.uio.no/charlepm/datasets/tiny_performance_datasets.npz
wget http://folk.uio.no/charlepm/datasets/metatone_dataset.npz
cd ..
