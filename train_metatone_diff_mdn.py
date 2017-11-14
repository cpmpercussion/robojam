"""Trains a 2D+1D MDRNN on the Metatone _differential_ touchscreen performance data."""
from __future__ import print_function
import sample_data
import mixture_rnn
import h5py
import numpy as np
# import tensorflow as tf
import random

# Hyperparameters:
SEQ_LEN = 256
BATCH_SIZE = 128
HIDDEN_UNITS = 512
LAYERS = 3
MIXES = 16
EPOCHS = 5 # trying 10 (60 hours) should be interesting to see if this works.

# SEED = 1234
SEED = 3456 # 2345 seems to be good.

random.seed(SEED)
np.random.seed(SEED)
# tf.set_random_seed(5791)  # only works for current graph.

metatone_data_file_name = "MetatoneDifferentialTinyPerfCorpus.h5"

with h5py.File(metatone_data_file_name, 'r') as data_file:
    metatone_diff_corpus = data_file['total_performances'][:]

# load metatone data and train MDRNN from that.

print("Loading Data")
# Load Data
sequence_loader = sample_data.SequenceDataLoader(num_steps=SEQ_LEN + 1, batch_size=BATCH_SIZE, corpus=metatone_diff_corpus)

print("Loading Network")
# Setup network
net = mixture_rnn.MixtureRNN(mode=mixture_rnn.NET_MODE_TRAIN, n_hidden_units=HIDDEN_UNITS, n_mixtures=MIXES, batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, n_layers=LAYERS)

print("Training Network:", net.model_name())
# Train
losses = net.train(sequence_loader, EPOCHS, saving=True)
print(losses)

print("Done")
