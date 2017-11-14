"""Trains a 2D+1D MDRNN on the MicroJam touchscreen performance data."""
from __future__ import print_function
import sample_data
import mixture_rnn
import tiny_performance_loader
import numpy as np
# import tensorflow as tf
import random

random.seed(1357)
np.random.seed(2468)
# tf.set_random_seed(5791)  # only works for current graph.

# Hyperparameters:
SEQ_LEN = 64
BATCH_SIZE = 64
HIDDEN_UNITS = 64
LAYERS = 2
MIXES = 8
EPOCHS = 1

print("Loading Network")

# Load Data
data_loader = tiny_performance_loader.TinyPerformanceLoader(verbose=False)
tiny_performance_corpus = data_loader.single_sequence_corpus()
sequence_loader = sample_data.SequenceDataLoader(num_steps=SEQ_LEN + 1, batch_size=BATCH_SIZE, corpus=tiny_performance_corpus)
# Network
net = mixture_rnn.MixtureRNN(mode=mixture_rnn.NET_MODE_TRAIN, n_hidden_units=HIDDEN_UNITS, n_mixtures=MIXES, batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, n_layers=LAYERS)
# Train

print("Ready to train")

losses = net.train(sequence_loader, EPOCHS, saving=True)
print(losses)

print("Done")
