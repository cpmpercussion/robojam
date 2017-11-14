# training test of MDRNN
import sample_data
import tiny_performance_loader
import mixture_density_rnn
import tensorflow as tf
import numpy as np
import random

PY_RANDOM_STATE = 8901
NP_RANDOM_STATE = 6789
TF_RANDOM_STATE = 2345

SEQ_LEN = 64
BATCH_SIZE = 64
HIDDEN_UNITS = 64
LAYERS = 1
MIXES = 16
EPOCHS = 1

random.seed(PY_RANDOM_STATE)
np.random.seed(NP_RANDOM_STATE)
tf.set_random_seed(TF_RANDOM_STATE)

# Load Data
data_loader = tiny_performance_loader.TinyPerformanceLoader(verbose=False)
tiny_performance_corpus = data_loader.single_sequence_corpus()
loader = sample_data.SequenceDataLoader(num_steps=SEQ_LEN + 1, batch_size=BATCH_SIZE, corpus=tiny_performance_corpus)
net = mixture_density_rnn.MixtureDensityRNN(mode=mixture_density_rnn.NET_MODE_TRAIN, n_hidden_units=HIDDEN_UNITS, n_mixtures=MIXES, batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, n_layers=LAYERS)
losses = net.train(loader, EPOCHS, saving=True)
print(losses)
