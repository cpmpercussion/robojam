"""Evaluate Mixture RNN using Tiny Performance Data."""
from __future__ import print_function
import sample_data
import mixture_rnn
import h5py
import numpy as np
import random
import pandas as pd

# Hyperparameters:
SEQ_LEN = 256
BATCH_SIZE = 1
HIDDEN_UNITS = 512
LAYERS = 3
MIXES = 16

# SEED = 1234
SEED = 2345  # 2345 seems to be good.

random.seed(SEED)
np.random.seed(SEED)
# tf.set_random_seed(5791)  # only works for current graph.


# model_files = [
#     "models/mdrnn-2d-1d-3layers-512units-20mixtures-20171027-170715/mdrnn-2d-1d-3layers-512units-20mixtures-8000",
#     "models/mdrnn-2d-1d-3layers-512units-20mixtures-20171027-170715/mdrnn-2d-1d-3layers-512units-20mixtures-14000",
#     "models/mdrnn-2d-1d-3layers-512units-20mixtures-20171027-170715/mdrnn-2d-1d-3layers-512units-20mixtures-20000",
#     "models/mdrnn-2d-1d-3layers-512units-20mixtures-20171027-170715/mdrnn-2d-1d-3layers-512units-20mixtures-24000",
#     "models/mdrnn-2d-1d-3layers-512units-20mixtures-20171027-170715/mdrnn-2d-1d-3layers-512units-20mixtures-26000",
#     "models/mdrnn-2d-1d-3layers-512units-20mixtures-20171027-170715/mdrnn-2d-1d-3layers-512units-20mixtures-28000",
#     "models/mdrnn-2d-1d-3layers-512units-20mixtures-20171027-170715/mdrnn-2d-1d-3layers-512units-20mixtures-30000",
#     "models/mdrnn-2d-1d-3layers-512units-20mixtures-20171026-115510/mdrnn-2d-1d-3layers-512units-20mixtures.ckpt-33579",
#     "models/mdrnn-2d-1d-3layers-512units-20mixtures-1epoch"
# ]

model_files = ['models/mdrnn-2d-1d-3layers-512units-16mixtures']


microjam_data_file_name = "TinyPerformanceCorpus.h5"
metatone_data_file_name = "MetatoneTinyPerformanceRecords.h5"

with h5py.File(microjam_data_file_name, 'r') as data_file:
    microjam_corpus = data_file['total_performances'][:]

# load metatone data and train MDRNN from that.

print("Loading Data")
# Load Data
sequence_loader = sample_data.SequenceDataLoader(num_steps=SEQ_LEN + 1, batch_size=BATCH_SIZE, corpus=microjam_corpus, overlap=False)
print("Loading Network")
# Setup network
# has to be NET_MODE_TRAIN in order to generate loss and accuracy tensors
net = mixture_rnn.MixtureRNN(mode=mixture_rnn.NET_MODE_TRAIN, n_hidden_units=HIDDEN_UNITS, n_mixtures=MIXES, batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, n_layers=LAYERS)
print("Loaded Network:", net.model_name())
experiment_results = {}


def evaluate_model_with_name(net, model_file, data_manager):
    """Evaluate a network with a certain model checkpoint on a data manager."""
    print("Evaluating:", model_file)
    l, a = net.evaluate(data_manager, model_file)
    losses = np.array(l)
    accuracies = np.array(a)
    print("Mean Loss:", losses.mean())
    print("Mean Acc:", accuracies.mean())
    return {'loss': losses.mean(), 'accuracy': accuracies.mean()}

for mf in model_files:
    result = evaluate_model_with_name(net, mf, sequence_loader)
    experiment_results[mf] = result

er_df = pd.DataFrame.from_dict(experiment_results, orient='index')
er_df.to_csv("evaluation_results.csv")

print("Done with evaluation experiment.")
