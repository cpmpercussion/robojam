"""Evaluate Mixture RNN using Tiny Performance Data."""
from __future__ import print_function
import numpy as np
import pandas as pd
import random
import h5py
import robojam

# Hyperparameters:
SEQ_LEN = 256
BATCH_SIZE = 10
HIDDEN_UNITS = 512
LAYERS = 3
MIXES = 16

# random seed
SEED = 2345  # 2345 seems to be good.
random.seed(SEED)
np.random.seed(SEED)

# model file to evaluate
model_files = ['models/mdrnn-2d-1d-3layers-512units-16mixtures']
# load validation data
validation_corpus_name = "datasets/TinyPerformanceCorpus.h5"
with h5py.File(validation_corpus_name, 'r') as data_file:
    validation_corpus = data_file['total_performances'][:]
sequence_loader = robojam.SequenceDataLoader(num_steps=SEQ_LEN + 1, batch_size=BATCH_SIZE, corpus=validation_corpus, overlap=False)


# Setup network
# has to be NET_MODE_TRAIN in order to generate loss and accuracy tensors
net = robojam.MixtureRNN(mode=robojam.NET_MODE_TRAIN, n_hidden_units=HIDDEN_UNITS, n_mixtures=MIXES, batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, n_layers=LAYERS)
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
