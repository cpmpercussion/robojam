""" Test Initialisation and training of MDRNN."""

import robojam
import numpy as np
import random


def test_network_initialisation():
    """Test Initialising the MDRNN."""
    # Hyperparameters:
    SEQ_LEN = 256
    BATCH_SIZE = 128
    HIDDEN_UNITS = 512
    LAYERS = 3
    MIXES = 16
    # Random seed
    SEED = 2345
    random.seed(SEED)
    np.random.seed(SEED)
    robojam.RANDOM_SEED = SEED
    print("Loading Network")
    # Setup network
    net = robojam.MixtureRNN(mode=robojam.NET_MODE_TRAIN, n_hidden_units=HIDDEN_UNITS, n_mixtures=MIXES, batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, n_layers=LAYERS)
    print("Done.")

if __name__ == '__main__':
    test_network_initialisation()
