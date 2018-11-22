""" Arranges sequential data in batches and epochs for applying to an RNN. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import random


def batch_generator(seq_len, batch_size, dim, corpus):
    """Returns a generator to cut up datasets into batches of features and labels."""
    # Create empty arrays to contain batch of features and labels#
    # # Produce the generator for training
    # generator = batch_generator(SEQ_LEN, BATCH_SIZE, 3, corpus)
    batch_X = np.zeros((batch_size, seq_len, dim))
    batch_y = np.zeros((batch_size, dim))
    while True:
        for i in range(batch_size):
            # choose random example
            l = random.choice(corpus)
            last_index = len(l) - seq_len - 1
            start_index = np.random.randint(0, high=last_index)
            batch_X[i] = l[start_index:start_index+seq_len]
            batch_y[i] = l[start_index+1:start_index+seq_len+1]  # .reshape(1,dim)
        yield batch_X, batch_y


# Functions for slicing up data
def slice_sequence_examples(sequence, num_steps, step_size=1):
    """ Slices a sequence into examples of length num_steps with step size step_size."""
    xs = []
    for i in range((len(sequence) - num_steps) // step_size + 1):
        example = sequence[(i * step_size): (i * step_size) + num_steps]
        xs.append(example)
    return xs


def seq_to_overlapping_format(examples):
    """Takes sequences of seq_len+1 and returns overlapping
    sequences of seq_len."""
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[1:])
    return (xs, ys)


def seq_to_singleton_format(examples):
    """Return the examples in seq to singleton format.
    """
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[-1])
    return (xs, ys)


def generate_synthetic_3D_data():
    """
    Generates some slightly fuzzy sine wave data in through dimensions (plus time).
    """
    NSAMPLE = 50000
    print("Generating", str(NSAMPLE), "toy data samples.")
    t_data = np.float32(np.array(range(NSAMPLE)) / 10.0)
    t_interval = t_data[1] - t_data[0]
    t_r_data = np.random.normal(0, t_interval / 20.0, size=NSAMPLE)
    t_data = t_data + t_r_data
    r_data = np.random.normal(size=NSAMPLE)
    x_data = (np.sin(t_data) + (r_data / 10.0) + 1) / 2.0
    y_data = (np.sin(t_data * 3.0) + (r_data / 10.0) + 1) / 2.0
    df = pd.DataFrame({'a': x_data, 'b': y_data, 't': t_data})
    df.t = df.t.diff()
    df.t = df.t.fillna(1e-4)
    print(df.describe())
    return np.array(df)


# TODO: is this class still relevant?


class SequenceDataLoader(object):
    """Manages data from a single sequence and generates epochs"""

    def __init__(self, num_steps, batch_size, corpus, overlap=True):
        """
        Load a data corpus and generate training examples.
        This class only works for data that is represented as
        long sequence.

        Keyword arguments:
        num_steps -- Length in steps of each example.
        batch_size -- The number of examples in each batch for generating epochs.
        corpus -- A sequence of data points used as the training data.
        """
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.corpus = corpus
        if overlap:
            self.examples = self.setup_training_examples()
        else:
            self.examples = self.non_overlapping_examples()
        print("Done initialising loader.")

    def setup_training_examples(self):
        """
        Generates training examples of length num_steps
        from the data corpus.
        """
        xs = []
        for i in range(len(self.corpus) - self.num_steps - 1):
            example = self.corpus[i: i + self.num_steps]
            xs.append(example)
        print("Total training examples:", str(len(xs)))
        return xs

    def next_epoch(self):
        """
        Return an epoch of batches of shuffled examples.
        """
        np.random.shuffle(self.examples)
        batches = []
        for i in range(len(self.examples) // self.batch_size):
            batch = self.examples[i * self.batch_size: (i + 1) * self.batch_size]
            batches.append(batch)
        return(np.array(batches))

    def non_overlapping_examples(self):
        """
        Generates a corpus of non-overlapping examples.
        """
        xs = []
        for i in range(len(self.corpus) // self.num_steps):
            example = self.corpus[i * self.num_steps: (i * self.num_steps) + self.num_steps]
            xs.append(example)
        print("Total non-overlapping examples:", str(len(xs)))
        return xs

    def seq_to_singleton_format(self):
        """
        Return the examples in seq to singleton format.
        """
        xs = []
        ys = []
        for ex in self.examples:
            xs.append(ex[:-1])
            ys.append(ex[-1])
        return (xs,ys)
