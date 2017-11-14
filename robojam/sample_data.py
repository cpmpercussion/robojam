""" Arranges sequential data in batches and epochs for applying to an RNN. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd


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
