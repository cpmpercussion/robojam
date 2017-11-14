Mixture Density RNN to create touchscreen performances
=========================================================================

The goal of these scripts is to train a Mixture Density RNN model of the Tiny Touch Screen corpus and to use it as to generate new performances.

Data Format.
------------

Tiny Touchscreen Performances should be stored in numpy arrays in the following format:

  [x, y, dt]
  
Where `x` and `y` are in [0,1] and `dt` is in [0,5].

Todo:
-----

- Implement freezing model for more convenient loading in server.
- Implement restart training from checkpoint
- Include links to pre-processed data for training and validation.
