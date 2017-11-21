RoboJam: A Mixture Density RNN for creating touchscreen performances
====================================================================

[![DOI](https://zenodo.org/badge/110691324.svg)](https://zenodo.org/badge/latestdoi/110691324)

RoboJam is a Mixture Density RNN and web app for creating and responding to musical touchscreen performances.
The RNN design here is a novel application of mixture density network (MDN) to musical touchscreen data.
This data consists of a sequence of touch interaction events in the format `[x, y, dt]`. 
This network learns to predict these events so that a user's interaction can be continued from where they leave off.
The web app runs uses Flask with a public API that can be used for interaction with touchscreen music apps running on phones or tablets.
More information is in the paper (to be added soon).

Here's an example:

![](https://github.com/cpmpercussion/robojam/raw/master/notebooks/example_unconditioned_1.png)

Data Format.
------------

Touchscreen performances should be stored in numpy arrays in the following format:

  [x, y, dt]
  
Where `x` and `y` are in [0,1] and `dt` is in [0,5].

Todo:
-----

- Implement freezing model for more convenient loading in server.
- Implement restart training from checkpoint
- Include links to pre-processed data for training and validation.

Examples:
---------

![](https://github.com/cpmpercussion/robojam/raw/master/notebooks/example_conditioned_1.png)

![](https://github.com/cpmpercussion/robojam/raw/master/notebooks/example_conditioned_2.png)

![](https://github.com/cpmpercussion/robojam/raw/master/notebooks/example_unconditioned_2.png)
