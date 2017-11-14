""" Functions for building a Mixture Density Network using Tensorflow's MultivariateNormal and Mixture distributions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.distributions import Categorical, Mixture, MultivariateNormalDiag


N_MIXTURES = 24
MDN_SPLITS = 7  # (pi, sigma_1, sigma_2, sigma_3, mu_1, mu_2, mu_3)
N_OUTPUT_UNITS = N_MIXTURES * MDN_SPLITS


def split_tensor_to_mixture_parameters(output):
    """ Split up the output nodes into three groups for Pis, Sigmas and Mus to parameterise mixture model.
    Pis (logits) are transformed by a softmax function, scales are transformed by the exponent function to ensure they are positive."""
    with tf.name_scope('mixture_split'):
        logits, scales_1, scales_2, scales_3, locs_1, locs_2, locs_3 = tf.split(value=output, num_or_size_splits=MDN_SPLITS, axis=1)
        # softmax the mixture weights:
        logits = tf.nn.softmax(logits)
        logits = tf.clip_by_value(logits, 1e-8, 1.)
        # Transform the sigmas to e^sigma
        scales_1 = tf.add(tf.nn.elu(scales_1), 1. + 1e-8)  # tf.add(tf.exp(scales_1), 0.00001)
        scales_2 = tf.add(tf.nn.elu(scales_2), 1. + 1e-8)  # tf.add(tf.exp(scales_2), 0.00001)
        scales_3 = tf.add(tf.nn.elu(scales_3), 1. + 1e-8)  # tf.add(tf.exp(scales_3), 0.00001)
    return logits, scales_1, scales_2, scales_3, locs_1, locs_2, locs_3


def get_mixture_model(logits, locs_1, locs_2, locs_3, scales_1, scales_2, scales_3, input_shape):
    with tf.name_scope('mixture_model'):
        cat = Categorical(logits=logits)
        locs = tf.stack([locs_1, locs_2, locs_3], axis=1)
        scales = tf.stack([scales_1, scales_2, scales_3], axis=1)
        coll = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(tf.unstack(locs, axis=-1), tf.unstack(scales, axis=-1))]
        # init_value = tf.zeros(input_shape, dtype=tf.float32)
        mixture = Mixture(cat=cat, components=coll, allow_nan_stats=False)  # , value=init_value)
        tf.logging.info('Mixture allows NaN stats: ' + str(mixture.allow_nan_stats))
    return mixture


def get_loss_func(mixture, Y):
    with tf.name_scope('mixture_loss'):
        loss = mixture.log_prob(Y)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
    return loss


def sample_mixture_model(mixture):
    return mixture.sample()
