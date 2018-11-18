""" Mixture Model Operations as used in Sketch RNN """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

N_MIXTURES = 24
MDN_SPLITS = 3  # (pi, sigma, mu)
N_OUTPUT_UNITS = N_MIXTURES * MDN_SPLITS


def split_tensor_to_mixture_parameters(output):
    """Split up the output nodes into three groups for Pis, Sigmas and Mus to parameterise mixture model.
    This uses eqs. 25 and 26 of Bishop's "Mixture Density Networks" """
    with tf.name_scope('1d_mixture_split'):
        logits, scales, locs = tf.split(value=output, num_or_size_splits=MDN_SPLITS, axis=1, name="1d_params")
        # softmax the mixture weights:
        pis = tf.nn.softmax(logits)
        pis = tf.clip_by_value(pis, 1e-8, 1., name="logits_1d")  # clip pis at 0.
        # Transform the sigmas to e^sigma
        scales = tf.add(tf.nn.elu(scales), 1. + 1e-8, name="scales_1d")
    return pis, scales, locs


def tf_normal(x, mu, s):
    """Returns the  probability of (x) occuring in the
    gaussian model parameterised by location mu, and variance s.
    This uses eq. 23 of Bishop's "Mixture Density Networks" """
    with tf.name_scope('1d_normal_prob'):
        epsilon = 1e-8  # protect against divide by zero.
        norm = tf.square(tf.subtract(x, mu))
        sig_square = tf.square(s)
        exp_component = tf.exp(tf.div(tf.negative(norm), 2. * sig_square + epsilon))  # div zero protection
        denominator = tf.sqrt(tf.constant(2.0 * np.pi) * sig_square) + epsilon  # div zero protection
        normal_density = tf.divide(exp_component, denominator, name='final_divide')
    return normal_density


def get_lossfunc(z_pi, z_mu, z_sigma, x_data):
    """Returns a loss function for a mixture of bivariate normal distributions given a true value.
    This uses eq. 29 of Bishop's "Mixture Density Networks"
    """
    with tf.name_scope('1d_mixture_loss'):
        summed_probs = tf_1d_mixture_prob(z_pi, z_mu, z_sigma, x_data)
        epsilon = 1e-6
        neg_log_prob = tf.negative(tf.log(summed_probs + epsilon))  # avoid log(0)
    return neg_log_prob


def tf_1d_mixture_prob(z_pi, z_mu, z_sigma, x_data):
    """ Returns the probability of x_data occurring in the mixture of 1D normals.
    This uses eq. 22 of Bishop's "Mixture Density Networks"  """
    with tf.name_scope('1d_mixture_prob'):
        kernel_probs = tf_normal(x_data, z_mu, z_sigma)
        weighted_probs = tf.multiply(kernel_probs, z_pi)
        summed_probs = tf.reduce_sum(weighted_probs, 1, keep_dims=True)
        # tf.summary.histogram("1d_kernel_probs", kernel_probs)
        # tf.summary.histogram("1d_weighted_probs", weighted_probs)
        # tf.summary.histogram("1d_summed_probs", summed_probs)
        # tf.summary.histogram("1d_kernel_weights", z_pi)
    return summed_probs


def adjust_temp(pi_pdf, temp):
    """ Adjusts temperature of a PDF describing a categorical model """
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf


def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a categorical model PDF, optionally greedily."""
    if greedy:
        return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
        accumulate += pdf[i]
        if accumulate >= x:
            return i
    tf.logging.info('Error sampling mixture model.')
    return -1


def sample_categorical(dist, temp=1.0):
    """Samples a categorical distribution with optional temp adjustment."""
    pdf = adjust_temp(np.copy(dist), temp)
    sample = np.random.multinomial(1, pdf)
    for idx, val in np.ndenumerate(sample):
        if val == 1:
            return idx[0]
    tf.logging.info('Error sampling mixture model.')
    return -1   


def sample_gaussian(mu, s, temp=1.0, greedy=False):
    if greedy:
        return mu
    s *= temp * temp
    x = np.random.normal(mu, s, 1)
    return x[0]


def sample_mixture_model(pi, mu, s, temp=1.0, greedy=False):
    """ Takes a sample from a mixture of bivariate normals, with temporature and greediness. """
    idx = sample_categorical(pi, temp)
    x1 = sample_gaussian(mu[idx], s[idx], np.sqrt(temp), greedy)
    return x1
