from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
# import pandas as pd
import tensorflow as tf
from robojam import mixture_2d_normals
from robojam import mixture_1d_normals

sess = tf.InteractiveSession()

print("Test 1D Normal Model")

u1 = tf.constant([1e-16])
s1 = tf.constant([1e-8])
x1 = tf.constant([1e-16])

prob = mixture_1d_normals.tf_normal(x1, u1, s1)

print(prob.eval())

print("Test 1D Mixture Model")

p2 = tf.constant([0.2, 0.2, 0.2, 0.4])
u2 = tf.constant([1.0, 2.0, 3.0, 4.0])
s2 = tf.constant([1.0, 1.0, 1.0, 1.0])
x2 = tf.constant([1.0])

# prob2 = mixture_1d_normals.tf_1d_mixture_prob(p2,u2,s2,x2)

prob2 = mixture_1d_normals.tf_normal(x2, u2, s2)
prob3 = tf.multiply(prob2, p2)
prob4 = tf.reduce_sum(prob3, 0, keep_dims=True)

print(prob2.eval())
print(prob3.eval())
print(prob4.eval())

print("Test 2D Normal Model")

mu1 = tf.constant([1.0])
mu2 = tf.constant([2.0])
s1 = tf.constant([1e-8])
s2 = tf.constant([1e-8])
rho = tf.constant([1.0 - (1e-16)])  # rho cannot be "1."
x1 = tf.constant([1.0])
x2 = tf.constant([2.0])

prob = mixture_2d_normals.tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho)

print(prob.eval())

print("Test 2D Normal Model Again!")

epsilon = 1e-8  # protect against divide by zero.
norm1 = tf.subtract(x1, mu1)
norm2 = tf.subtract(x2, mu2)
s1s2 = tf.multiply(s1, s2)
z = (tf.square(tf.divide(norm1, s1)) + tf.square(tf.divide(norm2, s2)) -
     2. * tf.divide(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
one_neg_rho_sq = 1 - tf.square(rho)
two_one_neg_rho_sq = 2. * one_neg_rho_sq + epsilon  # possible div zero
result_rhs = tf.exp(tf.divide(-z, two_one_neg_rho_sq))
denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(one_neg_rho_sq)) + epsilon  # possible div zero
normal_x_usp = tf.divide(result_rhs, denom)

print("two_one_neg_rho_sq", two_one_neg_rho_sq.eval())
print("result_rhs", result_rhs.eval())
print("s1s2", s1s2.eval())
print("denom", denom.eval())
print("normal_x_usp", normal_x_usp.eval())
