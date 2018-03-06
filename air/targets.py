import numpy as np
import tensorflow as tf
from pynverse import inversefunc

import math

import ops


def iwae(log_weights):
    k_particles = log_weights.shape.as_list()[-1]
    return tf.reduce_logsumexp(log_weights, -1) - math.log(float(k_particles))


def vimco(log_weights, log_probs, elbo_iwae=None):
    control_variate = ops.vimco_baseline(log_weights)
    learning_signal = tf.stop_gradient(log_weights - control_variate)
    log_probs = tf.reshape(log_probs, tf.shape(log_weights))
    reinforce_target = learning_signal * log_probs

    if elbo_iwae is None:
        elbo_iwae = iwae(log_weights)

    proxy_loss = -tf.expand_dims(elbo_iwae, -1) - reinforce_target
    return tf.reduce_mean(proxy_loss)


def reinforce(log_weights, log_probs, elbo_iwae=None):
    learning_signal = tf.stop_gradient(log_weights)
    log_probs = tf.reshape(log_probs, tf.shape(log_weights))
    reinforce_target = learning_signal * log_probs

    if elbo_iwae is None:
        elbo_iwae = iwae(log_weights)

    proxy_loss = -tf.expand_dims(elbo_iwae, -1) - reinforce_target
    return tf.reduce_mean(proxy_loss)


def pynverse_find_alpha_impl(target_ess, weights):
    def ess(weights, alpha):
        weights = weights ** alpha
        res = weights.sum(-1) ** 2 / (weights ** 2).sum(-1)
        return res.mean()

    f = lambda a: ess(weights, a)
    try:
        alpha = inversefunc(f, y_values=target_ess, domain=[0, 1])
    except ValueError as err:
        try:
            target_ess = float(err.message.split(' ')[-4]) * 1.01
            if target_ess >= weights.shape[-1]:
                alpha = 1.
            else:
                alpha = inversefunc(f, y_values=target_ess, domain=[0, 1])

        except ValueError:
            alpha = 1.0

    return np.float32(alpha)


def alpha_for_ess(target_ess, log_weights):
    weights = tf.nn.softmax(log_weights)
    func = tf.py_func(pynverse_find_alpha_impl, [target_ess, weights], tf.float32)
    func.set_shape([])
    return func