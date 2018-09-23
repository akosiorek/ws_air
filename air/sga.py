import itertools

import numpy as np
import tensorflow as tf
from tensorflow.contrib import kfac


def chain_sequence(seq):
    return list(itertools.chain(*seq))


def jac_vec(values, params, vecs):
    """Computes matmul(Jacobian(values), vecs), where the Jacobian is taken w.r.t. params.

    :param values: list of value tensors
    :param params: list of parameters
    :param vecs: list of vecs to be multiplied by the Jacobian
    :return:
    """
    return kfac.utils.fwd_gradients(values, params, grad_xs=vecs, stop_gradients=params)


def jac_tran_vec(ys, xs, vs):
    dydxs = tf.gradients(ys, xs, grad_ys=vs, stop_gradients=xs)
    return [tf.zeros_like(x) if dydx is None else dydx for (x, dydx) in zip(xs, dydxs)]


def get_sym_adj(grads, params):

    H_xi = jac_vec(grads, params, grads)
    Ht_xi = jac_tran_vec(grads, params, grads)

    At_xi = [0.5 * (ht - h) for (h, ht) in zip(H_xi, Ht_xi)]
    return At_xi


def list_dot_product(vals_a, vals_b):
    val = 0.
    for a, b in zip(vals_a, vals_b):
        val += tf.reduce_sum(a * b)

    return val


def sga(losses, params, align=True, eps=.1):

    # losses = [[l]*len(p) for l, p in zip(losses, params)]
    # losses, params = [chain_sequence(i) for i in (losses, params)]
    # grads = [tf.gradients(l, p)[0] for l, p in zip(losses, params)]

    grads = [tf.gradients(l, p) for l, p in zip(losses, params)]
    grads, params = [chain_sequence(i) for i in (grads, params)]
    adjustments = get_sym_adj(grads, params)

    if align:
        hamiltonian = 0.5 * sum((tf.reduce_sum(tf.square(g)) for g in grads))
        ham_grads = tf.gradients(hamiltonian, params)

        a = list_dot_product(grads, ham_grads)
        b = list_dot_product(adjustments, ham_grads)
        n_dims = sum([np.prod(p.shape.as_list()) for p in params])
        alpha = a * b / n_dims + eps
        alpha = tf.sign(alpha)
    else:
        alpha = 1.

    def adjust((a, b)):
        return a + alpha * b

    return map(adjust, zip(grads, adjustments))