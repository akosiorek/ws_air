import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import sonnet as snt

from ops import clip_preserve, sample_from_tensor, anneal_weight


def masked_apply(tensor, op, mask):
    """Applies `op` to tensor only at locations indicated by `mask` and sets the rest to zero.

    Similar to doing `tensor = tf.where(mask, op(tensor), tf.zeros_like(tensor))` but it behaves correctly
    when `op(tensor)` is NaN or inf while tf.where does not.

    :param tensor: tf.Tensor
    :param op: tf.Op
    :param mask: tf.Tensor with dtype == bool
    :return: tf.Tensor
    """
    chosen = tf.boolean_mask(tensor, mask)
    applied = op(chosen)
    idx = tf.to_int32(tf.where(mask))
    result = tf.scatter_nd(idx, applied, tf.shape(tensor))
    return result


def geometric_prior(success_prob, n_steps):
    # clipping here is ok since we don't compute gradient wrt success_prob
    success_prob = tf.clip_by_value(success_prob, 1e-7, 1. - 1e-15)
    geom = tfd.Geometric(probs=1. - success_prob)
    events = tf.range(n_steps + 1, dtype=geom.dtype)
    probs = geom.prob(events)
    return probs, geom


def _cumprod(tensor, axis=0):
    """A custom version of cumprod to prevent NaN gradients when there are zeros in `tensor`
    as reported here: https://github.com/tensorflow/tensorflow/issues/3862

    :param tensor: tf.Tensor
    :return: tf.Tensor
    """
    transpose_permutation = None
    n_dim = len(tensor.get_shape())
    if n_dim > 1 and axis != 0:

        if axis < 0:
            axis += n_dim

        transpose_permutation = np.arange(n_dim)
        transpose_permutation[-1], transpose_permutation[0] = 0, axis

    tensor = tf.transpose(tensor, transpose_permutation)

    def prod(acc, x):
        return acc * x

    prob = tf.scan(prod, tensor)
    tensor = tf.transpose(prob, transpose_permutation)
    return tensor


def bernoulli_to_modified_geometric(presence_prob):
    presence_prob = tf.cast(presence_prob, tf.float64)
    inv = 1. - presence_prob
    prob = _cumprod(presence_prob, axis=-1)
    modified_prob = tf.concat([inv[..., :1], inv[..., 1:] * prob[..., :-1], prob[..., -1:]], -1)
    modified_prob /= tf.reduce_sum(modified_prob, -1, keep_dims=True)
    return tf.cast(modified_prob, tf.float32)


class NumStepsDistribution(object):
    """Probability distribution used for the number of steps

    Transforms Bernoulli probabilities of an event = 1 into p(n) where n is the number of steps
    as described in the AIRModel paper."""

    def __init__(self, probs):
        """

        :param probs: tensor; Bernoulli success probabilities
        """
        self._steps_probs = probs
        self._joint = bernoulli_to_modified_geometric(probs)
        self._bernoulli = None

    def sample(self, n=None):
        if self._bernoulli is None:
            self._bernoulli = tfd.Bernoulli(self._steps_probs)

        sample = self._bernoulli.sample(n)
        sample = tf.cumprod(sample, tf.rank(sample) - 1)
        sample = tf.reduce_sum(sample, -1)
        return sample

    def prob(self, samples):
        probs = sample_from_tensor(self._joint, samples)
        return probs

    def log_prob(self, samples):
        prob = self.prob(samples)
        
        prob = clip_preserve(prob, 1e-16, 1.)
        return tf.log(prob)

    @property
    def probs(self):
        return self._joint


def geom_success_prob(init_step_success_prob, final_step_success_prob):
    hold_init = 1e3
    steps_div = 1e4
    anneal_steps = 1e5
    global_step = tf.train.get_or_create_global_step()
    steps_prior_success_prob = anneal_weight(init_step_success_prob, final_step_success_prob, 'exp',
                                                 global_step,
                                                 anneal_steps, hold_init, steps_div)
    return tf.to_float(steps_prior_success_prob)


class RecurrentNormalImpl(snt.AbstractModule):

    def __init__(self, n_dim, n_hidden):
        super(RecurrentNormalImpl, self).__init__()
        self._n_dim = n_dim
        self._n_hidden = n_hidden

        with self._enter_variable_scope():
            self._rnn = snt.VanillaRNN(self._n_dim)
            self._readout = snt.Linear(n_dim * 2)
            self._init_state = self._rnn.initial_state(1, trainable=True)
            self._init_sample = tf.zeros((1, self._n_hidden))

    def _build(self, batch_size=1, seq_len=1, override_samples=None):

        s = self._init_sample, self._init_state
        sample, state = (tf.tile(ss, (batch_size, 1)) for ss in s)

        outputs = [[] for _ in xrange(4)]
        if override_samples is not None:
            override_samples = tf.unstack(override_samples, axis=-2)
            seq_len = len(override_samples)

        for i in xrange(seq_len):

            if override_samples is None:
                override_sample = None
            else:
                override_sample = override_samples[i]

            results = self._forward(sample, state, override_sample)

            for res, output in zip(results, outputs):
                output.append(res)

        return [tf.stack(o, axis=-2) for o in outputs]

    def _forward(self, sample_m1, hidden_state, sample=None):
        output, state = self._rnn(sample_m1, hidden_state)
        stats = self._readout(output)
        loc, scale = tf.split(stats, 2, -1)
        scale = tf.nn.softplus(scale)
        pdf = tfd.Normal(loc, scale)

        if sample is None:
            sample = pdf.sample()

        return sample, loc, scale, pdf.log_prob(sample)


class RecurrentNormal(object):

    def __init__(self, n_dim, n_hidden):
        self._impl = RecurrentNormalImpl(n_dim, n_hidden)

    def log_prob(self, samples):
        batch_size = samples.shape.as_list()[0]
        _, _, _, logprob = self._impl(batch_size=batch_size, override_samples=samples)
        return logprob

    def sample(self, sample_size=1, length=1):
        samples, _, _, _ = self._impl(batch_size=sample_size, seq_len=length)
        return samples



