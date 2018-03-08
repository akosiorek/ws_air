import numpy as np
import sonnet as snt
import tensorflow as tf
from attrdict import AttrDict
from tensorflow.contrib.distributions import Bernoulli, Normal

from modules import SpatialTransformer, ParametrisedGaussian
from neural import MLP


class AIRCell(snt.RNNCore):
    """Implements the inference model for AIRModel"""

    _n_transform_param = 4
    _output_names = 'what what_loc what_scale what_log_prob where where_loc where_scale where_log_prob presence' \
                    ' presence_logit presence_log_prob'.split()
    _what_loc_mult = 1.
    _what_scale_bias = 0.
    _initial_state = None

    _latent_name_to_idx = dict(
        what=0,
        where=4,
        pres=8,
        pres_logit=9,
    )

    def __init__(self, img_size, glimpse_size, n_what,
                 rnn, input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                 gradients_through_z=True,
                 debug=False):
        """Creates the cell

        :param img_size: int tuple, size of the image
        :param glimpse_size: int tuple, size of the attention glimpse
        :param n_what: number of latent units describing the "what"
        :param rnn: an RNN cell for maintaining the internal hidden state
        :param input_encoder: callable, encodes the original input image before passing it into the transition
        :param glimpse_encoder: callable, encodes the glimpse into latent representation
        :param transform_estimator: callabe, transforms the hidden state into parameters for the spatial transformer
        :param steps_predictor: callable, predicts whether to take a step
        :param debug: boolean, adds checks for NaNs in the inputs to distributions
        """

        super(AIRCell, self).__init__()
        self._img_size = img_size
        self._n_pix = np.prod(self._img_size)
        self._glimpse_size = glimpse_size
        self._n_what = n_what
        self._rnn = rnn
        self._n_hidden = int(self._rnn.output_size[0])

        self._gradients_through_z = gradients_through_z
        self._debug = debug

        with self._enter_variable_scope():
            self._spatial_transformer = SpatialTransformer(img_size, glimpse_size)

            self._transform_estimator = transform_estimator()
            self._input_encoder = input_encoder()
            self._glimpse_encoder = glimpse_encoder()

            self._what_distrib = ParametrisedGaussian(n_what,
                                                      loc_mult=self._what_loc_mult,
                                                      scale_offset=self._what_scale_bias,
                                                      validate_args=self._debug, allow_nan_stats=not self._debug)
            self._steps_predictor = steps_predictor()

    @property
    def n_what(self):
        return self._n_what

    @property
    def n_where(self):
        return self._n_transform_param

    @property
    def state_size(self):
        return [
            np.prod(self._img_size),  # image
            self._n_what,  # what
            self._n_transform_param,  # where
            1,  # presence
            self._rnn.state_size,  # hidden state of the rnn
        ]

    @property
    def output_size(self):
        return [
            self._n_what,  # what
            self._n_what,  # what loc
            self._n_what,  # what scale
            1,  # what sample log prob
            self._n_transform_param,  # where
            self._n_transform_param,  # where loc
            self._n_transform_param,  # where scale
            1,  # where sample log prob
            1,  # presence
            1,  # presence logit
            1   # presence sample log prob
        ]

    @property
    def output_names(self):
        return self._output_names

    @staticmethod
    def outputs_by_name(hidden_outputs):
        return AttrDict({n: o for n, o in zip(AIRCell._output_names, hidden_outputs)})

    def initial_state(self, img, hidden_state=None):
        batch_size = img.get_shape().as_list()[0]

        if self._initial_state is None:
            # if hidden_state is None:
            hidden_state = self._rnn.initial_state(batch_size, tf.float32, trainable=True)

            where_code = tf.zeros([1, self._n_transform_param], dtype=tf.float32, name='where_init')
            what_code = tf.zeros([1, self._n_what], dtype=tf.float32, name='what_init')

            where_code, what_code = (tf.tile(i, (batch_size, 1)) for i in (where_code, what_code))


            init_presence = tf.ones((batch_size, 1), dtype=tf.float32)
            self._initial_state = [what_code, where_code, init_presence, hidden_state]

        flat_img = tf.reshape(img, (batch_size, self._n_pix))
        return [flat_img] + self._initial_state

    def _build(self, inpt, state):
        """

        :param inpt: tuple of (boolean, [tf.Tensor] * 3); if the boolean is True, tensors are used as latent samples
            (new latents are NOT sampled).
        :param state: see self.initial_state
        :return:
        """
        img_flat, what, where, presence, hidden_state = state
        img = tf.reshape(img_flat, (-1,) + tuple(self._img_size))

        do_reuse_samples, reuse_samples = inpt
        if not do_reuse_samples:
            reuse_samples = [None] * 3
        what_o, where_o, presence_o = reuse_samples

        with tf.variable_scope('rnn_inpt'):
            rnn_inpt = self._input_encoder(img)
            rnn_inpt = [rnn_inpt, what, where, presence]
            rnn_inpt = tf.concat(rnn_inpt, -1)
            hidden_output, hidden_state = self._rnn(rnn_inpt, hidden_state)

        with tf.variable_scope('where'):
            where, where_loc, where_scale, where_log_prob = self._compute_where(hidden_output, where_o)

        with tf.variable_scope('presence'):
            presence, presence_logit, presence_log_prob \
                = self._compute_presence(presence, hidden_output, presence_o)

        with tf.variable_scope('what'):
            what, what_loc, what_scale, what_log_prob = self._compute_what(img, where, what_o)

        output = [
            what, what_loc, what_scale, what_log_prob,
            where, where_loc, where_scale, where_log_prob,
            presence, presence_logit, presence_log_prob
        ]

        new_state = [img_flat, what, where, presence, hidden_state]

        return output, new_state

    def _compute_where(self, hidden_output, reuse_sample):
        loc, scale = self._transform_estimator(hidden_output)
        scale = tf.nn.softplus(scale) + 1e-4
        where_distrib = Normal(loc, scale, validate_args=self._debug, allow_nan_stats=not self._debug)
        sample, sample_log_prob = self._maybe_sample(where_distrib, reuse_sample)
        return sample, where_distrib.loc, where_distrib.scale, sample_log_prob

    def _compute_presence(self, presence, hidden_output, reuse_sample):
        presence_logit = self._steps_predictor(hidden_output)

        presence_distrib = Bernoulli(logits=presence_logit, dtype=tf.float32,
                                     validate_args=self._debug, allow_nan_stats=not self._debug)
        new_presence, sample_log_prob = self._maybe_sample(presence_distrib, reuse_sample)
        presence *= new_presence

        return presence, presence_logit, sample_log_prob

    def _compute_what(self, img, where_code, reuse_sample):

        cropped = self._spatial_transformer(img, logits=where_code)
        flat_crop = snt.BatchFlatten()(cropped)

        what_params = self._glimpse_encoder(flat_crop)
        what_distrib = self._what_distrib(what_params)
        sample, sample_log_prob = self._maybe_sample(what_distrib, reuse_sample)
        return sample, what_distrib.loc, what_distrib.scale, sample_log_prob

    def _maybe_sample(self, distrib, reuse_sample=None):
        """Take a sample from the pdf `distrib` if `reuse_sample` is None. If `reuse_sample` is given, it is usead
        instead. The method also evaluates the log probability of the sample and optionally blocks the gradient flow
        through it.

        :param distrib: tf.Distribution
        :param reuse_sample: tf.Tensor or None
        :return: tuple of (tf.Tensor, tf.Tensor) representing a sample and its log probability
        """
        if reuse_sample is None:
            reuse_sample = distrib.sample()

        if not self._gradients_through_z:
            reuse_sample = tf.stop_gradient(reuse_sample)

        return reuse_sample, tf.reduce_sum(distrib.log_prob(reuse_sample), -1, keep_dims=True)
