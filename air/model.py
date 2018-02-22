import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.distributions import Geometric
from tensorflow.contrib.distributions import Normal

import ops
from cell import AIRCell
from modules import AIRDecoder
from prior import NumStepsDistribution


class AttendInferRepeat(snt.AbstractModule):
    """Implements both the inference and the generative mdoel for AIRModel"""

    def __init__(self, n_steps, batch_size, output_std, prior_step_success_prob, cell, glimpse_decoder):
        super(AttendInferRepeat, self).__init__()
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._output_std = output_std
        self._cell = cell

        self._what_prior = Normal(0., 1.)
        self._where_prior = Normal(0., 1.)
        self._num_steps_prior = Geometric(probs=1 - prior_step_success_prob)

        with self._enter_variable_scope():
            self._decoder = AIRDecoder(self._cell._img_size, self._cell._glimpse_size, glimpse_decoder, batch_dims=2)

    def _build(self, img):
        # Inference
        # ho is hidden outputs; short name due to frequent usage
        initial_state = self._cell.initial_state(img)
        ho, hidden_state = self._unroll_timestep(initial_state)

        # Generation
        latents = [ho[i] for i in 'what where presence'.split()]
        canvas, glimpse = self._decoder(*latents)

        ho['canvas'] = canvas
        ho['glimpse'] = glimpse
        ho['data_ll_per_pixel'] = Normal(canvas, self._output_std).log_prob(img)
        ho['data_ll'] = tf.reduce_sum(ho.data_ll_per_pixel, (-2, -1))

        # Post-processing
        num_steps_posterior = NumStepsDistribution(tf.nn.sigmoid(ho.presence_logit[..., 0]))
        ho['num_steps'] = tf.reduce_sum(ho.presence[..., 0], -1)
        ho['steps_log_prob'] = num_steps_posterior.log_prob(ho.num_steps)
        ho['steps_prior_log_prob'] = self._num_steps_prior.log_prob(ho.num_steps)

        for name in 'what where'.split():
            sample = ho[name]
            prior_log_prob = getattr(self, '_{}_prior'.format(name)).log_prob(sample)
            prior_log_prob = tf.reduce_sum(prior_log_prob, -1, keep_dims=True)
            ho['{}_prior_log_prob'.format(name)] = prior_log_prob

        # TODO: consider using Bernoulli log-probs for steps instead of the NumStep ones
        # use presence to mask entries in log_prob
        pres = ho.presence
        log_p_z = ho.what_prior_log_prob + ho.where_prior_log_prob
        log_p_z = tf.squeeze(tf.reduce_sum(log_p_z * pres, -2), -1) + ho.steps_prior_log_prob
        ho['log_p_x_and_z'] = ho.data_ll + log_p_z

        log_q_z = ho.what_log_prob + ho.where_log_prob
        ho['log_q_z'] = tf.squeeze(tf.reduce_sum(log_q_z * pres, -2), -1) + ho.steps_log_prob

        ho['log_weights'] = ho.log_p_x_and_z - ho.log_q_z

        return ho

    def _unroll_timestep(self, hidden_state):
        inpt = [tf.zeros((self._batch_size, 1))] * self._n_steps
        hidden_outputs, hidden_state = tf.nn.static_rnn(self._cell, inpt, hidden_state)
        hidden_outputs = ops.stack_states(hidden_outputs)
        hidden_outputs = AIRCell.outputs_by_name(hidden_outputs)
        return hidden_outputs, hidden_state[-1]
