import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.distributions import Geometric
from tensorflow.contrib.distributions import Normal

import ops
from cell import AIRCell
from modules import AIRDecoder
from prior import NumStepsDistribution, RecurrentNormal
import targets


class AttendInferRepeat(snt.AbstractModule):
    """Implements both the inference and the generative mdoel for AIRModel"""

    def __init__(self, n_steps, output_std, prior_step_success_prob, cell, glimpse_decoder,
                 mean_img=None, recurrent_prior=False):

        super(AttendInferRepeat, self).__init__()
        self._n_steps = n_steps
        self._output_std = output_std
        self._cell = cell

        zeros = tf.zeros(self._cell.n_what)
        self._what_prior = Normal(zeros, 1.)

        if recurrent_prior:
            with tf.variable_scope('attend_infer_repeat/air_decoder'):
                self._where_prior = RecurrentNormal(self._cell.n_where, 10)
        else:
            zeros = tf.zeros(self._cell.n_where)
            self._where_prior = Normal(zeros, 1.)

        self._num_steps_prior = Geometric(probs=1 - prior_step_success_prob)

        with self._enter_variable_scope():
            self._decoder = AIRDecoder(self._cell._img_size, self._cell._glimpse_size,
                                       glimpse_decoder, batch_dims=2, mean_img=mean_img)

    def _build(self, img, latent_override=None):
        # Inference
        # ho is hidden outputs; short name due to frequent usage
        initial_state = self._cell.initial_state(img)
        ho, hidden_state = self._unroll_timestep(initial_state, latent_override)

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

    def _unroll_timestep(self, hidden_state, latents=None):
        if latents is None:
            inpt = [tf.zeros((1, 1))] * 3
            inpt = [[False, inpt]] * self._n_steps
        else:
            latents = [tf.unstack(l, axis=-2) for l in latents]
            latents = zip(*latents)
            inpt = [[True, l] for l in latents]

        # hidden_outputs, hidden_state = tf.nn.static_rnn(self._cell, inpt, hidden_state)
        hidden_outputs = []
        for i in inpt:
            ho, hidden_state = self._cell(i, hidden_state)
            hidden_outputs.append(ho)

        hidden_outputs = ops.stack_states(hidden_outputs)
        hidden_outputs = AIRCell.outputs_by_name(hidden_outputs)
        return hidden_outputs, hidden_state[-1]

    def sample(self, sample_size=1):
        w = []
        for pdf, arg in zip((self._what_prior, self._where_prior), ([sample_size * self._n_steps], [sample_size, self._n_steps])):
            sample = pdf.sample(*arg)
            shape = [sample_size, self._n_steps] + sample.shape.as_list()[-1:]
            sample = tf.reshape(sample, shape)
            w.append(sample)
        what, where = w

        n = self._num_steps_prior.sample(sample_size)
        presence = tf.to_float(tf.sequence_mask(n, maxlen=self._n_steps))
        presence = tf.expand_dims(presence, -1)

        obs, _ = self._decoder(what, where, presence)
        return obs, what, where, presence


class Model(object):
    """Generic AIRModel model"""
    output_std = 1.
    internal_decode = False

    def __init__(self, obs, model, k_particles, target, num_objects=None, debug=False):
        """

        :param obs:
        :param model:
        :param output_multiplier:
        :param k_particles:
        :param debug:
        """
        self.obs = obs
        self.model = model
        self.k_particles = k_particles
        self.target = target
        self.gt_num_objects = num_objects
        self.debug = debug

        shape = self.obs.get_shape().as_list()
        self.batch_size = shape[0]

        self.img_size = shape[1:]
        self.tiled_batch_size = self.batch_size * self.k_particles
        self.tiled_obs = ops.tile_input_for_iwae(obs, self.k_particles, with_time=False)

        with tf.variable_scope(self.__class__.__name__):
            self._build()

    def _build(self):

        self.outputs = self.model(self.tiled_obs)

        log_weights = self.outputs.log_weights
        self.log_weights = tf.reshape(log_weights, (self.batch_size, self.k_particles))

        self.importance_weights = tf.stop_gradient(tf.nn.softmax(log_weights, -1))
        self.elbo_vae = tf.reduce_mean(self.log_weights)
        self.elbo_iwae_per_example = tf.reduce_logsumexp(self.log_weights, -1) - tf.log(float(self.k_particles))
        self.elbo_iwae = tf.reduce_mean(self.elbo_iwae_per_example)

        # if self.gt_num_objects is not None:
        #     self.gt_num_steps = tf.reduce_sum(self.gt_num_objects, -1)
        #     num_step_per_sample = self._resample(self.outputs.num_steps)
        #     self.num_step_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.gt_num_steps, num_step_per_sample)))

    def make_target(self, opt):

        if self.target == 'iwae':
            target = targets.vimco(self.log_weights, self.outputs.steps_log_prob, self.elbo_iwae_per_example)
            gvs = opt.compute_gradients(target)

        elif self.target in 'ws rws rws+sleep':
            decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope='attend_infer_repeat/air_decoder')
            encoder_vars = list(set(tf.trainable_variables()) - set(decoder_vars))

            if self.target == 'rws':
                args = self.outputs.log_p_x_and_z, self.outputs.log_q_z, self.importance_weights
                decoder_target, encoder_target = targets.reweighted_wake_wake(*args)

            elif self.target == 'rws+sleep':
                args = self.outputs.log_p_x_and_z, self.outputs.log_q_z, self.importance_weights
                decoder_target, encoder_wake_target = targets.reweighted_wake_wake(*args)

                obs, what, where, presence = self.model.sample(self.batch_size)
                sleep_outputs = self.model(obs, latent_override=[what, where, presence])

                encoder_sleep_target = -tf.reduce_mean(sleep_outputs.log_q_z)
                encoder_target = (encoder_sleep_target + encoder_wake_target) / 2.

            elif self.target == 'ws':
                obs, what, where, presence = self.model.sample(self.batch_size)
                sleep_outputs = self.model(obs, latent_override=[what, where, presence])

                args = self.outputs.log_p_x_and_z, sleep_outputs.log_q_z, 1.
                decoder_target, encoder_target = targets.reweighted_wake_wake(*args)

            target = (decoder_target + encoder_target) / 2.
            decoder_gvs = opt.compute_gradients(decoder_target, var_list=decoder_vars)
            encoder_gvs = opt.compute_gradients(encoder_target, var_list=encoder_vars)
            gvs = decoder_gvs + encoder_gvs

        else:
            raise ValueError('Invalid target: {}'.format(self.target))

        assert len(gvs) == len(tf.trainable_variables())
        for g, v in gvs:
            assert g is not None, 'Gradient for variable {} is None'.format(v)

        return target, gvs

#     def _resample(self, tensor, axis=-1):
#         return resample(tensor, self.importance_weights, self.batch_size, self.k_particles, axis)
#
#
# def resample(tensor, index, batch_size, k_particles, axis=-1):
#     """
#
#     :param tensor: tf.Tensor of shape [..., batch_size * k_particles, ...]
#     :param index: tf.Tensor of shape [batch_size * k_particles] of integers filled with numbers in [1, ..., k_particles]
#     :param batch_size:
#     :param k_particles:
#     :param axis:
#     :return:
#     """
#     index = index + tf.range(batch_size) * k_particles
#     shape = tensor.shape.as_list()
#     shape[axis] = batch_size
#     resampled = ops.gather_axis(tensor, index, axis)
#     resampled.set_shape(shape)
#     return resampled