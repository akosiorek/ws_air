import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.distributions as tfd

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
        self._what_prior = tfd.Normal(zeros, 1.)

        if recurrent_prior:
            with tf.variable_scope('attend_infer_repeat/air_decoder'):
                self._where_prior = RecurrentNormal(self._cell.n_where, 10)
        else:
            zeros = tf.zeros(self._cell.n_where)
            self._where_prior = tfd.Normal(zeros, 1.)

        self._num_steps_prior = tfd.Geometric(probs=1 - prior_step_success_prob)

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
        ho['data_ll_per_pixel'] = tfd.Normal(canvas, self._output_std).log_prob(img)
        ho['data_ll'] = tf.reduce_sum(ho.data_ll_per_pixel, (-2, -1))

        # Post-processing
        num_steps_posterior = NumStepsDistribution(logits=ho.presence_logit[..., 0])
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
            sample = pdf.sample(arg)
            shape = [sample_size, self._n_steps] + sample.shape.as_list()[-1:]
            sample = tf.reshape(sample, shape)
            w.append(sample)
        what, where = w

        n = self._num_steps_prior.sample(sample_size)
        presence = tf.to_float(tf.sequence_mask(n, maxlen=self._n_steps))
        presence = tf.expand_dims(presence, -1)

        what, where, presence = ops.sort_by_distance_to_origin(what, where, presence)
        obs, _ = self._decoder(what, where, presence)
        return obs, what, where, presence


class Model(object):
    """Generic AIRModel model"""
    output_std = 1.
    internal_decode = False
    TARGETS = 'iwae w+s rw+s rw+rw rw+rws'.split()

    def __init__(self, obs, model, k_particles, target, target_arg=None, presence=None, debug=False):
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
        self.target_arg = target_arg

        self.gt_presence = presence
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

        self.importance_weights = tf.stop_gradient(tf.nn.softmax(self.log_weights, -1))
        self.ess = tf.reduce_mean(tf.pow(tf.reduce_sum(self.importance_weights, -1), 2) / tf.reduce_sum(tf.pow(self.importance_weights, 2), -1))
        tf.summary.scalar('ess/value', self.ess)

        self.elbo_vae = tf.reduce_mean(self.log_weights)
        self.elbo_iwae_per_example = tf.reduce_logsumexp(self.log_weights, -1) - tf.log(float(self.k_particles))
        self.elbo_iwae = tf.reduce_mean(self.elbo_iwae_per_example)

        self.num_steps_per_example = self.outputs.num_steps
        self.num_steps = tf.reduce_mean(self.num_steps_per_example)

        if self.gt_presence is not None:
            self.gt_num_steps = tf.reduce_sum(self.gt_presence, -1)
            # num_step_per_sample = self._resample(self.outputs.num_steps)
            num_step_per_sample = self.num_steps_per_example
            num_step_per_sample = tf.reshape(num_step_per_sample, (self.batch_size, self.k_particles))
            gt_num_steps = tf.expand_dims(self.gt_num_steps, -1)
            self.num_step_accuracy = tf.reduce_mean(tf.to_float(tf.equal(gt_num_steps, num_step_per_sample)))

    def make_target(self, opt):

        if self.target == 'iwae':
            target = targets.vimco(self.log_weights, self.outputs.steps_log_prob, self.elbo_iwae_per_example)
            gvs = opt.compute_gradients(target)

        elif self.target in self.TARGETS[1:]:
            decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope='attend_infer_repeat/air_decoder')
            encoder_vars = list(set(tf.trainable_variables()) - set(decoder_vars))

            importance_weights = self.importance_weights

            if 'annealed' in self.target_arg.lower():
                target_ess = float(self.target_arg.split('_')[-1]) * self.k_particles
                self.alpha = targets.pynverse_find_alpha(target_ess, self.log_weights)
                importance_weights = tf.nn.softmax(self.log_weights * self.alpha, -1)
                self.alpha_importance_weights = importance_weights
                self.alpha_ess = tf.reduce_mean(tf.pow(tf.reduce_sum(importance_weights, -1), 2) / tf.reduce_sum(tf.pow(importance_weights, 2), -1))
                tf.summary.scalar('ess/alpha_value', self.alpha_ess)
                tf.summary.scalar('ess/alpha', self.alpha)

            if self.target == 'rw+rw':
                decoder_target, encoder_target = self._rwrw_targets(importance_weights)

            elif self.target == 'rw+s':
                decoder_target, _ = self._rwrw_targets(importance_weights)

                sample = self.model.sample(self.batch_size)
                obs, what, where, presence = (tf.stop_gradient(i) for i in sample)
                sleep_outputs = self.model(obs, latent_override=[what, where, presence])

                encoder_target = -tf.reduce_mean(sleep_outputs.log_q_z)

            elif self.target == 'rw+rws':
                decoder_target, encoder_wake_target = self._rwrw_targets(importance_weights)

                sample = self.model.sample(self.batch_size)
                obs, what, where, presence = (tf.stop_gradient(i) for i in sample)
                sleep_outputs = self.model(obs, latent_override=[what, where, presence])

                encoder_sleep_target = -tf.reduce_mean(sleep_outputs.log_q_z)
                encoder_target = (encoder_sleep_target + encoder_wake_target) / 2.

            elif self.target == 'w+s':
                sample = self.model.sample(self.batch_size)
                obs, what, where, presence = (tf.stop_gradient(i) for i in sample)
                sleep_outputs = self.model(obs, latent_override=[what, where, presence])

                decoder_target = -tf.reduce_mean(self.outputs.log_p_x_and_z)
                encoder_target = -tf.reduce_mean(sleep_outputs.log_q_z)

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

    def _rwrw_targets(self, imp_weights):
        logpxz = tf.reshape(self.outputs.log_p_x_and_z, (self.batch_size, self.k_particles))
        logqz = tf.reshape(self.outputs.log_q_z, (self.batch_size, self.k_particles))
        return targets.reweighted_wake_wake(logpxz, logqz, imp_weights)
