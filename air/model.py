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
                 mean_img=None, recurrent_prior=False, output_type='normal'):

        super(AttendInferRepeat, self).__init__()
        self._n_steps = n_steps
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
            self._decoder = AIRDecoder(self._cell._img_size, self._cell._glimpse_size, glimpse_decoder,
                                       batch_dims=2,
                                       mean_img=mean_img,
                                       output_std=output_std,
                                       output_type=output_type
                                       )

    def _build(self, img, reuse_samples=None):
        # Inference
        # ho is hidden outputs; short name due to frequent usage
        initial_state = self._cell.initial_state(img)
        ho, hidden_state = self._unroll_timestep(initial_state, reuse_samples)
        ho['hidden_state'] = hidden_state

        # Generation
        latents = [ho[i] for i in 'what where presence'.split()]
        pdf_x_given_z, glimpse = self._decoder(*latents)

        ho['canvas'] = pdf_x_given_z.mean()
        ho['glimpse'] = glimpse
        ho['data_ll_per_pixel'] = pdf_x_given_z.log_prob(img)
        ho['data_ll'] = tf.reduce_sum(ho.data_ll_per_pixel, (-2, -1))

        # Post-processing
        num_steps_posterior = NumStepsDistribution(logits=ho.presence_logit[..., 0])
        ho['num_steps'] = tf.reduce_sum(tf.squeeze(ho.presence, -1), -1)
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

    def _unroll_timestep(self, hidden_state, reuse_samples=None):

        if reuse_samples is None:
            inpt = [tf.zeros((1, 1))] * 3
            inpt = [[False, inpt]] * self._n_steps
        else:
            reuse_samples = [tf.unstack(l, axis=-2) for l in reuse_samples]
            reuse_samples = zip(*reuse_samples)
            inpt = [[True, l] for l in reuse_samples]

        hidden_outputs = []

        for i in inpt:
            ho, hidden_state = self._cell(i, hidden_state)
            hidden_outputs.append(ho)

        hidden_outputs = ops.stack_states(hidden_outputs)
        hidden_outputs = AIRCell.outputs_by_name(hidden_outputs)
        return hidden_outputs, hidden_state[-1]

    def sample(self, sample_size=1, k_particles=1, mean=False):

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

        latents = ops.sort_by_distance_to_origin(what, where, presence)
        if k_particles > 1:
            latents = [ops.tile_input_for_iwae(i, k_particles) for i in latents]

        pdf, _ = self._decoder(*latents)

        if mean:
            obs = pdf.mean()
        else:
            obs = pdf.sample()

        return [obs] + latents


class Model(object):
    """Generic AIRModel model"""
    output_std = 1.
    internal_decode = False
    VI_TARGETS = 'iwae reinforce'.split()
    WS_TARGETS = 'w+s rw+s rw+rw rw+rws'.split()
    TARGETS = VI_TARGETS + WS_TARGETS
    INPUT_TYPES = 'normal binary logit'.split()

    def __init__(self, obs, model, k_particles, target,
                 target_arg=None, presence=None, input_type='normal', ws_annealing=None, ws_annealing_arg=None, debug=False):
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
        self.input_type = input_type
        self.ws_annealing = ws_annealing
        self.ws_annealing_arg = ws_annealing_arg
        self.debug = debug

        shape = self.obs.get_shape().as_list()
        self.batch_size = shape[0]

        if self.input_type == 'binary':
            self.obs = tfd.Bernoulli(probs=self.obs).sample()

        elif self.input_type == 'logit':
            obs = tf.clip_by_value(obs, 1e-4, 1. - 1e-4)
            self.obs = tf.log(obs / (1. - obs))

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
        self.ess = ops.ess(self.importance_weights, average=True)
        tf.summary.scalar('ess/value', self.ess)

        self.elbo_vae = tf.reduce_mean(self.log_weights)
        self.elbo_iwae_per_example = tf.reduce_logsumexp(self.log_weights, -1) - tf.log(float(self.k_particles))
        self.elbo_iwae = tf.reduce_mean(self.elbo_iwae_per_example)

        self.num_steps_per_example = self.outputs.num_steps
        self.num_steps = self._imp_weighted_mean(self.num_steps_per_example)

        if self.gt_presence is not None:
            self.gt_num_steps = tf.reduce_sum(self.gt_presence, -1)
            # num_step_per_sample = self._resample(self.outputs.num_steps)
            num_step_per_sample = self.num_steps_per_example
            num_step_per_sample = tf.reshape(num_step_per_sample, (self.batch_size, self.k_particles))
            gt_num_steps = tf.expand_dims(self.gt_num_steps, -1)

            acc = tf.to_float(tf.equal(gt_num_steps, num_step_per_sample))
            self.num_step_accuracy = self._imp_weighted_mean(acc)

    def make_target(self, opt, n_train_itr=None, l2_reg=0.):

        if self.target in self.VI_TARGETS:
            if self.target == 'iwae':
                target = targets.vimco(self.log_weights, self.outputs.steps_log_prob, self.elbo_iwae_per_example)
            elif self.target == 'reinforce':
                target = targets.reinforce(self.log_weights, self.outputs.steps_log_prob, self.elbo_iwae_per_example)

            target += self._l2_reg(l2_reg)
            gvs = opt.compute_gradients(target)

        elif self.target in self.WS_TARGETS:

            decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope='attend_infer_repeat/air_decoder')
            encoder_vars = list(set(tf.trainable_variables()) - set(decoder_vars))

            importance_weights = self.importance_weights

            if 'annealed' in self.target_arg.lower():
                target_ess = float(self.target_arg.split('_')[-1]) * self.k_particles

                self.alpha = tf.Variable(1., trainable=False)

                alpha_ess = tf.exp(ops.log_ess(self.alpha * self.log_weights, average=True))

                def exact():
                    return targets.alpha_for_ess(target_ess, self.log_weights)

                def inc():
                    return (1. + self.alpha) / 2.

                less = tf.less(alpha_ess, target_ess)
                alpha_update = self.alpha.assign(tf.cond(less, exact, inc))
                # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, alpha_update)

                # self.alpha = targets.alpha_for_ess(target_ess, self.log_weights)

                with tf.control_dependencies([alpha_update]):
                    importance_weights = tf.nn.softmax(self.log_weights * self.alpha, -1)

                self.alpha_importance_weights = importance_weights
                self.alpha_ess = ops.ess(importance_weights, average=True)
                tf.summary.scalar('ess/alpha_value', self.alpha_ess)
                tf.summary.scalar('ess/alpha', self.alpha)

            if self.target == 'w+s':
                decoder_target = -tf.reduce_mean(self.outputs.log_p_x_and_z)
                encoder_target = self.sleep_encoder_target()

            elif self.target == 'rw+s':
                decoder_target = self.wake_decoder_target(importance_weights)
                encoder_target = self.sleep_encoder_target()

            elif self.target == 'rw+rw':
                decoder_target = self.wake_decoder_target(importance_weights)
                encoder_target = self.wake_encoder_target(importance_weights)

            elif self.target == 'rw+rws':
                decoder_target = self.wake_decoder_target(importance_weights)

                encoder_wake_target = self.wake_encoder_target(importance_weights)
                encoder_sleep_target, sleep_outputs = self.sleep_encoder_target(True)

                encoder_target = self.annealed_wake_update(encoder_sleep_target, encoder_wake_target, n_train_itr,
                                                           sleep_outputs)
                # encoder_target = (encoder_wake_target + encoder_sleep_target) / 2.

            l2_reg = self._l2_reg(l2_reg)
            decoder_target += l2_reg
            encoder_target += l2_reg

            target = decoder_target + encoder_target
            decoder_gvs = opt.compute_gradients(decoder_target, var_list=decoder_vars)
            encoder_gvs = opt.compute_gradients(encoder_target, var_list=encoder_vars)
            gvs = decoder_gvs + encoder_gvs

        else:
            raise ValueError('Invalid target: {}'.format(self.target))

        assert len(gvs) == len(tf.trainable_variables())
        for g, v in gvs:
            assert g is not None, 'Gradient for variable {} is None'.format(v)

        return target, gvs

    def sleep_encoder_target(self, return_sleep_outputs=False):
        sample = self.model.sample(self.batch_size, self.k_particles)
        obs, what, where, presence = (tf.stop_gradient(i) for i in sample)
        sleep_outputs = self.model(obs, reuse_samples=[what, where, presence])
        target = -tf.reduce_mean(sleep_outputs.log_q_z)
        if return_sleep_outputs:
            target = target, sleep_outputs

        return target

    def wake_encoder_target(self, imp_weights=None):
        if imp_weights is None:
            imp_weights = self.importance_weights

        imp_weights = tf.stop_gradient(imp_weights)
        logqz = tf.reshape(self.outputs.log_q_z, (self.batch_size, self.k_particles))

        shape1, shape2 = logqz.shape.as_list(), imp_weights.shape.as_list()
        assert shape1 == shape2, 'shapes are not equal: logqz = {} vs imp weight = {}'.format(shape1, shape2)
        return -tf.reduce_mean(imp_weights * logqz * self.k_particles)

    def wake_decoder_target(self, imp_weights=None):
        if imp_weights is None:
            imp_weights = self.importance_weights

        imp_weights = tf.stop_gradient(imp_weights)
        logpxz = tf.reshape(self.outputs.log_p_x_and_z, (self.batch_size, self.k_particles))

        shape1, shape2 = logpxz.shape.as_list(), imp_weights.shape.as_list()
        assert shape1 == shape2, 'shapes are not equal: logpxz = {} vs imp weight = {}'.format(shape1, shape2)
        return -tf.reduce_mean(imp_weights * logpxz * self.k_particles)

    def annealed_wake_update(self, encoder_sleep_target, encoder_wake_target, n_train_itr=None, sleep_outputs=None):
        if isinstance(self.ws_annealing, basestring):
            self.ws_annealing = self.ws_annealing.lower()
        elif self.ws_annealing is None:
            self.ws_annealing = 'none'

        if self.ws_annealing not in 'none linear exp dist'.split():
            raise ValueError('Unknown value of ws_annealing = {}'.format(self.ws_annealing))

        if self.ws_annealing == 'none':
            alpha = 0.5
        else:
            assert n_train_itr is not None, 'n_train_itr cannot be None for wake-sleep annealing!'

            global_step = tf.train.get_or_create_global_step()
            progress = tf.to_float(global_step) / n_train_itr
            if self.ws_annealing == 'linear':
                alpha = 1. - 0.5 * progress

            elif self.ws_annealing == 'exp':
                c = self.ws_annealing_arg
                alpha = 1. - 0.5 * tf.exp(c * (progress - 1.))
            elif self.ws_annealing == 'dist':
                assert sleep_outputs is not None
                wake_logqz = self._imp_weighted_mean(self.outputs.log_q_z)
                sleep_logqz = tf.reduce_mean(sleep_outputs.log_q_z)
                distance = abs(wake_logqz - sleep_logqz)
                alpha = 1. - tf.exp(-distance)
                alpha = tf.stop_gradient(alpha)

        self.dist = distance
        self.anneal_alpha = alpha
        return alpha * encoder_wake_target + (1. - alpha) * encoder_sleep_target

    def _l2_reg(self, weight):
        if weight == 0.:
            return 0.

        return weight * sum(map(tf.nn.l2_loss, tf.trainable_variables()))

    def _imp_weighted_mean(self, tensor):
        tensor = tf.reshape(tensor, (self.batch_size, self.k_particles))
        return tf.reduce_mean(self.importance_weights * tensor * self.k_particles)
