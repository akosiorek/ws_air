import tensorflow as tf

import ops


class AIRModel(object):
    """Generic AIRModel model"""
    output_std = 1.
    internal_decode = False

    def __init__(self, obs, model, k_particles=1, debug=False):
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
        for k, v in self.outputs.iteritems():
            print k, v.shape.as_list()

        # self.gvs = self.gradients()

        log_weights = self.outputs.log_weights
        log_weights = tf.reshape(log_weights, (self.batch_size, self.k_particles))

        importance_weights = tf.nn.softmax(log_weights, -1)
        self.elbo_vae = tf.reduce_mean(log_weights)
        elbo_iwae_per_example = tf.reduce_logsumexp(log_weights, -1) - tf.log(float(self.k_particles))
        self.elbo_iwae = tf.reduce_mean(elbo_iwae_per_example)
        
        control_variate = ops.vimco_baseline(log_weights)
        learning_signal = tf.stop_gradient(log_weights - control_variate)
        steps_log_prob = tf.reshape(self.outputs.steps_log_prob, (self.batch_size, self.k_particles))
        reinforce_target = learning_signal * steps_log_prob

        proxy_loss = -tf.expand_dims(elbo_iwae_per_example, -1) - reinforce_target
        resample = False
        if resample:
            # resample proxy loss
            pass
            
        self.proxy_loss = tf.reduce_mean(proxy_loss)

    def make_target(self, target, opt):
        target = self.proxy_loss
        gvs = opt.compute_gradients(target)
        return target, gvs






    # def gradients(self, resample=False):
    #     # if resample:
    #     #     posterior_num_steps_log_prob = self.resample(posterior_num_steps_log_prob)
    #     #     posterior_num_steps_log_prob = tf.reshape(posterior_num_steps_log_prob,
    #     #                                               (self.n_timesteps, self.batch_size, 1))
    #     #
    #     #     baseline = tf.reshape(self.baseline, (-1, self.effective_batch_size,))
    #     #     elbo_per_sample, baseline = self.resample(self.cumulative_elbo_per_sample, baseline)
    #     #     self.nelbo_per_sample = -1 * tf.reshape(elbo_per_sample, (self.batch_size, 1))
    #     #     self.baseline = tf.reshape(baseline, (-1, self.batch_size, 1))
    #     #
    #     #     learning_signal = -self.resample(self.elbo_per_sample)
    #     #
    #     #     if self.per_timestep_vimco:
    #     #         learning_signal = tf.cumsum(learning_signal, reverse=True)
    #     #     else:
    #     #         learning_signal = tf.reduce_sum(learning_signal, 0, keep_dims=True)
    #     #         posterior_num_steps_log_prob = tf.reduce_sum(posterior_num_steps_log_prob, 0, keep_dims=True)
    #     #
    #     #     num_steps_learning_signal = tf.reshape(learning_signal, (-1, self.batch_size, 1))
    #     #
    #     #     # this could be constant e.g. 1, but the expectation of this is zero anyway,
    #     #     #  so there's no point in adding that.
    #     #     r_imp_weight = 0.
    #     # else:
    #     posterior_num_steps_log_prob = tf.reshape(posterior_num_steps_log_prob, (self.batch_size, self.k_particles))
    #     r_imp_weight = self.cumulative_imp_weights
    #     self.nelbo_per_sample = -tf.reshape(self.cumulative_iw_elbo_per_sample, (self.batch_size, 1))
    #     num_steps_learning_signal = self.nelbo_per_sample
    #
    #     self.nelbo = tf.reduce_mean(self.nelbo_per_sample) / self.n_timesteps
    #
    #     self.reinforce_loss = self._reinforce(num_steps_learning_signal - r_imp_weight, posterior_num_steps_log_prob)
    #     self.proxy_loss = self.nelbo + self.reinforce_loss / self.n_timesteps
    #
    #     opt = make_opt(self.learning_rate)
    #     gvs = opt.compute_gradients(self.proxy_loss, var_list=self.model_vars)
    #
    #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #     global_step = tf.train.get_or_create_global_step()
    #     with tf.control_dependencies(update_ops):
    #         train_step = opt.apply_gradients(gvs, global_step=global_step)
    #
    #     return train_step, gvs
    #
    # def _make_baseline(self, per_sample_elbo):
    #
    #     if self.k_particles == 1:
    #         return tf.zeros((self.n_timesteps, self.batch_size, self.k_particles), dtype=tf.float32)
    #
    #     if self.per_timestep_vimco:
    #         per_sample_elbo = tf.cumsum(per_sample_elbo, reverse=True)
    #         leading_dim = self.n_timesteps
    #     else:
    #         per_sample_elbo = tf.reduce_sum(per_sample_elbo, 0, keep_dims=True)
    #         leading_dim = 1
    #
    #     reshaped_per_sample_elbo = tf.reshape(per_sample_elbo, (leading_dim, self.batch_size, self.k_particles))
    #
    #     baseline = ops.vimco_baseline(reshaped_per_sample_elbo)
    #     return -baseline
    #
    # def _reinforce(self, learning_signal, posterior_num_steps_log_prob):
    #     """Implements REINFORCE for training the discrete probability distribution over number of steps and train-step
    #      for the baseline"""
    #
    #     self.num_steps_learning_signal = learning_signal
    #     if self.baseline is not None:
    #         self.num_steps_learning_signal -= self.baseline
    #
    #     axes = range(len(self.num_steps_learning_signal.get_shape()))
    #     imp_weight_mean, imp_weight_var = tf.nn.moments(self.num_steps_learning_signal, axes)
    #     tf.summary.scalar('imp_weight_mean', imp_weight_mean)
    #     tf.summary.scalar('imp_weight_var', imp_weight_var)
    #     reinforce_loss_per_sample = tf.stop_gradient(self.num_steps_learning_signal) * posterior_num_steps_log_prob
    #
    #     shape = reinforce_loss_per_sample.shape.as_list()
    #     excepted_nt = self.n_timesteps if self.per_timestep_vimco else 1
    #     assert len(shape) == 3 and shape[0] == excepted_nt and shape[1] == self.batch_size and shape[2] in (
    #         1, self.k_particles), 'shape is {}'.format(shape)
    #
    #     self.reinforce_loss_per_sample = tf.squeeze(reinforce_loss_per_sample, -1)
    #     reinforce_loss = tf.reduce_mean(self.reinforce_loss_per_sample)
    #     tf.summary.scalar('reinforce_loss', reinforce_loss)
    #     return reinforce_loss
    #
    # def resample(self, *args, **kwargs):
    #     axis = -1
    #     if 'axis' in kwargs:
    #         axis = kwargs['axis']
    #         del kwargs['axis']
    #
    #     res = list(args)
    #
    #     if self.k_particles > 1:
    #         for i, arg in enumerate(res):
    #             res[i] = self._resample(arg, axis)
    #
    #     if len(res) == 1:
    #         res = res[0]
    #
    #     return res
    #
    # def _resample(self, arg, axis=-1):
    #     iw_sample_idx = self.imp_resampling_idx + tf.range(self.batch_size) * self.k_particles
    #     shape = arg.shape.as_list()
    #     shape[axis] = self.batch_size
    #     resampled = ops.gather_axis(arg, iw_sample_idx, axis)
    #     resampled.set_shape(shape)
    #     return resampled
    #
    # def _log_resampled(self, resampled, name):
    #     resampled = self._resample(resampled)
    #     setattr(self, 'resampled_' + name, resampled)
    #     value = tf.reduce_mean(resampled)
    #     setattr(self, name, value)
    #     tf.summary.scalar(name, value)
