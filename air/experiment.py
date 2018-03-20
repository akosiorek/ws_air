import os
import sys
import numpy as np

import tensorflow as tf

from model_config import load as load_model
from data_config import load as load_data
from tools import save_flags, get_session, make_logger, gradient_summaries

flags = tf.flags


flags.DEFINE_string('logdir', '../checkpoints', 'Root folder for log files')
flags.DEFINE_string('run_name', 'run', 'Folder in which all run information is stored')

flags.DEFINE_integer('batch_size', 32, '')

flags.DEFINE_integer('train_itr', int(2e6), 'Number of training iterations')
flags.DEFINE_integer('log_itr', int(1e3), 'Number of iterations between logs')
flags.DEFINE_integer('report_loss_every', int(1e3), 'Number of iterations better reporting minibatch loss - hearbeat')
flags.DEFINE_integer('snapshot_itr', int(2.5e4), 'Number of iterations between model snapshots')
flags.DEFINE_integer('eval_itr', int(5e3), 'Number of iterations between log p(x) is estimated')

flags.DEFINE_float('learning_rate', 1e-5, 'Initial values of the learning rate')
flags.DEFINE_float('l2', 0.0, 'Weight for the l2 regularisation of parameters')
flags.DEFINE_boolean('schedule', False, 'Uses a learning rate schedule if True')

flags.DEFINE_boolean('test_run', True, 'Only a small run if True')
flags.DEFINE_boolean('restore', False, 'Tries to restore the latest checkpoint if True')
flags.DEFINE_boolean('eval_on_train', True, 'Evaluates on the train set if True')

flags.DEFINE_float('clip_gradient', 0.0, 'clips gradient by global norm if nonzero')
flags.DEFINE_string('gpu', '0', 'Id of the gpu to allocate')


flags.DEFINE_boolean('tfdbg', False, 'Attaches the tf debugger to the session (and has_inf_or_nan_filter)')
flags.DEFINE_boolean('debug', False, 'Adds a lot of summaries if True')

F = flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = F.gpu


if F.test_run:
    F.run_name = 'test'
    F.eval_on_train = False
    F.report_loss_every = 10
    F.log_itr = 10
    F.target = 'rw+s'
    F.step_success_prob = .75
    F.rec_prior = True
    F.k_particles = 5
    # F.target_arg = 'annealed_0.5'
    F.input_type = 'binary'
    # F.output_std = 1.
    F.clip_gradient = 1e-3
    # F.ws_annealing = 'dist'
    # F.ws_annealing_arg = 3.
    # F.schedule = True
    F.debug = True

# Load Data and model
data_dict = load_data(F.batch_size)

mean_img = data_dict.train_data.imgs.mean(0)
model = load_model(img=data_dict.train_img, num=data_dict.train_num, mean_img=mean_img, debug=F.debug)

num_params = sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
print 'Number of trainable parameters:', num_params

run_name = '{}_k={}_target={}'.format(F.run_name, F.k_particles, F.target)
if F.target_arg:
    run_name += '_{}'.format(F.target_arg)

checkpoint_dir = os.path.join(F.logdir, run_name)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

save_flags(F, checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')

# ELBO
tf.summary.scalar('elbo/iwae', model.elbo_iwae)
tf.summary.scalar('elbo/vae', model.elbo_vae)
tf.summary.scalar('steps/num', model.num_steps)
tf.summary.scalar('steps/accuracy', model.num_step_accuracy)

# Training setup
global_step = tf.train.get_or_create_global_step()
n_iters_per_epoch = data_dict['train_data'].imgs.shape[0] // F.batch_size

lr = F.learning_rate
if F.schedule:
    boundary_itr = [int(i * 1e5) for i in (2, 8)]
    lrs = [lr, lr / 3.33, lr / 10.]
    learning_rate = tf.train.piecewise_constant(global_step, boundary_itr, lrs)

opt = tf.train.RMSPropOptimizer(lr, momentum=.9)

# Optimisation target
target, gvs = model.make_target(opt, n_train_itr=F.train_itr, l2_reg=F.l2)
tf.summary.scalar('target', target)

gs = [gv[0] for gv in gvs]
gradient_global_norm = tf.global_norm(gs, 'gradient_global_norm')
tf.summary.scalar('gradient_norm', gradient_global_norm)

if F.debug:
    gradient_summaries(gvs, norm=False)


if F.clip_gradient != 0.0:
    threshold = F.clip_gradient * num_params
    clipped_gs, _ = tf.clip_by_global_norm(gs, threshold, gradient_global_norm)
    gvs = [(g, gv[1]) for g, gv in zip(clipped_gs, gvs)]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = opt.apply_gradients(gvs, global_step=global_step)

grad_abs_mean, grad_mean, grad_var = 0., 0., 0.
for g, v in gvs:
    mean, var = tf.nn.moments(g, range(len(g.shape)))
    grad_abs_mean += tf.reduce_mean(abs(g))
    grad_mean += mean
    grad_var += var

tf.summary.scalar('grad/abs_mean', grad_abs_mean / len(gvs))
tf.summary.scalar('grad/mean', grad_mean / len(gvs))
tf.summary.scalar('grad/var', grad_var / len(gvs))


saver = tf.train.Saver(max_to_keep=10000)

summary_writer = tf.summary.FileWriter(checkpoint_dir, tf.get_default_graph())
summary_op = tf.summary.merge_all()

sess = get_session(tfdbg=F.tfdbg)

sess.run(tf.global_variables_initializer())

if F.restore:
    checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
    checkpoint_paths = checkpoint_state.all_model_checkpoint_paths
    last_checkpoint = checkpoint_paths[-1]
    saver.restore(sess, last_checkpoint)


report = [model.elbo_iwae, model.num_steps, model.num_step_accuracy, gradient_global_norm]

if F.ws_annealing == 'dist':
    report += [model.anneal_alpha, model.dist]

if 'annealed' in F.target_arg:
    report += [model.alpha, model.ess, model.alpha_ess]

train_data, valid_data = [data_dict[k] for k in 'train_tensors valid_tensors'.split()]
num_train_batches, num_valid_batches = [data_dict[k].imgs.shape[0] // F.batch_size for k in 'train_data valid_data'.split()]
evaluate = make_logger(model, sess, summary_writer, train_data, num_train_batches, valid_data, num_valid_batches, F.eval_on_train)


train_itr = sess.run(global_step)
print sess.run(report)

if train_itr == 0:
    evaluate(train_itr)

tensors = [report, global_step, train_step]
tensors_with_summary = tensors + [summary_op]

while train_itr < F.train_itr:

    if train_itr % F.log_itr == 0 and summary_op is not None:
        l, train_itr, _, summary = sess.run(tensors_with_summary)
        summary_writer.add_summary(summary, train_itr)
    else:
        l, train_itr, _ = sess.run(tensors)

    if train_itr % F.report_loss_every == 0:
        print train_itr, l

    if np.isnan(l).any():
        print 'NaN in reports; breaking...'
        sys.exit(1)

    if train_itr > 0 and train_itr % F.snapshot_itr == 0:
        saver.save(sess, checkpoint_path, global_step=train_itr)

    if train_itr > 0 and train_itr % F.eval_itr == 0:
        evaluate(train_itr)

saver.save(sess, checkpoint_path, global_step=train_itr)
evaluate(train_itr)
