import os
import sys
import numpy as np

import tensorflow as tf

from model_config import load as load_model
from data_config import load as load_data
from tools import save_flags, get_session, make_logger

flags = tf.flags


flags.DEFINE_string('logdir', '../checkpoints', 'Root folder for log files')
flags.DEFINE_string('run_name', 'run', 'Folder in which all run information is stored')

flags.DEFINE_integer('batch_size', 32, '')

flags.DEFINE_integer('train_itr', int(1e6), 'Number of training iterations')
flags.DEFINE_integer('log_itr', int(1e3), 'Number of iterations between logs')
flags.DEFINE_integer('report_loss_every', int(1e3), 'Number of iterations better reporting minibatch loss - hearbeat')
flags.DEFINE_integer('snapshot_itr', int(2.5e4), 'Number of iterations between model snapshots')
flags.DEFINE_integer('eval_itr', int(5e3), 'Number of iterations between log p(x) is estimated')

flags.DEFINE_float('learning_rate', 1e-5, 'Initial values of the learning rate')

flags.DEFINE_boolean('test_run', True, 'Only a small run if True')
flags.DEFINE_boolean('restore', False, 'Tries to restore the latest checkpoint if True')
flags.DEFINE_boolean('tfdbg', False, 'Attaches the tf debugger to the session (and has_inf_or_nan_filter)')
flags.DEFINE_boolean('eval_on_train', True, 'Evaluates on the train set if True')

flags.DEFINE_string('gpu', '0', 'Id of the gpu to allocate')

F = flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = F.gpu


if F.test_run:
    F.run_name = 'test'
    F.eval_on_train = False
    F.report_loss_every = 10
    F.log_itr = 10
    F.target = 'rwrw+sleep'
    F.init_step_success_prob = .75
    F.final_step_success_prob = .75
    F.rec_prior = True
    F.n_iw_samples = 5
    # F.target_arg = 'annealed_0.90'

run_name = '{}_k={}_target={}'.format(F.run_name, F.n_iw_samples, F.target)
if F.target_arg:
    run_name += '_{}'.format(F.target_arg)

checkpoint_dir = os.path.join(F.logdir, run_name)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

save_flags(F, checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')


# Load Data and model
data_dict = load_data(F.batch_size)

mean_img = data_dict.train_data.imgs.mean(0)
model = load_model(img=data_dict.train_img, num=data_dict.train_num, mean_img=mean_img)


# ELBO
tf.summary.scalar('elbo/iwae', model.elbo_iwae)
tf.summary.scalar('elbo/vae', model.elbo_vae)
tf.summary.scalar('steps/num', model.num_steps)
tf.summary.scalar('steps/accuracy', model.num_step_accuracy)

# Training setup
global_step = tf.train.get_or_create_global_step()
opt = tf.train.RMSPropOptimizer(F.learning_rate, momentum=.9)

# Optimisation target
target, gvs = model.make_target(opt)
train_step = opt.apply_gradients(gvs, global_step=global_step)
tf.summary.scalar('target', target)

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


report = [model.elbo_iwae, model.num_steps, model.num_step_accuracy]

train_data, valid_data = [data_dict[k] for k in 'train_tensors valid_tensors'.split()]
num_train_batches, num_valid_batches = [data_dict[k].imgs.shape[0] // F.batch_size for k in 'train_data valid_data'.split()]
evaluate = make_logger(model, sess, summary_writer, train_data, num_train_batches, valid_data, num_valid_batches, F.eval_on_train)


train_itr = sess.run(global_step)
print sess.run(report)

if train_itr == 0:
    evaluate(train_itr)

while train_itr < F.train_itr:
    l, train_itr, _ = sess.run([report, global_step, train_step])
    if train_itr % F.report_loss_every == 0:
        print train_itr, l

    if np.isnan(l).any():
        print 'NaN in reports; breaking...'
        sys.exit(1)

    if train_itr % F.log_itr == 0 and summary_op is not None:
        summary = sess.run(summary_op)
        summary_writer.add_summary(summary, train_itr)

    if train_itr > 0 and train_itr % F.snapshot_itr == 0:
        saver.save(sess, checkpoint_path, global_step=train_itr)

    if train_itr > 0 and train_itr % F.eval_itr == 0:
        evaluate(train_itr)

saver.save(sess, checkpoint_path, global_step=train_itr)
evaluate(train_itr)
