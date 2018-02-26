import os
import numpy as np

import tensorflow as tf

from model_config import load as load_model
from data_config import load as load_data
from tools import save_flags, get_session

flags = tf.flags


flags.DEFINE_string('logdir', '../checkpoints', 'Root folder for log files')
flags.DEFINE_string('run_name', 'test', 'Folder in which all run information is stored')

flags.DEFINE_integer('batch_size', 32, '')

flags.DEFINE_integer('train_itr', int(3e5), 'Number of training iterations')
flags.DEFINE_integer('log_itr', int(1e3), 'Number of iterations between logs')
flags.DEFINE_integer('report_loss_every', int(1e3), 'Number of iterations better reporting minibatch loss - hearbeat')
flags.DEFINE_integer('snapshot_itr', int(1e5), 'Number of iterations between model snapshots')
flags.DEFINE_integer('eval_itr', int(1e5), 'Number of iterations between log p(x) is estimated')

flags.DEFINE_float('learning_rate', 1e-5, 'Initial values of the learning rate')

flags.DEFINE_boolean('test_run', True, 'Only a small run if True')

F = flags.FLAGS

if F.test_run:
    F.report_loss_every = 1
    F.eval_itr = 1e2
    F.target = 'rws+sleep'
    F.init_step_success_prob = .5

run_name = '{}_target={}'.format(F.run_name, F.target)
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

# Training setup
global_step = tf.train.get_or_create_global_step()
opt = tf.train.RMSPropOptimizer(F.learning_rate, momentum=.9)

# Optimisation target
target, gvs = model.make_target(opt)
train_step = opt.apply_gradients(gvs, global_step=global_step)
tf.summary.scalar('target', target)

saver = tf.train.Saver(max_to_keep=10000)
summary_writer = tf.summary.FileWriter(checkpoint_dir, tf.get_default_graph())
summary_op = tf.summary.merge_all()

sess = get_session()
sess.run(tf.global_variables_initializer())

itr = sess.run(global_step)


def estimate(n_itr, dataset, dataset_name):
    """

    :param n_itr: int, number of training iteration
    :param dataset: one MNIST datasets
    :param dataset_name: string, name to use for logging
    :return: float, estimate of the mean_log_weights
    """
    tensors = [target, model.vae, model.miwae, model.iwae, model.data_ll, model.kl]
    names = 'target vae miwae iwae data_ll kl'.split()
    return model.estimate(sess, dataset, tensors, names, dataset_name, n_itr, summary_writer)


def evaluate(n_itr):
    # print 'test target = {:.02f}, vae = {:.02f}, miwae = {:.02f}, iwae = {:.02f}, rec = {:.02f}'\
        # .format(*estimate(n_itr, test_data, 'test'))
    # print 'train target = {:.02f}, vae = {:.02f}, miwae = {:.02f}, iwae = {:.02f}, rec = {:.02f}'\
    #     .format(*estimate(n_itr, train_data, 'train'))
    pass

num_steps = tf.reduce_mean(model.outputs.num_steps)
report = [model.elbo_iwae, num_steps]

train_itr = sess.run(global_step)
print sess.run(report)
if train_itr == 0:
    evaluate(train_itr)

while train_itr < F.train_itr:
    l, train_itr, _ = sess.run([report, global_step, train_step])
    if train_itr % F.report_loss_every == 0:
        print train_itr, l

    if train_itr % F.log_itr == 0 and summary_op is not None:
        summary = sess.run(summary_op)
        summary_writer.add_summary(summary, itr)

    if itr > 0 and itr % F.snapshot_itr == 0:
        saver.save(sess, checkpoint_path, global_step=train_itr)

    if itr > 0 and itr % F.eval_itr == 0:
        evaluate(itr)

saver.save(sess, checkpoint_path, global_step=train_itr)
evaluate(train_itr)