import os
import time

import tensorflow as tf
from tensorflow.python.util import nest

import sys
sys.path.append('../')

import tools
from model_config import load as load_model
from data_config import load as load_data

flags = tf.flags

flags.DEFINE_string('checkpoint_dir', '../checkpoints', '')
flags.DEFINE_string('run_name', '', '')

flags.DEFINE_integer('batch_size', 5, '')

flags.DEFINE_integer('n_samples', 5, 'this many importance weights will be used')

flags.DEFINE_integer('every_nth_checkpoint', 1, 'takes 1 in nth checkpoints to evaluate; takes only the last checkpoint if -1')

flags.DEFINE_string('dataset', 'test', 'test or train')

flags.DEFINE_boolean('eval_logp', True, '')
flags.DEFINE_string('gpu', '0', 'Id of the gpu to allocate')


F = flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = F.gpu

if __name__ == '__main__':


    run_names = F.run_name
    if len(run_names) == 0:
        run_names = [p for p in os.listdir(F.checkpoint_dir) if os.path.isdir(os.path.join(F.checkpoint_dir, p))]
    else:
        run_names = nest.flatten(run_names)

    for run_name in run_names:
        print 'Processing run:', run_name
        logdir = os.path.abspath(os.path.join(F.checkpoint_dir, run_name))

        print logdir
        checkpoint_state = tf.train.get_checkpoint_state(logdir)
        if checkpoint_state is None:
            print 'Skipping', F.run_name
            continue

        checkpoint_paths = checkpoint_state.all_model_checkpoint_paths
        last_checkpoint = checkpoint_paths[-1]

        if F.every_nth_checkpoint >= 0:
            checkpoint_paths = checkpoint_paths[::F.every_nth_checkpoint]
            if checkpoint_paths[-1] != last_checkpoint:
                checkpoint_paths.append(last_checkpoint)
        elif F.every_nth_checkpoint == -1:
            checkpoint_paths = [checkpoint_paths[-1]]
        else:
            raise ValueError('every_nth_checkpoint has an invalid value of {}'.format(F.every_nth_checkpoint))

        tf.reset_default_graph()

        data_dict = load_data(F.batch_size, shuffle=False)
        mean_img = data_dict.train_data.imgs.mean(0)

        if F.dataset == 'test':
            n_batches = data_dict.train_data.imgs.shape[0] // F.batch_size
            img, num = data_dict.train_img, data_dict.train_num
        else:
            n_batches = data_dict.test_data.imgs.shape[0] // F.batch_size
            img, num = data_dict.test_img, data_dict.test_num

        model = load_model(img=img, num=num, mean_img=mean_img)

        saver = tf.train.Saver()
        sess = tools.get_session()
        sess.run(tf.global_variables_initializer())

        log_p_x_file = os.path.join(logdir, 'logpx_{}.txt'.format(F.dataset))

        for checkpoint_path in checkpoint_paths:
            n_itr = int(checkpoint_path.split('-')[-1])

            print 'Processing checkpoint:', n_itr,
            saver.restore(sess, checkpoint_path)

            log_p_x_estimate = 0.
            log_ess = []

            start = time.time()
            for batch_num in xrange(n_batches):
                log_p_x_batch = sess.run(model.elbo_iwae)
                log_p_x_estimate += log_p_x_batch

            log_p_x_estimate /= n_batches
            with open(log_p_x_file, 'a') as f:
                f.write('{}: {}\n'.format(n_itr, log_p_x_estimate))

            duration = time.time() - start
            print 'took {}s'.format(duration)
