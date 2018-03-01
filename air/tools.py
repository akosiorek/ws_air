import time
import json
import os
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python import debug as tf_debug

import cPickle as pickle


def log_values(writer, itr, tags=None, values=None, dict=None):
    if dict is not None:
        assert tags is None and values is None
        tags = dict.keys()
        values = dict.values()
    elif tags is None or values is None:
        raise ValueError('both tags and values have to be specified but are tags={}; values={}'.format(tags, values))
    else:
        if not nest.is_sequence(tags):
            tags, values = [tags], [values]

        elif len(tags) != len(values):
            raise ValueError('tag and value have different lenghts:'
                             ' {} vs {}'.format(len(tags), len(values)))

    for t, v in zip(tags, values):
        summary = tf.Summary.Value(tag=t, simple_value=v)
        summary = tf.Summary(value=[summary])
        writer.add_summary(summary, itr)


def save_flags(flags, path):
    if hasattr(flags, '_flags'):
        path = os.path.join(path, 'flags.tf')
        flags.append_flags_into_file(path)
    else:
        path = os.path.join(path, 'flags.json')
        with open(path, 'w') as f:
            json.dump(flags.__flags, f, sort_keys=True, indent=4)


def save_pickle(path, data):
    with open(path, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def get_session(tfdbg=False):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    if tfdbg:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    return sess


def make_logger(air, sess, writer, train_tensor, num_train_batches, test_tensor, num_test_batches, eval_on_train):
    exprs = {
        'elbo_vae': air.elbo_vae,
        'elbo_iwae': air.elbo_iwae,
        'num_steps': air.num_steps,
        'num_steps_acc': air.num_step_accuracy,
    }

    data_dict = {
        train_tensor['imgs']: test_tensor['imgs'],
        train_tensor['nums']: test_tensor['nums']
    }
    test_log = make_expr_logger(sess, writer, num_test_batches, exprs, name='test', data_dict=data_dict)

    if eval_on_train:
        train_log = make_expr_logger(sess, writer, num_train_batches, exprs, name='train')

        def log(train_itr):
            train_log(train_itr)
            test_log(train_itr)
            print
    else:
        def log(train_itr):
            test_log(train_itr)
            print

    return log


def make_expr_logger(sess, writer, num_batches, expr_dict, name, data_dict=None,
                     constants_dict=None, measure_time=True):
    """Creates a logging function. It evaluates tensors in `expr_dict` `num_batches` times,
    logs the result into standard output and saves them as tensorboard events.

    :param sess: tf.Session
    :param writer: tf.summary.FileWriter
    :param num_batches: int, number of iterations to run
    :param expr_dict: dict of tag: tensor; tags are used to log result of tensor evaluation into tensorboard.
    :param name: name of the dataset used.
    :param data_dict: dict of placeholder: tensor pairs; used to create a feed_dict for expression in expr_dict.
    :param constants_dict: dict of placeholder: tensor pairs; used to feed constants in a feed dict.
    :param measure_time: bool; reports time taken to log if True.
    :return: callable log(current_iter, ...), the logging funciion
    """

    tags = {k: '/'.join((k, name)) for k in expr_dict}
    data_name = 'Data {}'.format(name)
    log_string = ', '.join((''.join((k + ' = {', k, ':.4f}')) for k in expr_dict))
    log_string = ' '.join(('Step {},', data_name, log_string))

    if measure_time:
        log_string += ', eval time = {:.4}s'

        def log(itr, l, t):
            return log_string.format(itr, t, **l)
    else:
        def log(itr, l, t):
            return log_string.format(itr, **l)

    def logger(itr=0, num_batches_to_eval=None, write=True):
        l = {k: 0. for k in expr_dict}
        start = time.time()
        if num_batches_to_eval is None:
            num_batches_to_eval = num_batches

        for i in xrange(num_batches_to_eval):
            if data_dict is not None:
                vals = sess.run(data_dict.values())
                feed_dict = {k: v for k, v in zip(data_dict.keys(), vals)}
                if constants_dict:
                    feed_dict.update(constants_dict)
            else:
                feed_dict = constants_dict

            r = sess.run(expr_dict, feed_dict)
            for k, v in r.iteritems():
                l[k] += v

        for k, v in l.iteritems():
            l[k] /= num_batches_to_eval
        t = time.time() - start
        print log(itr, l, t)

        if write:
            log_values(writer, itr, [tags[k] for k in l.keys()], l.values())

        return l

    return logger

