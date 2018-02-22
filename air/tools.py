import json
import os
import tensorflow as tf
from tensorflow.python.util import nest

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


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess