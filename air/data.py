import os
import sys
import numpy as np
import itertools
import cPickle as pickle

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.examples.tutorials.mnist import input_data

from attrdict import AttrDict
from scipy.misc import imresize


_this_dir = os.path.dirname(__file__)
_data_dir = os.path.abspath(os.path.join(_this_dir, '../'), )
_data_dir = os.path.join(_data_dir, 'data')
_MNIST_PATH = os.path.join(_data_dir, 'MNIST_data')


def dim_coords(proj):
    proj = np.greater(proj, 0.)
    size = proj.sum()
    start = np.argmax(np.arange(len(proj)) * proj) - size + 1
    return start, size


def template_dimensions(template):
    y_proj = template.sum(1)
    x_proj = template.sum(0)
    y_start, y_size = dim_coords(y_proj)
    x_start, x_size = dim_coords(x_proj)
    return (y_start, x_start), (y_size, x_size)


def create_mnist(partition='train', canvas_size=(50, 50), obj_size=(28, 28), n_objects=(0, 2), n_samples=None,
                 dtype=np.uint8, expand_nums=True, with_overlap=False, include_templates=False,
                 include_coords=False):

    mnist = input_data.read_data_sets(_MNIST_PATH, one_hot=False)
    mnist_data = getattr(mnist, partition)

    n_templates = mnist_data.num_examples
    if n_samples is None:
        n_samples = n_templates

    n_objects = nest.flatten(n_objects)
    if len(n_objects) == 1:
        n_objects *= 2
    assert len(n_objects) == 2
    n_objects.sort()
    min_objects, max_objects = n_objects

    imgs = np.zeros((n_samples,) + tuple(canvas_size), dtype=dtype)
    labels = np.zeros((n_samples, max_objects), dtype=np.uint8)
    nums = np.random.randint(min_objects, max_objects + 1, size=n_samples, dtype=np.uint8)

    templates = np.reshape(mnist_data.images, (-1, 28, 28))
    resize = lambda x: imresize(x, obj_size)
    obj_size = np.asarray(obj_size, dtype=np.int32)

    def make_p(size):
        position_range = np.asarray(canvas_size) - size
        return np.round(np.random.rand(2) * position_range).astype(np.int32)

    occupancy = np.zeros(canvas_size, dtype=bool)

    used_templates = np.empty(n_samples, dtype=np.object)
    used_coords = np.empty(n_samples, dtype=np.object)

    i = 0
    n_tries = 5
    print 'Creating {} samples'.format(n_samples)
    while i < n_samples:
        if i % 500 == 0:
            print '\r{} / {}'.format(i, n_samples),
            sys.stdout.flush()

        tries = 0
        retry = False
        n = nums[i]
        used_templates[i] = []
        used_coords[i] = []
        if n > 0:
            indices = np.random.choice(n_templates, n, replace=False)

            occupancy[...] = False
            for j in xrange(n):
                idx = indices[j]
                labels[i, j] = mnist_data.labels[idx]
                template = resize(templates[idx])
                st, size = template_dimensions(template)
                template = template[st[0]:st[0]+size[0], st[1]:st[1]+size[1]]

                p = make_p(size)
                if not with_overlap:
                    while occupancy[p[0]:p[0]+size[0], p[1]:p[1]+size[1]].any() and tries < n_tries:
                        p = make_p(size)
                        tries += 1
                    if tries == n_tries:
                        retry = True
                        break

                if include_templates:
                    used_templates[i].append(template)
                if include_coords:
                    used_coords[i].append(p)

                imgs[i, p[0]:p[0]+size[0], p[1]:p[1]+size[1]] = template
                occupancy[p[0]:p[0]+size[0], p[1]:p[1]+size[1]] = True
        if not retry:
            i += 1
        else:
            imgs[i, ...] = 0.

    print '\nfinished'
    if expand_nums:
        expanded = np.zeros((max_objects + 1, n_samples, 1), dtype=np.uint8)
        for i, n in enumerate(nums):
            expanded[:n, i] = 1
        nums = expanded

    data = dict(imgs=imgs, labels=labels, nums=nums)

    if include_coords:
        data['coords'] = used_coords

    if include_templates:
        data['templates'] = used_templates

    return data


def load_data(path, data_path=_MNIST_PATH):
    path = os.path.join(data_path, path)

    with open(path) as f:
        data = pickle.load(f)

    data['imgs'] = data['imgs'].astype(np.float32) / 255.
    data['nums'] = data['nums'].astype(np.float32)
    return AttrDict(data)


def tensors_from_data(data_dict, batch_size, axes=None, shuffle=False):
    keys = data_dict.keys()
    if axes is None:
        axes = {k: 0 for k in keys}

    key = keys[0]
    ax = axes[key]
    n_entries = data_dict[key].shape[ax]

    if shuffle:
        def idx_fun():
            return np.random.choice(n_entries, batch_size)

    else:
        rolling_idx = itertools.cycle(xrange(0, n_entries - batch_size + 1, batch_size))

        def idx_fun():
            start = next(rolling_idx)
            end = start + batch_size
            return np.arange(start, end)

    def data_fun():
        idx = idx_fun()
        minibatch = []
        for k in keys:
            item = data_dict[k]
            minibatch_item = item.take(idx, axes[k])
            minibatch.append(minibatch_item)
        return minibatch

    minibatch = data_fun()
    types = [getattr(tf, str(m.dtype)) for m in minibatch]

    tensors = tf.py_func(data_fun, [], types)
    for t, m in zip(tensors, minibatch):
        t.set_shape(m.shape)

    tensors = {k: v for k, v in zip(keys, tensors)}
    return tensors


if __name__ == '__main__':
    partitions = ['train', 'validation']
    nums = [60000, 10000]

    canvas_size = (50, 50)
    obj_size = (28, 28)
    n_objects = (0, 2)
    include_templates = False
    include_coords = False

    for p, n in zip(partitions, nums):
        print 'Processing partition "{}"'.format(p)
        data = create_mnist(p, canvas_size, obj_size, n_objects, n_samples=n,
                            include_coords=include_coords, include_templates=include_templates)

        filename = 'seq_mnist_{}.pickle'.format(p)
        filename = os.path.join(_MNIST_PATH, filename)
    
        print 'saving to "{}"'.format(filename)
        with open(filename, 'w') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
