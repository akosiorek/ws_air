import tensorflow as tf
from attrdict import AttrDict

from air.data import load_data as _load_data, tensors_from_data as _tensors

flags = tf.flags

flags.DEFINE_string('train_path', 'mnist_train.pickle', '')
flags.DEFINE_string('valid_path', 'mnist_validation.pickle', '')
flags.DEFINE_integer('seq_len', 0, '')

axes = {'imgs': 0, 'labels': 0, 'nums': 1}


def load(batch_size):

    f = flags.FLAGS

    valid_data = _load_data(f.valid_path)
    train_data = _load_data(f.train_path)

    train_tensors = _tensors(train_data, batch_size, axes, shuffle=True)
    valid_tensors = _tensors(valid_data, batch_size, axes, shuffle=False)

    train_tensors['nums'] = tf.transpose(train_tensors['nums'][..., 0])
    valid_tensors['nums'] = tf.transpose(valid_tensors['nums'][..., 0])
    
    if f.seq_len > 0:
        train_tensors['nums'] = train_tensors['nums'][tf.newaxis]
        valid_tensors['nums'] = valid_tensors['nums'][tf.newaxis]
        train_tensors['imgs'] = train_tensors['imgs'][tf.newaxis]
        valid_tensors['imgs'] = valid_tensors['imgs'][tf.newaxis]

    data_dict = AttrDict(
        train_img=train_tensors['imgs'],
        valid_img=valid_tensors['imgs'],
        train_num=train_tensors['nums'],
        valid_num=valid_tensors['nums'],
        train_tensors=train_tensors,
        valid_tensors=valid_tensors,
        train_data=train_data,
        valid_data=valid_data,
        axes=axes
    )

    return data_dict