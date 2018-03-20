import tensorflow as tf
import sonnet as snt
from functools import partial

from modules import Encoder, Decoder, StochasticTransformParam, StepsPredictor
from cell import AIRCell
from model import AttendInferRepeat
from model import Model
from prior import geom_success_prob

flags = tf.flags

tf.flags.DEFINE_float('step_bias', 1., '')
tf.flags.DEFINE_float('transform_var_bias', -3., '')

tf.flags.DEFINE_float('output_multiplier', .25, '')
tf.flags.DEFINE_float('init_step_success_prob', 1. - 1e-7, '')
tf.flags.DEFINE_float('final_step_success_prob', 1e-5, '')
tf.flags.DEFINE_float('step_success_prob', -1., 'in [0, 1.]; it\'s annealed from `init` to `final` if not set.')

tf.flags.DEFINE_float('n_anneal_steps_loss', 1e3, '')
tf.flags.DEFINE_integer('k_particles', 5, '')

tf.flags.DEFINE_integer('n_steps_per_image', 3, '')
tf.flags.DEFINE_string('input_type', Model.INPUT_TYPES[0], 'Choose from: {}'.format(Model.INPUT_TYPES))

tf.flags.DEFINE_boolean('rec_prior', False, '')
tf.flags.DEFINE_string('target_arg', '', '')

tf.flags.DEFINE_float('output_std', .3, '')

tf.flags.DEFINE_string('ws_annealing', 'none', 'choose from: exp, linear, dist')
tf.flags.DEFINE_float('ws_annealing_arg', 3., '')

flags.DEFINE_string('target', 'iwae', 'choose from: {}'.format(Model.TARGETS))


def choose_present(presence, tensor):
    n_entries = tf.reduce_prod(presence.shape)
    mask = tf.reshape(presence, [n_entries])
    mask = tf.cast(mask, bool)
    flat = tf.reshape(tensor, (n_entries, -1))
    return tf.boolean_mask(flat, mask)


def add_debug_logs(model):
    o = model.outputs
    p = o.presence

    with tf.name_scope('debug'):
        for k, v in o.iteritems():
            if 'what' in k or 'where' in k or ('presence' in k and k != 'presence'):
                print 'histogram', k
                v = choose_present(p, v)
                tf.summary.histogram(k, v)
                tf.summary.scalar(k + '/mean', tf.reduce_mean(v))
                tf.summary.scalar(k + '/max', tf.reduce_max(v))
                tf.summary.scalar(k + '/min', tf.reduce_min(v))

        tf.summary.histogram('data_ll', o.data_ll)

        tf.summary.scalar('canvas/max', tf.reduce_max(o.canvas))
        tf.summary.scalar('canvas/min', tf.reduce_min(o.canvas))

        tf.summary.scalar('data_ll/max', tf.reduce_max(o.data_ll_per_pixel))
        tf.summary.scalar('data_ll/min', tf.reduce_min(o.data_ll_per_pixel))


def load(img, num, mean_img=None, debug=False):
    F = tf.flags.FLAGS

    target = F.target.lower()
    assert target in Model.TARGETS, 'Target is {} and not in {}'.format(F.target, Model.TARGETS)

    assert F.input_type in Model.INPUT_TYPES, 'Invalid input type: {}'.format(F.input_type)

    gradients_through_z = True
    if target in Model.WS_TARGETS:
        gradients_through_z = False

    glimpse_size = [20, 20]
    n_hidden = 32 * 8
    n_layers = 2
    n_hiddens = [n_hidden] * n_layers
    n_what = 50
    steps_pred_hidden = [128, 64]

    shape = img.shape.as_list()
    batch_size, img_size = shape[0], shape[1:]

    air_cell = AIRCell(img_size, glimpse_size, n_what,
                       rnn=snt.VanillaRNN(256),
                       input_encoder=partial(Encoder, n_hiddens),
                       glimpse_encoder=partial(Encoder, n_hiddens),
                       transform_estimator=partial(StochasticTransformParam, n_hiddens, scale_bias=F.transform_var_bias),
                       steps_predictor=partial(StepsPredictor, steps_pred_hidden, F.step_bias),
                       gradients_through_z=gradients_through_z
    )

    glimpse_decoder = partial(Decoder, n_hiddens, output_scale=F.output_multiplier)
    if F.step_success_prob != -1.:
        assert 0. <= F.step_success_prob <= 1.
        step_success_prob = F.step_success_prob
    else:
        step_success_prob = geom_success_prob(F.init_step_success_prob, F.final_step_success_prob)

    air = AttendInferRepeat(F.n_steps_per_image, F.output_std, step_success_prob,
                            air_cell, glimpse_decoder, mean_img=mean_img,
                            recurrent_prior=F.rec_prior,
                            output_type=F.input_type
                            )

    model = Model(img, air, F.k_particles, target=target, target_arg=F.target_arg, presence=num,
                  input_type=F.input_type, ws_annealing=F.ws_annealing, ws_annealing_arg=F.ws_annealing_arg)

    if debug:
        add_debug_logs(model)

    return model
