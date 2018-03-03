import tensorflow as tf
import sonnet as snt
from functools import partial

from modules import Encoder, Decoder, StochasticTransformParam, StepsPredictor, AIRDecoder
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
tf.flags.DEFINE_boolean('importance_resample', False, '')
tf.flags.DEFINE_boolean('binary', False, '')

tf.flags.DEFINE_boolean('rec_prior', False, '')
tf.flags.DEFINE_string('target_arg', '', '')

tf.flags.DEFINE_string('opt', '', '')
tf.flags.DEFINE_string('transition', 'LSTM', '')

tf.flags.DEFINE_float('output_std', .3, '')

flags.DEFINE_string('target', 'iwae', 'choose from: {}'.format(Model.TARGETS))


def load(img, num, mean_img=None):
    F = tf.flags.FLAGS

    target = F.target.lower()
    assert target in Model.TARGETS, 'Target is {} and not in {}'.format(F.target, Model.TARGETS)

    gradients_through_z = True
    if target != Model.TARGETS[0]:
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
                            binary=F.binary
                            )

    model = Model(img, air, F.k_particles, target=target, target_arg=F.target_arg, presence=num,
                  binary=F.binary)
    return model
