import tensorflow as tf
import sonnet as snt
from functools import partial

from modules import Encoder, Decoder, StochasticTransformParam, StepsPredictor, AIRDecoder
from cell import AIRCell
from model import AttendInferRepeat
from static_model import AIRModel
from prior import geom_success_prob

flags = tf.flags

tf.flags.DEFINE_float('step_bias', 1., '')
tf.flags.DEFINE_float('transform_var_bias', -3., '')

tf.flags.DEFINE_float('output_multiplier', .25, '')
tf.flags.DEFINE_float('init_step_success_prob', 1. - 1e-7, '')
tf.flags.DEFINE_float('final_step_success_prob', 1e-5, '')

tf.flags.DEFINE_float('n_anneal_steps_loss', 1e3, '')
tf.flags.DEFINE_integer('n_iw_samples', 5, '')

tf.flags.DEFINE_integer('n_steps_per_image', 3, '')
tf.flags.DEFINE_boolean('importance_resample', False, '')

tf.flags.DEFINE_string('opt', '', '')
tf.flags.DEFINE_string('transition', 'LSTM', '')

tf.flags.DEFINE_float('output_std', .3, '')
tf.flags.DEFINE_boolean('gradients_through_z', True, '')


def load(img, num):
    F = tf.flags.FLAGS

    glimpse_size = (20, 20)

    n_hidden = 32 * 8
    n_layers = 2
    n_hiddens = [n_hidden] * n_layers
    n_what = 50
    steps_pred_hidden = [128, 64]

    shape = img.shape.as_list()
    batch_size, img_size = shape[0], shape[1:]

    air_cell = AIRCell(img_size, glimpse_size, n_what,
                       rnn=snt.VanillaRNN(256),
                       input_encoder=partial(Encoder, n_hidden),
                       glimpse_encoder=partial(Encoder, n_hidden),
                       transform_estimator=partial(StochasticTransformParam, n_hidden, scale_bias=F.transform_var_bias),
                       steps_predictor=partial(StepsPredictor, steps_pred_hidden, F.step_bias),
                       gradients_through_z=F.gradients_through_z
    )

    glimpse_decoder = partial(Decoder, n_hiddens, output_scale=F.output_multiplier)
    step_success_prob = geom_success_prob(F.init_step_success_prob, F.final_step_success_prob)

    air = AttendInferRepeat(F.n_steps_per_image, batch_size, F.output_std, step_success_prob,
                                air_cell, glimpse_decoder)

    model = AIRModel(img, air, F.n_iw_samples)
    return model
