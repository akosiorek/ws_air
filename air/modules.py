import functools
import numpy as np

import tensorflow as tf
from tensorflow.contrib.distributions import NormalWithSoftplusScale
from tensorflow.python.util import nest
from tensorflow.contrib.resampler import resampler as tf_resampler

import sonnet as snt

from neural import MLP
from ops import clip_preserve


class ParametrisedGaussian(snt.AbstractModule):

    def __init__(self, n_params, loc_mult=1., scale_offset=0., *args, **kwargs):
        super(ParametrisedGaussian, self).__init__()
        self._n_params = n_params
        self._loc_mult = loc_mult
        self._scale_offset = scale_offset
        self._create_distrib = lambda x, y: NormalWithSoftplusScale(x, y, *args, **kwargs)

    def _build(self, inpt):
        transform = snt.Linear(2 * self._n_params)
        params = transform(inpt)
        loc, scale = tf.split(params, 2, params.shape.ndims - 1)
        distrib = self._create_distrib(loc * self._loc_mult, scale + self._scale_offset)
        return distrib


class StochasticTransformParam(snt.AbstractModule):

    def __init__(self, n_hidden, min_glimpse_size=0.0, max_glimpse_size=1.0, scale_bias=-2.):
        super(StochasticTransformParam, self).__init__()
        self._n_hidden = n_hidden
        assert 0 <= min_glimpse_size < max_glimpse_size <= 1.
        self._min_glimpse_size = min_glimpse_size
        self._max_glimpse_size = max_glimpse_size
        self._scale_bias = scale_bias

    def _build(self, inpt):

        flatten = snt.BatchFlatten()
        mlp = MLP(self._n_hidden, n_out=8)
        seq = snt.Sequential([flatten, mlp])
        params = seq(inpt)

        return params[..., :4], params[..., 4:] + self._scale_bias


class Encoder(snt.AbstractModule):

    def __init__(self, n_hidden):
        super(Encoder, self).__init__()
        self._n_hidden = n_hidden

    def _build(self, inpt):
        flat = snt.BatchFlatten()
        mlp = MLP(self._n_hidden)
        seq = snt.Sequential([flat, mlp])
        return seq(inpt)


class Decoder(snt.AbstractModule):

    def __init__(self, n_hidden, output_size, output_scale=1.):
        super(Decoder, self).__init__()
        self._n_hidden = n_hidden
        self._output_size = output_size
        self._output_scale = output_scale

    def _build(self, inpt):
        n = np.prod(self._output_size)
        mlp = MLP(self._n_hidden, n_out=n)
        reshape = snt.BatchReshape(self._output_size)
        seq = snt.Sequential([mlp, reshape])
        return seq(inpt) * self._output_scale


class SpatialTransformer(snt.AbstractModule):

    def __init__(self, img_size, crop_size, inverse=False):
        super(SpatialTransformer, self).__init__()

        with self._enter_variable_scope():
            constraints = snt.AffineWarpConstraints.no_shear_2d()
            self._warper = snt.AffineGridWarper(img_size, crop_size, constraints)
            if inverse:
                self._warper = self._warper.inverse()

    def _sample_image(self, img, transform_params):
        grid_coords = self._warper(transform_params)
        return tf_resampler(img, grid_coords)

    def _build(self, img, sentinel=None, coords=None, logits=None):
        """Assume that `transform_param` has the shape of (..., n_params) where n_params = n_scales + n_shifts
        and:
            scale = transform_params[..., :n_scales]
            shift = transform_params[..., n_scales:n_scales+n_shifts]
        """

        if sentinel is not None:
            raise ValueError('Either coords or logits must be given by kwargs!')

        if coords is not None and logits is not None:
            raise ValueError('Please give eithe coords or logits, not both!')

        if coords is None and logits is None:
            raise ValueError('Please give coords or logits!')

        if coords is None:
            coords = self.to_coords(logits)

        axis = coords.shape.ndims - 1
        sx, sy, tx, ty = tf.split(coords, 4, axis=axis)
        sx, sy = (clip_preserve(s, 1e-4, s) for s in (sx, sy))

        transform_params = tf.concat([sx, tx, sy, ty], -1)

        if len(img.get_shape()) == 3:
            img = img[..., tf.newaxis]

        if len(transform_params.get_shape()) == 2:
            return self._sample_image(img, transform_params)
        else:
            transform_params = tf.unstack(transform_params, axis=1)
            samples = [self._sample_image(img, tp) for tp in transform_params]
            return tf.stack(samples, axis=1)

    @staticmethod
    def to_coords(logits):

        axis = logits.shape.ndims - 1
        scale_logit, shift_logit = tf.split(logits, 2, axis)

        scale = tf.nn.sigmoid(scale_logit)
        shift = tf.nn.tanh(shift_logit)
        coords = tf.concat((scale, shift), -1)
        return coords


class StepsPredictor(snt.AbstractModule):

    def __init__(self, n_hidden, steps_bias=0.):
        super(StepsPredictor, self).__init__()
        self._n_hidden = n_hidden
        self._steps_bias = steps_bias

    def _build(self, inpt):
        mlp = MLP(self._n_hidden, n_out=1)
        logit = mlp(inpt) + self._steps_bias
        return logit
    

class AIRDecoder(snt.AbstractModule):

    def __init__(self, img_size, glimpse_size, glimpse_decoder, batch_dims=2):
        super(AIRDecoder, self).__init__()
        self._inverse_transformer = SpatialTransformer(img_size, glimpse_size, inverse=True)
        self._batch_dims = batch_dims

        with self._enter_variable_scope():
            self._glimpse_decoder = glimpse_decoder(glimpse_size)

    def _build(self, what, where, presence):
        batch = functools.partial(snt.BatchApply, n_dims=self._batch_dims)
        glimpse = batch(self._glimpse_decoder)(what)

        inversed = batch(self._inverse_transformer)(glimpse, logits=where)
        inversed *= presence[..., tf.newaxis, tf.newaxis]
        canvas = tf.reduce_sum(inversed, axis=-4)

        return tf.squeeze(canvas, -1), glimpse
