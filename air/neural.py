import tensorflow as tf
from tensorflow.python.util import nest
import sonnet as snt


class Nonlinear(snt.Linear):
    """Layer implementing an affine non-linear transformation"""

    def __init__(self, n_output, transfer=tf.nn.elu, initializers=None):

        super(Nonlinear, self).__init__(output_size=n_output, initializers=initializers)
        self._transfer = transfer

    def _build(self, inpt):
        output = super(Nonlinear, self)._build(inpt)
        if self._transfer is not None:
            output = self._transfer(output)
        return output


class MLP(snt.AbstractModule):
    """Implements a multi-layer perceptron"""

    def __init__(self, n_hiddens, hidden_transfer=tf.nn.elu, n_out=None, transfer=None,
                 initializers=None, output_initializers=None, name=None):
        """Initialises the MLP

        :param n_hiddens: int or an interable of ints, number of hidden units in layers
        :param hidden_transfer: callable or iterable; a transfer function for hidden layers or an interable thereof. If it's an iterable its length should be the same as length of `n_hiddens`
        :param n_out: int or None, number of output units
        :param transfer: callable or None, a transfer function for the output
        """

        super(MLP, self).__init__(name=name)
        self._n_hiddens = nest.flatten(n_hiddens)
        transfers = nest.flatten(hidden_transfer)
        if len(transfers) > 1:
            assert len(transfers) == len(self._n_hiddens)
        else:
            transfers *= len(self._n_hiddens)
        self._hidden_transfers = nest.flatten(transfers)
        self._n_out = n_out
        self._transfer = transfer
        self._initializers = initializers

        if output_initializers is None:
            output_initializers = initializers
        self._output_initializers = output_initializers

    @property
    def output_size(self):
        if self._n_out is not None:
            return self._n_out
        return self._n_hiddens[-1]

    def _build(self, inpt):
            layers = []
            for n_hidden, hidden_transfer in zip(self._n_hiddens, self._hidden_transfers):
                layers.append(Nonlinear(n_hidden, hidden_transfer, self._initializers))

            if self._n_out is not None:
                layers.append(Nonlinear(self._n_out, self._transfer, self._output_initializers))

            module = snt.Sequential(layers)
            return module(inpt)