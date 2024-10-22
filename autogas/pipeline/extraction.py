from torch import Tensor

from .basePipeline import BasePipeLine


def reshape_op(x: Tensor):
    """ Reshape the input tensor to [bs, -1] """
    return x.reshape(x.shape[0], -1)


class ExtractionLine(BasePipeLine):
    """ Feature extraction pipeline.

    Here we provided one types of feature extraction ways.

    1. reshape to [bs, -1]

    """

    def __init__(self, *args, **kwargs):
        self._modules_operations = {
            'reshape': reshape_op,
        }
        self._modules_probability = 0.5
        self._selected = True

    def __call__(self, x):
        if self._selected is None:
            raise ValueError(
                'The selected index of extractionPipeline is None.')

        if isinstance(x, Tensor) and self._selected:
            return self._modules_operations['reshape'](x)
        elif isinstance(x, list) and self._selected:
            for candidx in x:
                if isinstance(candidx, Tensor):
                    return self.__call__(candidx)
        else:
            raise TypeError(f'Unsupported type {type(x)}')

    def __repr__(self):
        return f'ExtractionLine(selected={self._selected})'

    @property
    def identity(self):
        return self._selected

    @identity.setter
    def identity(self, value):
        self._selected = value
