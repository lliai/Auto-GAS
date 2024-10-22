import random

import torch
from torch import Tensor

from .basePipeline import BasePipeLine


def gram_matrix_op(x: Tensor):
    assert len(
        x.shape
    ) == 2, f'Gram matrix operation only support 2D tensor, but got {len(x)}D tensor.'
    return x @ x.T


def corref_op(x: Tensor):
    assert len(
        x.shape
    ) == 2, f'Correlation operation only support 2D tensor, but got {len(x)}D tensor.'
    return torch.corrcoef(x)


class CorrelationPipeline(BasePipeLine):
    """ Correlation pipeline.

    Here we provided one types of correlation ways.

    1. correlation implementation(manually)
    2. torch.corref

    """

    def __init__(self, *args, **kwargs):
        super(CorrelationPipeline, self).__init__(*args, **kwargs)
        self._modules_operations = {
            'gram': gram_matrix_op,
            'corref': corref_op,
        }
        self._modules_probability = [0.5, 0.5]

        # initalize the selected idx
        self._selected_idx = random.choices(
            range(len(self._modules_operations)),
            weights=self._modules_probability,
            k=1)[0]

    def __call__(self, x, *args, **kwargs):
        if self._selected_idx is None:
            raise ValueError(
                'The selected index of correlationPipeline is None.')

        selected_correlation = self._modules_operations[list(
            self._modules_operations.keys())[self._selected_idx]]
        return selected_correlation(x)

    def __repr__(self):
        return f'CorrelationPipeline(ops={list(self._modules_operations.keys())[self._selected_idx]})'

    @property
    def identity(self):
        return self._selected_idx

    @identity.setter
    def identity(self, value):
        self._selected_idx = value
