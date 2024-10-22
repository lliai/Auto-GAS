import random

import torch
from torch import Tensor

from .basePipeline import BasePipeLine


def above_zero_op(x: Tensor):
    """ Binarization operation. """
    return (x > 0).float()


def above_mean_op(x: Tensor):
    """ Binarization operation. """
    return (x > x.mean()).float()


def above_median_op(x: Tensor):
    """ Binarization operation. """
    return (x > x.median()).float()


def sign_op(x: Tensor):
    """ Binarization operation. """
    return torch.sign(x).float()


class BinaryPipeline(BasePipeLine):
    """ Binary pipeline.

    Here we provided one types of binary ways.

    1. above_zero
    2. above_mean
    3. above_median
    4. sign

    """

    def __init__(self, *args, **kwargs):
        super(BinaryPipeline, self).__init__(*args, **kwargs)
        self._modules_operations = {
            'above_zero': above_zero_op,
            'above_mean': above_mean_op,
            'above_median': above_median_op,
            'sign': sign_op,
        }
        self._modules_probability = [0.25, 0.25, 0.25, 0.25]

        # initailize the selected index
        self._selected_idx = random.choices(
            range(len(self._modules_operations)),
            weights=self._modules_probability,
            k=1)[0]

    def __call__(self, x, *args, **kwargs):
        if self._selected_idx is None:
            raise ValueError('The selected index of binaryPipeline is None.')

        selected_binary = self._modules_operations[list(
            self._modules_operations.keys())[self._selected_idx]]
        return selected_binary(x)

    def __repr__(self):
        return f'BinaryPipeline(ops={list(self._modules_operations.keys())[self._selected_idx]})'

    @property
    def identity(self):
        return self._selected_idx

    @identity.setter
    def identity(self, value):
        self._selected_idx = value
