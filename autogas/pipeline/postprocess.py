import random
from typing import TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gaswot.pipeline.handling import ExceptionHandling
from .basePipeline import BasePipeLine

handler = ExceptionHandling()

Scalar = TypeVar('Scalar')
Vector = TypeVar('Vector')
Matrix = TypeVar('Matrix')

ALLTYPE = Union[Union[Scalar, Vector], Matrix]

import torch


def no_op(x: ALLTYPE) -> ALLTYPE:
    """ No operation. """
    return x


def element_wise_log_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        x[x <= 0] = 1
        return torch.log(x)
    elif isinstance(x, np.ndarray):
        x[x <= 0] = 1
        return np.log(x)
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


def element_wise_abslog_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        x[x == 0] = 1
        x = torch.abs(x)
        return torch.log(x)
    elif isinstance(x, np.ndarray):
        x[x == 0] = 1
        x = np.abs(x)
        return np.log(x)
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


def element_wise_abs_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        return torch.abs(x)
    elif isinstance(x, np.ndarray):
        return np.abs(x)
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


def element_wise_pow_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        return torch.pow(x, 2)
    elif isinstance(x, np.ndarray):
        return np.power(x, 2)
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


def element_wise_exp_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        return torch.exp(x)
    elif isinstance(x, np.ndarray):
        return np.exp(x)
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


def normalize_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        m = torch.mean(x)
        s = torch.std(x)
        C = (x - m) / s
        C[C != C] = 0
        return C
    elif isinstance(x, np.ndarray):
        m = np.mean(x)
        s = np.std(x)
        C = (x - m) / (s + 1e-6)
        C[np.isnan(C)] = 0
        return C
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


def frobenius_norm_op(x: Matrix) -> Scalar:
    if isinstance(x, torch.Tensor):
        return torch.norm(x, p='fro')
    elif isinstance(x, np.ndarray):
        return np.linalg.norm(x, ord='fro')
    else:
        raise ValueError(f'x should be a matrix, but got {type(x)}')


def element_wise_normalized_sum_op(x: ALLTYPE) -> Scalar:
    if isinstance(x, torch.Tensor):
        return torch.sum(x) / x.numel()
    elif isinstance(x, np.ndarray):
        return np.sum(x) / x.size
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


def l1_norm_op(x: ALLTYPE) -> Scalar:
    if isinstance(x, torch.Tensor):
        return torch.sum(torch.abs(x)) / x.numel()
    elif isinstance(x, np.ndarray):
        return np.sum(np.abs(x)) / x.size
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


def softmax_op(x: Vector) -> Vector:
    if isinstance(x, torch.Tensor):
        return F.softmax(x, dim=0)
    elif isinstance(x, np.ndarray):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    else:
        raise ValueError(f'x should be a vector, but got {type(x)}')


def logsoftmax_op(x: Vector) -> Vector:
    if isinstance(x, torch.Tensor):
        return F.log_softmax(x, dim=0)
    elif isinstance(x, np.ndarray):
        return np.log(softmax_op(x))
    else:
        raise ValueError(f'x should be a vector, but got {type(x)}')


def slogdet_op(x: Matrix) -> Scalar:
    if isinstance(x, torch.Tensor):
        sign, value = torch.linalg.slogdet(x)
        return value
    elif isinstance(x, np.ndarray):
        sign, value = np.linalg.slogdet(x)
        return value
    else:
        raise ValueError(f'x should be a matrix, but got {type(x)}')


def eig_op(x: Matrix) -> Vector:
    if isinstance(x, torch.Tensor):
        assert len(x.shape) == 2 and x.shape[0] == x.shape[1]
        return torch.linalg.eig(x)[0]
    elif isinstance(x, np.ndarray):
        assert len(
            x.shape) == 2 and x.shape[0] == x.shape[1], f'x.shape: {x.shape}'
        return np.linalg.eigvals(x)[0]
    else:
        raise ValueError(f'x should be a matrix, but got {type(x)}')


def hamming_op(x: ALLTYPE) -> Scalar:
    if isinstance(x, (list, tuple)):
        K_list = []
        for x_item in x:
            if isinstance(x_item, torch.Tensor):
                K_list.append(x_item @ x_item.t())
                K_list.append((1. - x_item) @ (1. - x_item.t()))
            elif isinstance(x_item, np.ndarray):
                K_list.append(x_item @ x_item.T)
                K_list.append((1. - x_item) @ (1. - x_item.T))
        return sum(K_list)
    else:
        if isinstance(x, torch.Tensor):
            return x @ x.t() + (1. - x) @ (1. - x.t())
        elif isinstance(x, np.ndarray):
            return x @ x.T + (1. - x) @ (1. - x.T)
        else:
            raise ValueError(
                f'x should be a tensor or ndarray, but got {type(x)}')


def element_wise_sigmoid_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    elif isinstance(x, np.ndarray):
        return 1 / (1 + np.exp(-x))
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


def element_wise_tanh_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        return torch.tanh(x)
    elif isinstance(x, np.ndarray):
        return np.tanh(x)
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


# def inverse_op(x: Matrix) -> Matrix:
#     if isinstance(x, torch.Tensor):
#         return torch.inverse(x)
#     elif isinstance(x, np.ndarray):
#         return np.linalg.inv(x)
#     else:
#         raise ValueError(f'x should be a matrix, but got {type(x)}')

# def max_op(x: ALLTYPE) -> Scalar:
#     if isinstance(x, torch.Tensor):
#         return torch.max(x)
#     elif isinstance(x, np.ndarray):
#         return np.max(x)
#     else:
#         raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')

# def min_op(x: ALLTYPE) -> Scalar:
#     if isinstance(x, torch.Tensor):
#         return torch.min(x)
#     elif isinstance(x, np.ndarray):
#         return np.min(x)
#     else:
#         raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


def trace_op(x: Matrix) -> Scalar:
    if isinstance(x, torch.Tensor):
        return torch.trace(x)
    elif isinstance(x, np.ndarray):
        return np.trace(x)
    else:
        raise ValueError(f'x should be a matrix, but got {type(x)}')


# def determinant_op(x: Matrix) -> Scalar:
#     if isinstance(x, torch.Tensor):
#         return torch.det(x)
#     elif isinstance(x, np.ndarray):
#         return np.linalg.det(x)
#     else:
#         raise ValueError(f'x should be a matrix, but got {type(x)}')

# def diagonal_op(x: Matrix) -> Vector:
#     if isinstance(x, torch.Tensor):
#         return torch.diagonal(x)
#     elif isinstance(x, np.ndarray):
#         return np.diagonal(x)
#     else:
#         raise ValueError(f'x should be a matrix, but got {type(x)}')


def l2_norm_op(x: ALLTYPE) -> Scalar:
    if isinstance(x, torch.Tensor):
        return torch.norm(x, p=2)
    elif isinstance(x, np.ndarray):
        return np.linalg.norm(x, ord=2)
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


# def rank_op(x: Matrix) -> Scalar:
#     if isinstance(x, torch.Tensor):
#         return torch.matrix_rank(x)
#     elif isinstance(x, np.ndarray):
#         return np.linalg.matrix_rank(x)
#     else:
#         raise ValueError(f'x should be a matrix, but got {type(x)}')


def transpose_op(x: Matrix) -> Matrix:
    if isinstance(x, torch.Tensor):
        return torch.t(x)
    elif isinstance(x, np.ndarray):
        return np.transpose(x)
    else:
        raise ValueError(f'x should be a matrix, but got {type(x)}')


# def geometric_mean_op(x: ALLTYPE) -> Scalar:
#     if isinstance(x, torch.Tensor):
#         return torch.prod(x).pow(1.0 / x.numel())
#     elif isinstance(x, np.ndarray):
#         return np.prod(x)**(1.0 / x.size)
#     else:
#         raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


def element_wise_sqrt_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        # Ensuring non-negative values for sqrt
        x = torch.clamp(x, min=0)
        return torch.sqrt(x)
    elif isinstance(x, np.ndarray):
        # Ensuring non-negative values for sqrt
        x = np.clip(x, 0, None)
        return np.sqrt(x)
    else:
        raise ValueError(f'x should be a tensor or ndarray, but got {type(x)}')


class PostprocessPipeline(BasePipeLine):
    """ Postprocess pipeline.

    Here we provided one types of postprocess ways.

    1. slogdet 2. eig 3. abs
    4. sum 5. min 6. mean 7. log 8. no_op

    """

    def __init__(self, length=3, *args, **kwargs):
        super(PostprocessPipeline, self).__init__(*args, **kwargs)
        self._modules_operations = {
            'slogdet': slogdet_op,
            'eig': eig_op,
            'abs': element_wise_abs_op,
            'sum': torch.sum,
            'min': torch.min,
            'mean': torch.mean,
            'log': element_wise_log_op,
            'no_op': no_op,
            'softmax': softmax_op,
            'logsoftmax': logsoftmax_op,
            'frobenius_norm': frobenius_norm_op,
            'l1_norm': l1_norm_op,
            'element_wise_abslog': element_wise_abslog_op,
            'element_wise_pow': element_wise_pow_op,
            'element_wise_exp': element_wise_exp_op,
            'element_wise_normalized_sum': element_wise_normalized_sum_op,
            'normalize': normalize_op,
            'hamming': hamming_op,
        }
        self._modules_operations_probability = [
            1 / len(self._modules_operations)
            for _ in range(len(self._modules_operations))
        ]

        # initialize selected index
        self._selected_idx = [
            random.choices(
                range(len(self._modules_operations)),
                weights=self._modules_operations_probability,
                k=1)[0] for _ in range(length)
        ]
        self._length = length

    def __call__(self, x, *args, **kwargs):
        if self._selected_idx is None or (isinstance(self._selected_idx, list)
                                          and len(self._selected_idx) == 0):
            raise ValueError('The selected index is None or empty.')

        try:
            for idx in self._selected_idx:
                x = self._modules_operations[list(
                    self._modules_operations.keys())[idx]](
                        x)
        except Exception as e:
            return x

        # if x is still a vector or matrix, we sum it up, else we return it directly
        if isinstance(x, Tensor) and len(x.shape) > 1:
            return torch.sum(x)
        else:
            return x

    def __repr__(self):
        return f'PostprocessPipeline(ops=layer1:{list(self._modules_operations.keys())[self._selected_idx[0]]}, layer2:{list(self._modules_operations.keys())[self._selected_idx[1]]}, layer3:{list(self._modules_operations.keys())[self._selected_idx[2]]})'

    @property
    def identity(self):
        return self._selected_idx

    @identity.setter
    def identity(self, value):
        self._selected_idx = value
