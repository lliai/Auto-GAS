# assemble all pipelines to form an integrated pipeline
import torch
from torch import Tensor

from .binarization import BinaryPipeline
from .correlation import CorrelationPipeline
from .extraction import ExtractionLine
from .handling import ExceptionHandling
from .postprocess import PostprocessPipeline
from .preprocess import PreprocessLine


class Integration:
    """ Integration pipeline is a pipeline that integrates all other pipelines.

    Pipeline:
    1. PreprocessLine
    2. ExtractionLine
    3. CorrelationPipeline
    4. BinaryLine
    5. PostprocessLine

    Error handling is working in the whole process.
    TODO:
        Support visualization of the pipeline.

    """

    def __init__(self, *args, **kwargs):
        super(Integration, self).__init__(*args, **kwargs)
        self._preprocess = PreprocessLine()
        self._extraction = ExtractionLine()
        self._correlation = CorrelationPipeline()
        self._binary = BinaryPipeline()
        self._postprocess = PostprocessPipeline()
        self._handler = ExceptionHandling()  # no identity

        self._identity = self._record_identity()

    def _record_identity(self) -> dict:
        """ Record the identity of each pipeline. """
        _identity = {
            'preprocess': self._preprocess.identity,
            'extraction': self._extraction.identity,
            'correlation': self._correlation.identity,
            'binary': self._binary.identity,
            'postprocess': self._postprocess.identity,
        }
        return _identity

    def __call__(self,
                 model,
                 inputs,
                 targets,
                 loss_fn,
                 split_data=1,
                 *args,
                 **kwargs):
        # preprocess
        inputs = self._preprocess(
            model,
            inputs,
            targets,
            loss_fn,
            split_data=split_data,
            *args,
            **kwargs)
        # extraction no exception
        results = self._extraction(inputs)
        # correlation no exception
        results = self._correlation(results)
        # binary no exception
        results = self._binary(results)
        # postprocess with exception
        results = self._postprocess(results)
        # handling
        results = self._handler.filter_results(results)

        if isinstance(results, list):
            results = results.mean()
        elif isinstance(results, Tensor) and len(results.shape) >= 1:
            results = results.mean()

        if torch.is_complex(results):
            results = results.real

        return results

    def __repr__(self):
        repr_str = f'Integration(\n'
        repr_str += f'    preprocess={self._preprocess.__repr__()}\n'
        repr_str += f'    extraction={self._extraction.__repr__()}\n'
        repr_str += f'    correlation={self._correlation.__repr__()}\n'
        repr_str += f'    binary={self._binary.__repr__()}\n'
        repr_str += f'    postprocess={self._postprocess.__repr__()}\n'
        repr_str += f')'
        return repr_str

    @property
    def identity(self):
        return self._identity

    @identity.setter
    def identity(self, value: dict):
        self._preprocess.identity = value['preprocess']
        self._extraction.identity = value['extraction']
        self._correlation.identity = value['correlation']
        self._binary.identity = value['binary']
        self._postprocess.identity = value['postprocess']
