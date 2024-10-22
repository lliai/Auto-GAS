import torch

from .basePipeline import BasePipeLine


class ExceptionHandling:
    """ Handling pipeline.

    Handling exception that may occur during the process of pipeline, mainly include: nan, inf, -inf, and so on.

    """

    def __init__(self, *args, **kwargs):
        super(ExceptionHandling, self).__init__(*args, **kwargs)
        # Note that no probability is needed for handling pipeline.

    def __call__(self, x, *args, **kwargs):
        # handling nan, inf, -inf, and so on.
        x[torch.isnan(x)] = 0
        x[torch.isinf(x)] = 0
        return x

    def filter_results(self, results):
        results = results[torch.logical_not(torch.isnan(results))]
        return results

    def __repr__(self):
        return f'HandlingException()'
