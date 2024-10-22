import unittest

import torch

from gaswot.pipeline.correlation import CorrelationPipeline


class TestCorrelation(unittest.TestCase):

    def test_correlation(self):
        # test reshape
        x = torch.randn(5, 3, 256, 256)
        x = x.reshape(x.shape[0], -1)
        correlation = CorrelationPipeline()
        x = correlation(x)
        print(x.shape)


if __name__ == '__main__':
    unittest.main()
