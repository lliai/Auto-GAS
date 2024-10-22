import unittest

import torch

from gaswot.pipeline.binarization import BinaryPipeline


class TestBinarization(unittest.TestCase):

    def setUp(self):
        self.x = torch.randn(1, 3, 2, 2)
        self.x = self.x.reshape(self.x.shape[0], -1)
        self.binarization = BinaryPipeline()

    def test_above_zero_op(self):
        print(self.x)
        out = self.binarization(self.x)
        print(out)

    def test_binarization_repr(self):
        print(self.binarization)


if __name__ == '__main__':
    unittest.main()
