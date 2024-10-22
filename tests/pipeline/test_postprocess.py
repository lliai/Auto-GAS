import unittest

import torch

from gaswot.pipeline.postprocess import PostprocessPipeline


class TestPostprocess(unittest.TestCase):

    def setUp(self):
        self.x = torch.randn(3, 3, 256, 256)
        self.x = self.x.reshape(self.x.shape[0], -1)
        self.postprocess = PostprocessPipeline()

    def test_postprocess(self):
        print(self.postprocess)
        inp = self.x @ self.x.T
        print(inp.shape, inp)
        out = self.postprocess(inp)
        print(out.shape, out)


if __name__ == '__main__':
    unittest.main()
