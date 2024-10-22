import unittest

import torch

from gaswot.pipeline.extraction import ExtractionLine


class TestExtraction(unittest.TestCase):

    def test_extraction(self):
        # test reshape
        x = torch.randn(1, 3, 256, 256)
        extraction = ExtractionLine()
        x = extraction(x)
        print(x.shape)


if __name__ == '__main__':
    unittest.main()
