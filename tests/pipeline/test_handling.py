import unittest


# test the handling of exception including nan, inf, -inf, and so on.
class TestHandling(unittest.TestCase):

    def test_handling(self):
        import torch

        from gaswot.pipeline.handling import ExceptionHandling
        x = torch.randn(1, 3, 256, 256)
        x[torch.isnan(x)] = float('nan')
        x[torch.isinf(x)] = float('inf')
        x[torch.isinf(x)] = float('-inf')
        handling = ExceptionHandling()
        x = handling(x)
        print(x.shape)
        self.assertEqual(x.shape, (1, 3, 256, 256))


if __name__ == '__main__':
    unittest.main()
