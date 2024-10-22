import unittest

import torch

from gaswot.api import TransNASBenchAPI
from gaswot.pipeline.preprocess import PreprocessLine
from gaswot.tnb101.model_builder import create_model


class TestPreprocess(unittest.TestCase):

    def test_preprocess(self):
        api = TransNASBenchAPI('data/transnas-bench_v10141024.pth')
        arch = api.get_arch_list('macro')[0]
        net = create_model(arch, 'autoencoder')

        # test preprocess
        preprocess = PreprocessLine()

        inputs = torch.randn(1, 3, 256, 256)
        targets = torch.randn(1, 3, 256, 256)
        loss_fn = torch.nn.MSELoss()

        # test get_activation
        # activation_list = preprocess._get_activation(net, inputs, targets, loss_fn)
        # print(len(activation_list))
        # for a in activation_list:
        #     print(a.shape)

        # test get_gradient_by_weight
        gradient_list = preprocess._get_gradient_by_weight(
            net, inputs, targets, loss_fn)
        print(len(gradient_list))
        for g in gradient_list:
            print(g.shape)

        # test get_fake_jacob_by_input
        # fake_jacob_list = preprocess._get_fake_jacob_by_input(net, inputs, targets, loss_fn)
        # print(len(fake_jacob_list))
        # for j in fake_jacob_list:
        #     print(j.shape)


if __name__ == '__main__':
    unittest.main()
