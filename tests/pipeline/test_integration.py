import unittest

import torch
from torchvision import transforms

from gaswot.api import TransNASBenchAPI
from gaswot.dataset.naive_dataset import NaiveImageDataset
from gaswot.pipeline.integration import Integration
from gaswot.tnb101.model_builder import create_model


# test integration
class TestIntegration(unittest.TestCase):

    def test_integration(self):

        integ = Integration()

        api = TransNASBenchAPI('data/transnas-bench_v10141024.pth')
        arch = api.get_arch_list('macro')[0]
        net = create_model(arch, 'autoencoder')

        # transform for gan
        T = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_data = NaiveImageDataset(img_folder='./data/rgb', trans=T)

        trainloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True)

        loss_fn = torch.nn.MSELoss()

        inputs = next(iter(trainloader))
        targets = inputs.clone()

        output = integ(net, inputs, targets, loss_fn, split_data=1)

        print(output.shape)

        print(integ)


if __name__ == '__main__':
    unittest.main()
