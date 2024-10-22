import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

from gaswot.tnb101.models.net_infer.net_macro import MacroNet


class FFEncoder(nn.Module):
    """Encoder class for the definition of backbone including resnet50 and MacroNet()"""

    def __init__(self, encoder_str, task_name=None):
        super(FFEncoder, self).__init__()
        self.encoder_str = encoder_str

        # Initialize network
        if self.encoder_str == 'resnet50':
            self.network = models.resnet50()  # resnet50: Bottleneck, [3,4,6,3]
            # Adjust according to task
            if task_name in [
                    'autoencoder', 'normal', 'inpainting', 'segmentsemantic'
            ]:
                self.network.inplanes = 1024
                self.network.layer4 = self.network._make_layer(
                    models.resnet.Bottleneck, 512, 3, stride=1, dilate=False)
                self.network = nn.Sequential(
                    *list(self.network.children())[:-2], )
            else:
                self.network = nn.Sequential(
                    *list(self.network.children())[:-2])
        else:
            self.network = MacroNet(encoder_str, structure='backbone')

    def forward(self, x):
        x = self.network(x)
        return x

    def forward_with_features(self, x):
        return self.network.forward_with_features(x)


if __name__ == '__main__':
    # net = FFEncoder("64-41414-3_33_333", 'segmentsemantic').cuda()
    net = FFEncoder('resnet50', 'autoencoder').cuda()
    # x = torch.randn([2, 3, 256, 256])
    # print(net(x).shape)
    # print(net)
    summary(net, (3, 256, 256))
