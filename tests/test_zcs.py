# Import the module to be tested
import torch
from torchvision import transforms

from gaswot.dataset.naive_dataset import NaiveImageDataset
from gaswot.predictor.pruners.predictive import find_measures
from gaswot.tnb101.model_builder import create_model

m = create_model('64-41414-3_33_212', 'autoencoder')

# transform for gan
T = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_data = NaiveImageDataset(img_folder='./data/arona_rgb/rgb', trans=T)

trainloader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

# passed: nwot, fisher, zen, diswot, jacov, l2_norm, plain, snip, flops, params
#         entropy, zico, grad_angle, bn_score, mixup, size, grad_conflict, mgm
#         ntk, nst, sp1, sp2, at, pdist, cc, orm

# nan: epe_nas, grasp, condnum, ntk_trace, jacobian_trace
# 0: synflow, rkd_angle, logits_entropy
# 1: ickd
score = find_measures(
    m,
    trainloader,
    dataload_info=['random', 3, 100],
    device=torch.device('cpu'),
    measure_names=['orm'])
print(score)
