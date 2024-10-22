import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_msssim import ms_ssim, ssim
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm

from gaswot.api import TransNASBenchAPI
from gaswot.api.rank_consistency import kendalltau, pearson, spearman
from gaswot.dataset.naive_dataset import NaiveImageDataset
from gaswot.pipeline.integration import Integration
from gaswot.predictor.pruners.predictive import find_measures
from gaswot.tnb101.model_builder import create_model

api = TransNASBenchAPI('data/transnas-bench_v10141024.pth')


def all_same(items):
    """Return True if all elements are the same"""
    return all(x == items[0] for x in items)


def compute_zerocost_score(arch, loss_type='l1', image=None):
    """ compute the zerocost score for each architecture

    Description:
        1. create the model from the architecture
        2. train the model with the trainloader
        3. compute the zerocost score
    """
    if image is None:
        print('image is None, use random image instead')
        image = torch.randn(1, 3, 256, 256)

    # create the model from the architecture
    model = create_model(arch, 'autoencoder')
    # train the model with the trainloader
    output = model(image)
    # compute the zerocost score
    if loss_type == 'l1':
        return torch.mean(torch.abs(image - output)).detach().numpy()
    elif loss_type == 'l2':
        return torch.mean((image - output)**2).detach().numpy()
    elif loss_type == 'ssim':
        return ssim(image, output).detach().numpy()
    elif loss_type == 'ms-ssim':
        return ms_ssim(image, output).detach().numpy()
    else:
        raise NotImplementedError


def evaluate_rank_gaswot_autoencoder(integration: Integration,
                                     trainloader,
                                     loss_fn,
                                     ss_type: str = 'macro',
                                     sample_num: int = 50,
                                     seed: int = 0):
    """ evaluate the rank consistency of autoencoder task

    Description:
        1. enumerate `sample_num` architectures from the api
        2. query the performance of each architecture from api
        3. compute the zerocost score for each architecture
        4. compute the rank consistency of each architecture
    """
    print('Evaluating the rank consistency of autoencoder task...')

    # sample archs from api
    arch_list = api.get_arch_list(ss_type)
    sampled_archs = random.sample(arch_list, sample_num)

    # generate image
    inputs = next(iter(trainloader))
    targets = inputs.clone()

    # records
    gt_list = []
    zc_list = []

    # duplicate checking list
    duplicate_list = []

    for arch in tqdm(sampled_archs):
        # query the performance of each architecture from api
        gt_list.append(
            api.get_single_metric(
                arch, 'autoencoder', 'test_ssim', mode='best'))
        model = create_model(arch, 'autoencoder')

        # compute the zerocost score for each architecture
        zc_score = integration(model, inputs, targets, loss_fn, split_data=1)

        duplicate_list.append(zc_score)

        # duplicate checking
        if len(duplicate_list) > 3 and all_same(duplicate_list):
            return -1, -1, -1

        # exception handling
        if zc_score == -1 or zc_score == 0:
            return -1, -1, -1

        zc_list.append(zc_score.item())

    # compute the rank consistency of each architecture
    kd = kendalltau(gt_list, zc_list)
    sp = spearman(gt_list, zc_list)
    ps = pearson(gt_list, zc_list)

    return kd, sp, ps


def evaluate_rank_zc_autoencoder(trainloader,
                                 ss_type: str = 'macro',
                                 zc_type: str = 'nwot',
                                 sample_num: int = 25,
                                 seed: int = 0):
    """ evaluate the rank consistency of autoencoder task

    Description:
        1. enumerate `sample_num` architectures from the api
        2. query the performance of each architecture from api
        3. compute the zerocost score for each architecture
        4. compute the rank consistency of each architecture
    """
    print(f'Evaluating the rank {zc_type} of autoencoder task...')

    # sample archs from api
    arch_list = api.get_arch_list(ss_type)
    sampled_archs = random.sample(arch_list, sample_num)

    # generate image
    # image = next(iter(trainloader))

    # records
    gt_list = []
    zc_list = []

    for arch in tqdm(sampled_archs):
        # query the performance of each architecture from api
        gt_score = api.get_single_metric(
            arch, 'autoencoder', 'test_ssim', mode='best')

        # compute the zerocost score for each architecture
        net = create_model(arch, 'autoencoder')
        zc_score = find_measures(
            net,
            trainloader,
            dataload_info=['random', 3, 100],
            device=torch.device('cpu'),
            measure_names=[zc_type])

        if gt_score > 0.2:
            # for micro, there are some extreme cases
            gt_list.append(gt_score)
            zc_list.append(zc_score)

    # compute the rank consistency of each architecture
    kd = kendalltau(gt_list, zc_list)
    sp = spearman(gt_list, zc_list)
    ps = pearson(gt_list, zc_list)

    # save the plot of the rank consistency
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the scatter plot
    axs[0].scatter(gt_list, zc_list)
    axs[0].set_xlabel('gt score')
    axs[0].set_ylabel('zc score')
    axs[0].set_title(f'Rank of {zc_type} is {sp:.2f}')

    # Plot the rank plot
    rank_gt_list = np.argsort(gt_list)
    # rank_zc_list = np.argsort(zc_list)
    axs[1].scatter(rank_gt_list, zc_list)
    axs[1].set_xlabel('gt rank')
    axs[1].set_ylabel('zc score')
    axs[1].set_title(f'Rank of {zc_type} is {sp:.2f}')

    fig.suptitle(f'Spearman of {zc_type}', fontsize=16)

    return kd, sp, ps


def random_search_gaswot_proxy(trainloader,
                               loss_fn,
                               ss_type,
                               sample_num,
                               seed,
                               iteration=1000):
    """ random search for gaswot proxy using Pipeline """

    best_zc = -1
    best_instinct = None

    for i in range(iteration):
        print(f'Iteration {i}')

        # sample integration randomly
        instinct = Integration()

        # evaluate the integration
        rank_corr = evaluate_rank_gaswot_autoencoder(instinct, trainloader,
                                                     loss_fn, ss_type,
                                                     sample_num, seed)[-1]
        if rank_corr == -1:
            continue

        # update the best
        if rank_corr > best_zc:
            best_zc = rank_corr
            best_instinct = instinct
            print(f'===> update the best rank corr to {best_zc:.4f}')

        # print the best every 10 iterations
        if i % 10 == 0:
            print(f'Best rank corr is {best_zc:.4f}')
            print(best_instinct)

    return best_instinct, best_zc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_str', type=str, default='resnet101')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--initial_lr', type=float, default=0.001)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='exps/logs')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nopause', action='store_true')
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--device_list', type=str, default='0')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument(
        '--dist_url', type=str, default='tcp://localhost:10001')
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--task', type=str, default='autoencoder')
    parser.add_argument('--sample_num', type=int, default=5, help='sample num')
    parser.add_argument(
        '--loss_type',
        type=str,
        default='l1',
        choices=['l1', 'l2', 'ssim', 'ms-ssim'])
    parser.add_argument(
        '--zc_type', type=str, default='gaswot', help='zero-cost score type')
    parser.add_argument(
        '--ss_type',
        type=str,
        default='macro',
        choices=['macro', 'micro'],
        help='search space type')
    parser.add_argument(
        '--iteration', type=int, default=5, help='iteration num')

    args = parser.parse_args()

    # transform for gan
    T = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = NaiveImageDataset(img_folder='./data/rgb', trans=T)

    loss_fn = torch.nn.L1Loss()

    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True)

    # random search for gaswot proxy
    best_instinct, best_zc = random_search_gaswot_proxy(
        trainloader, loss_fn, args.ss_type, args.sample_num, args.seed,
        args.iteration)

    print(f'Best integration(sp={best_zc}) is:')
    print(best_instinct)
    print(best_instinct.identity)

    # evaluate the rank consistency of autoencoder task
    kd, sp, ps = evaluate_rank_gaswot_autoencoder(best_instinct, trainloader,
                                                  loss_fn, args.ss_type,
                                                  args.sample_num, args.seed)

    print(f'Kendalltau of gaswot is {kd:.4f}')
    print(f'Spearman of gaswot is {sp:.4f}')
    print(f'Pearson of gaswot is {ps:.4f}')
    print(f'Best integration is {best_instinct}')
