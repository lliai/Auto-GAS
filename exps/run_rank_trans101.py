import argparse
import random

# import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_msssim import ms_ssim, ssim
from torchvision import transforms
from tqdm import tqdm

from gaswot.api import TransNASBenchAPI
from gaswot.api.rank_consistency import kendalltau, pearson, spearman
from gaswot.dataset.naive_dataset import NaiveImageDataset
from gaswot.predictor.pruners.predictive import find_measures
from gaswot.tnb101.model_builder import create_model

api = TransNASBenchAPI(
    '/data2/dongpeijie/share/bench/transnas-bench_v10141024.pth')


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


def evaluate_rank_gaswot_autoencoder(trainloader,
                                     loss_type: str = 'l1',
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
    image = next(iter(trainloader))

    # records
    gt_list = []
    zc_list = []

    for arch in tqdm(sampled_archs):
        # query the performance of each architecture from api
        gt_list.append(
            api.get_single_metric(
                arch, 'autoencoder', 'test_ssim', mode='best'))
        # compute the zerocost score for each architecture
        zc_list.append(compute_zerocost_score(arch, loss_type, image))

    # compute the rank consistency of each architecture
    kd = kendalltau(gt_list, zc_list)
    sp = spearman(gt_list, zc_list)
    ps = pearson(gt_list, zc_list)

    # save the plot of the rank consistency
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # # Plot the scatter plot
    # axs[0].scatter(gt_list, zc_list)
    # axs[0].set_xlabel('gt_list')
    # axs[0].set_ylabel('zc_list')
    # axs[0].set_title('Scatter plot of gt_list and zc_list')

    # # Plot the rank plot
    # rank_gt_list = np.argsort(gt_list)
    # rank_zc_list = np.argsort(zc_list)
    # axs[1].scatter(rank_gt_list, rank_zc_list)
    # axs[1].set_xlabel('Rank of gt_list')
    # axs[1].set_ylabel('Rank of zc_list')
    # axs[1].set_title('Rank plot of gt_list and zc_list')

    # fig.suptitle('Spearman of gaswot', fontsize=16)

    # plt.savefig(
    #     f'./output/rank/{loss_type}_{ss_type}_gaswot_sp_{sp:.2f}_{seed:03d}.jpg'
    # )

    return kd, sp, ps


def evaluate_rank_zc_autoencoder(trainloader,
                                 ss_type: str = 'macro',
                                 zc_type: str = 'nwot',
                                 sample_num: int = 50,
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

        # print(zc_score)
        # import pdb; pdb.set_trace()

        # if gt_score > 0.2:
        # for micro, there are some extreme cases
        gt_list.append(gt_score)
        zc_list.append(zc_score)
        # print(f'gt_score: {gt_score}, zc_score: {zc_score}')

    def sum_arr(arr):
        sum = 0.0
        for i in range(len(arr)):
            val = arr[i]
            val = val if type(val) is torch.Tensor else torch.tensor(val)
            sum += torch.sum(val)
        return sum

    print(f'gt_list = {gt_list}')
    print(f'zc_list = {zc_list}')

    # compute the rank consistency of each architecture
    if isinstance(zc_list[0], list):
        zc_list = [sum_arr(zc) for zc in zc_list]
    try:
        kd = kendalltau(gt_list, zc_list)
        sp = spearman(gt_list, zc_list)
        ps = pearson(gt_list, zc_list)
    except:
        import pdb
        pdb.set_trace()

    # save the plot of the rank consistency
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # # Plot the scatter plot
    # axs[0].scatter(gt_list, zc_list)
    # axs[0].set_xlabel('gt score')
    # axs[0].set_ylabel('zc score')
    # axs[0].set_title(f'Rank of {zc_type} is {sp:.2f}')

    # # Plot the rank plot
    # rank_gt_list = np.argsort(gt_list)
    # # rank_zc_list = np.argsort(zc_list)
    # axs[1].scatter(rank_gt_list, zc_list)
    # axs[1].set_xlabel('gt rank')
    # axs[1].set_ylabel('zc score')
    # axs[1].set_title(f'Rank of {zc_type} is {sp:.2f}')

    # fig.suptitle(f'Spearman of {zc_type}', fontsize=16)

    # plt.savefig(
    #     f'./output/rank/{ss_type}_{zc_type}_sp_{sp:.2f}_{seed:03d}.jpg')

    return kd, sp, ps


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
    parser.add_argument(
        '--sample_num', type=int, default=50, help='sample num')
    parser.add_argument(
        '--loss_type',
        type=str,
        default='l1',
        choices=['l1', 'l2', 'ssim', 'ms-ssim'])
    parser.add_argument(
        '--zc_type', type=str, default='nwot', help='zero-cost score type')
    parser.add_argument(
        '--ss_type',
        type=str,
        default='macro',
        choices=['macro', 'micro'],
        help='search space type')

    args = parser.parse_args()

    # transform for gan
    T = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = NaiveImageDataset(
        img_folder='/data2/dongpeijie/share/rgb', trans=T)

    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True)

    if args.task != 'autoencoder':
        raise NotImplementedError

    if args.zc_type == 'gaswot':
        kd, sp, ps = evaluate_rank_gaswot_autoencoder(trainloader,
                                                      args.loss_type,
                                                      args.ss_type,
                                                      args.sample_num,
                                                      args.seed)
    else:
        kd_list, sp_list, ps_list = [], [], []
        for i in range(5):
            kd, sp, ps = evaluate_rank_zc_autoencoder(trainloader,
                                                      args.ss_type,
                                                      args.zc_type,
                                                      args.sample_num,
                                                      args.seed)
            print(f'{i}-th kd: {kd}, sp: {sp}, ps: {ps}')
            kd_list.append(kd)
            sp_list.append(sp)
            ps_list.append(ps)

        m_kd, v_kd = np.mean(kd_list), np.var(kd_list)
        m_sp, v_sp = np.mean(sp_list), np.var(sp_list)
        m_ps, v_ps = np.mean(ps_list), np.var(ps_list)

        print(' * Rank consistency of autoencoder task:')
        print(f' * zc type: {args.zc_type}')
        print(f' * Kendall Tau: {m_kd}+{v_kd}')
        print(f' * Spearman: {m_sp}+{v_sp}')
        print(f' * Pearson: {m_ps}+{v_ps}')

    # Rank existing zero-cost proxies in batch mode

    # not work: condnum, hawq, jacobian_trace, "knas", logits_entropy, "ntk", "ntk_trace" hessian_trace
    # candidate_zc_list = ["size", "bn_score", "diswot","entropy","epe_nas","fisher","grad_angle","grad_conflict","grad_norm","grasp", "jacov","l2_norm", "logsynflow","mixup","nwot","orm","plain","snip","synflow","zen","zico", 'meco']

    # OK: nwot_Kmats', 'lnwot',
    # 'lnwot_Kmats'
    # candidate_zc_list = ['tcet_syn_none', 'tcet_syn_log', 'tcet_syn_log1p', 'tcet_syn_norm',
    #             'tcet_snip_none', 'tcet_snip_log', 'tcet_snip_log1p',
    #             'tcet_snip_norm', 'tcet', 'nwot']

    # result_dict = {}

    # for ss_type in ['macro', 'micro']:

    #     if ss_type not in result_dict:
    #         result_dict[ss_type] = dict()

    #     for zc in candidate_zc_list:
    #         print("====== current zc is:", zc)

    #         # try:
    #         if True:
    #             kd, sp, ps = evaluate_rank_zc_autoencoder(trainloader, args.ss_type, zc, args.sample_num, args.seed)
    #         # except:
    #         #     print(f"There are errors in {zc}!!!")
    #         #     continue
    #         print(f' * Rank consistency of autoencoder task:')
    #         print(f' * ZC type: {zc}')
    #         print(' * Kendall Tau: {:.4f}'.format(kd))
    #         print(' * Spearman: {:.4f}'.format(sp))
    #         print(' * Pearson: {:.4f}'.format(ps))

    #         if zc not in result_dict[ss_type]:
    #             result_dict[ss_type][zc] = dict()
    #             result_dict[ss_type][zc]['kd'] = kd
    #             result_dict[ss_type][zc]['sp'] = sp
    #             result_dict[ss_type][zc]['ps'] = ps
    #         else:
    #             result_dict[ss_type][zc]['kd'] = kd
    #             result_dict[ss_type][zc]['sp'] = sp
    #             result_dict[ss_type][zc]['ps'] = ps

    # print(result_dict)
    # # print the result dict
    # for ss_type in result_dict:
    #     print("====== current ss_type is:", ss_type)
    #     for zc in result_dict[ss_type]:
    #         print("====== current zc is:", zc)
    #         print("kd:", result_dict[ss_type][zc]['kd'])
    #         print("sp:", result_dict[ss_type][zc]['sp'])
    #         print("ps:", result_dict[ss_type][zc]['ps'])
    #         print("======")

    # # # save the result dict to json file
    # # import json
    # # with open('result_dict.json', 'w') as fp:
    # #     json.dump(result_dict, fp)
