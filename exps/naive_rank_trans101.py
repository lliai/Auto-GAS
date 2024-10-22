import argparse
import random

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from gaswot.api import TransNASBenchAPI
from gaswot.api.rank_consistency import kendalltau, pearson, spearman
from gaswot.dataset.naive_dataset import NaiveImageDataset
from gaswot.pipeline.binarization import (above_mean_op, above_median_op,
                                          above_zero_op, sign_op)
from gaswot.pipeline.postprocess import (
    eig_op, element_wise_abs_op, element_wise_abslog_op, element_wise_log_op,
    element_wise_normalized_sum_op, element_wise_pow_op,
    element_wise_sigmoid_op, element_wise_sqrt_op, element_wise_tanh_op,
    frobenius_norm_op, hamming_op, l1_norm_op, l2_norm_op, logsoftmax_op,
    no_op, normalize_op, slogdet_op, softmax_op, trace_op, transpose_op)
from gaswot.pipeline.preprocess import (
    trans_abs, trans_batch, trans_batchnorm, trans_bmm, trans_catt,
    trans_channel, trans_drop, trans_exp, trans_leaky_relu, trans_local_s1,
    trans_local_s2, trans_local_s4, trans_log, trans_logsoftmax_C,
    trans_logsoftmax_HW, trans_logsoftmax_N, trans_mask,
    trans_min_max_normalize, trans_mish, trans_mm, trans_multi_scale_r1,
    trans_multi_scale_r2, trans_multi_scale_r4, trans_natt, trans_nop,
    trans_norm_C, trans_norm_HW, trans_norm_N, trans_relu, trans_satt,
    trans_scale, trans_sigmoid, trans_softmax_C, trans_softmax_HW,
    trans_softmax_N, trans_sqrt, trans_swish, trans_tanh)
from gaswot.tnb101.model_builder import create_model

api = TransNASBenchAPI(
    '/data2/dongpeijie/share/bench/transnas-bench_v10141024.pth')

s_operations = [
    trans_tanh, trans_swish, trans_sqrt, trans_softmax_N, trans_softmax_HW,
    trans_softmax_C, trans_abs, trans_batch, trans_batchnorm, trans_bmm,
    trans_catt, trans_channel, trans_drop, trans_exp, trans_leaky_relu,
    trans_local_s1, trans_local_s2, trans_local_s4, trans_log,
    trans_logsoftmax_C, trans_logsoftmax_HW, trans_logsoftmax_N, trans_mask,
    trans_min_max_normalize, trans_mish, trans_mm, trans_multi_scale_r1,
    trans_multi_scale_r2, trans_multi_scale_r4, trans_natt, trans_nop,
    trans_norm_C, trans_norm_HW, trans_norm_N, trans_relu, trans_satt,
    trans_scale, trans_sigmoid
]

h_operations = [above_mean_op, above_median_op, above_zero_op, sign_op]

p_operations = [
    no_op, no_op, element_wise_log_op, element_wise_abslog_op,
    element_wise_abs_op, element_wise_pow_op, slogdet_op, eig_op, trace_op,
    logsoftmax_op, l1_norm_op, l2_norm_op, element_wise_normalized_sum_op,
    normalize_op, element_wise_sigmoid_op, element_wise_tanh_op,
    element_wise_sqrt_op, transpose_op, frobenius_norm_op, hamming_op
]


def sample_ops(operations, num_ops):
    r_list = []
    for i in range(num_ops):
        r_list.append(random.choice(operations))
    return r_list


class GasWOTProxy(object):
    h_ops: list = None
    p_ops: list = None

    def __init__(self,
                 s_ops: list = None,
                 h_ops: list = None,
                 p_ops: list = None):
        self.s_ops = s_ops if s_ops is not None else sample_ops(
            s_operations, 1)
        self.h_ops = h_ops if h_ops is not None else sample_ops(
            h_operations, 1)
        self.p_ops = p_ops if p_ops is not None else sample_ops(
            p_operations, 3)

    def update(self,
               s_ops: list = None,
               h_ops: list = None,
               p_ops: list = None):
        assert s_ops is not None and len(s_ops) == 2
        assert h_ops is not None and len(h_ops) == 1
        assert p_ops is not None and len(p_ops) == 3
        self.s_ops = s_ops
        self.h_ops = h_ops
        self.p_ops = p_ops

    def __repr__(self) -> str:
        s_ops_str = ', '.join([op.__name__ for op in self.s_ops])
        h_ops_str = ', '.join([op.__name__ for op in self.h_ops])
        p_ops_str = ', '.join([op.__name__ for op in self.p_ops])
        return f'GasWOTProxy(s_ops=[{s_ops_str}], h_ops=[{h_ops_str}], p_ops=[{p_ops_str}])'


def compute_auto_gas_proxy(net, inputs):
    batch_size = len(inputs)

    def counting_forward_hook(module, inp, out):
        # preprocess
        inp = inp[0]
        inp = inp * torch.sigmoid(inp)

        # flatten the input
        inp = inp.view(inp.size(0), -1)
        # the operations are applied in order
        inp = (inp > inp.median()).float()

        # split the input into 0 or 1
        # x = (inp > 0).float()
        x = inp
        K = x @ x.t()
        K2 = (1.0 - x) @ (1.0 - x.t())

        # hamming distance
        net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()

    net.K = np.zeros((batch_size, batch_size))
    for name, module in net.named_modules():
        module_type = str(type(module))
        if ('ReLU' in module_type):
            module.register_forward_hook(counting_forward_hook)

    x = torch.clone(inputs)
    net(x)
    # s, jc = np.linalg.slogdet(net.K)

    # post process
    Kt = net.K
    p_ops = [element_wise_pow_op, element_wise_abslog_op, l2_norm_op]
    for op in p_ops:
        if not isinstance(Kt, np.ndarray):
            continue
        Kt = op(Kt)
        if Kt is None:
            import pdb
            pdb.set_trace()

    # convert to scalar
    if isinstance(Kt, np.ndarray):
        if len(Kt) > 1:
            Kt = Kt.mean()
    elif Kt is None:
        import pdb
        pdb.set_trace()
    return Kt


def compute_proxy(net, inputs, gaswot_proxy: GasWOTProxy):
    """ proxy is defined by operations, which is randomly sampled from the operation list
    """
    assert gaswot_proxy is not None, f'gaswot proxy is None'

    batch_size = len(inputs)

    def counting_forward_hook(module, inp, out):
        # preprocess
        inp = inp[0]
        for op in gaswot_proxy.s_ops:
            inp = op(inp)

        # flatten the input
        inp = inp.view(inp.size(0), -1)
        # the operations are applied in order
        for op in gaswot_proxy.h_ops:
            inp = op(inp)

        # split the input into 0 or 1
        # x = (inp > 0).float()
        x = inp
        K = x @ x.t()
        K2 = (1.0 - x) @ (1.0 - x.t())

        # hamming distance
        net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()

    net.K = np.zeros((batch_size, batch_size))
    for name, module in net.named_modules():
        module_type = str(type(module))
        if ('ReLU' in module_type):
            module.register_forward_hook(counting_forward_hook)

    x = torch.clone(inputs)
    net(x)
    # s, jc = np.linalg.slogdet(net.K)

    # post process
    Kt = net.K
    for op in gaswot_proxy.p_ops:
        if not isinstance(Kt, np.ndarray):
            continue
        Kt = op(Kt)
        if Kt is None:
            import pdb
            pdb.set_trace()

    # convert to scalar
    if isinstance(Kt, np.ndarray):
        if len(Kt) > 1:
            Kt = Kt.mean()
    elif Kt is None:
        import pdb
        pdb.set_trace()
    return Kt


def evaluate_proxy(gaswot_proxy, trainloader, ss_type, sample_num=50):
    """ evaluate the rank consistency of autoencoder task
    """
    print(f'Evaluating proxy ... {gaswot_proxy}')
    arch_list = api.get_arch_list(ss_type)
    sampled_archs = random.sample(arch_list, sample_num)

    inputs = next(iter(trainloader))
    targets = inputs.clone()  # autoencoder task

    def all_same(items):
        return all(x == items[0] for x in items)

    gt_list, zc_list = [], []

    for arch in tqdm(sampled_archs):
        model = create_model(arch, 'autoencoder')

        gt = api.get_single_metric(
            arch, 'autoencoder', 'test_ssim', mode='best')
        gt_list.append(gt)

        tmp_zc_list = []  # repeat for 3 times
        for _ in range(3):
            # tmp_zc = compute_proxy(model, inputs, gaswot_proxy)
            tmp_zc = compute_auto_gas_proxy(model, inputs)
            # tmp_zc[np.isinf(tmp_zc) | np.isnan(tmp_zc)] = 0
            tmp_zc_list.append(tmp_zc)
        zc = np.mean(tmp_zc_list)
        # zc = compute_proxy(model, inputs, gaswot_proxy)

        # Exception handling
        zc = zc[np.logical_not(np.isnan(zc))]
        zc = zc[np.logical_not(np.isinf(zc))]
        zc[np.isinf(zc)] = 0
        zc[np.isnan(zc)] = 0

        if isinstance(zc, complex):
            print('is complex!')
            zc = zc.real
            return -1

        if isinstance(zc, np.ndarray):
            zc = zc.item()
        else:
            zc = zc
        zc_list.append(zc)

        if len(zc_list) > 3 and all_same(zc_list):
            return -1

    # DEBUG
    print('gt_list:', f'{gt_list}')
    print('zc_list:', f'{zc_list}')

    # zc_list - max(zc_list)
    zc_list = np.array(zc_list)
    zc_list = zc_list - np.min(zc_list)
    zc_list = zc_list.tolist()

    print(f'Currently is evaluating {gaswot_proxy} ...')
    print('current pearson', f'{pearson(gt_list, zc_list)}')
    print('current kendalltau', f'{kendalltau(gt_list, zc_list)}')
    print('current spearman', f'{spearman(gt_list, zc_list)}')

    return pearson(gt_list,
                   zc_list), kendalltau(gt_list,
                                        zc_list), spearman(gt_list, zc_list)


def search_proxy(trainloader, ss_type='macro', num_trials=100, sample_num=50):
    best_proxy = None
    best_score = float('-inf')

    for _ in range(num_trials):
        candidate_proxy = GasWOTProxy()
        try:
            ps_coeff = evaluate_proxy(candidate_proxy, trainloader, ss_type,
                                      sample_num)
        except:
            continue

        if ps_coeff > best_score:
            best_proxy = candidate_proxy
            best_score = ps_coeff

    return best_proxy, best_score


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
        '--zc_type', type=str, default='gaswot', help='zero-cost score type')
    parser.add_argument(
        '--ss_type',
        type=str,
        default='macro',
        choices=['macro', 'micro'],
        help='search space type')
    parser.add_argument(
        '--iteration', type=int, default=100, help='iteration num')

    args = parser.parse_args()

    print(f'Current number of samples is {args.sample_num}')

    # transform for gan
    T = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = NaiveImageDataset(
        img_folder='/data2/dongpeijie/share/rgb', trans=T)

    loss_fn = torch.nn.L1Loss()

    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True)

    proxy1 = GasWOTProxy(
        s_ops=[trans_mask],
        h_ops=[above_mean_op],
        p_ops=[l2_norm_op, l1_norm_op, trace_op])

    proxy2 = GasWOTProxy(
        s_ops=[trans_local_s1],
        h_ops=[above_median_op],
        p_ops=[slogdet_op, no_op, element_wise_pow_op])

    proxy3 = GasWOTProxy(
        s_ops=[trans_softmax_HW],
        h_ops=[above_zero_op],
        p_ops=[element_wise_log_op, element_wise_abslog_op, hamming_op])

    proxy4 = GasWOTProxy(
        s_ops=[trans_swish],
        h_ops=[above_median_op],
        p_ops=[element_wise_pow_op, element_wise_abslog_op, l2_norm_op])

    proxy5 = GasWOTProxy(
        s_ops=[trans_local_s1],
        h_ops=[above_zero_op],
        p_ops=[slogdet_op, l1_norm_op, element_wise_normalized_sum_op])

    def avg_std(numbers):
        # Average
        avg = sum(numbers) / len(numbers)
        # Standard Deviation
        variance = sum((x - avg)**2 for x in numbers) / len(numbers)
        std_dev = variance**0.5
        return avg, std_dev

    ps_list, kd_list, sp_list = [], [], []
    proxy_cand = [proxy4]

    proxy2ps = dict()
    proxy2kd = dict()
    proxy2sp = dict()

    for n, pcand in enumerate(proxy_cand):
        print(f'*** Evaluating proxy {pcand} ***')
        ps_list = []
        for i in range(3):
            # evaluate the rank consistency of autoencoder task
            ps, kd, sp = evaluate_proxy(pcand, trainloader, args.ss_type,
                                        args.sample_num)
            ps_list.append(ps)
            kd_list.append(kd)
            sp_list.append(sp)

        avg, std = avg_std(ps_list)
        proxy2ps[f'proxy{n}'] = {'avg': avg, 'std': std}
        avg, std = avg_std(kd_list)
        proxy2kd[f'proxy{n}'] = {'avg': avg, 'std': std}
        avg, std = avg_std(sp_list)
        proxy2sp[f'proxy{n}'] = {'avg': avg, 'std': std}

        print(proxy2ps)
        print(proxy2kd)
        print(proxy2sp)

    for k, v in proxy2ps.items():
        print(f"For ps: {k}, avg is {v['avg']} +- {v['std']}")
    for k, v in proxy2kd.items():
        print(f"For kd: {k}, avg is {v['avg']} +- {v['std']}")
    for k, v in proxy2sp.items():
        print(f"For sp: {k}, avg is {v['avg']} +- {v['std']}")
