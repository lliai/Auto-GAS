# Code borrowed from the "EAGAN" by marsggbo
# Original code: https://github.com/marsggbo/EAGAN

import os
from copy import deepcopy

import numpy as np
import torch
from network import copy_params, load_params, validate
from tensorboardX import SummaryWriter

import gaswot.archs as archs
import gaswot.utils.cfg as cfg
from gaswot.archs.fully_super_network import Discriminator, Generator
from gaswot.utils.fid_score import (check_or_download_inception,
                                    create_inception_graph)
from gaswot.utils.flop_benchmark import print_FLOPs
from gaswot.utils.inception_score import _init_inception
from gaswot.utils.utils import (count_parameters_in_MB, create_logger,
                                set_log_dir)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # set TensorFlow environment for evaluation (calculate IS and FID)
    _init_inception()
    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    # the first GPU in visible GPUs is dedicated for evaluation (running Inception model)
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for id in range(len(str_ids)):
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 1:
        args.gpu_ids = args.gpu_ids[1:]
    else:
        args.gpu_ids = args.gpu_ids

    # genotype G
    genotype_G = np.load(os.path.join('exps', 'nswot_gen.npy'))

    # import network from genotype
    basemodel_gen = Generator(args, genotype_G)
    gen_net = torch.nn.DataParallel(
        basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # set writer
    print(f'=> resuming from {args.checkpoint}')
    assert os.path.exists(os.path.join('exps', args.checkpoint))

    checkpoint_file = os.path.join('exps', args.checkpoint)
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    epoch = checkpoint['epoch'] - 1
    gen_net.load_state_dict(checkpoint['gen_state_dict'])
    avg_gen_net = deepcopy(gen_net)
    avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    assert args.exp_name
    args.path_helper = set_log_dir('exps', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': epoch // args.val_freq,
    }

    # model size
    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    print_FLOPs(basemodel_gen, (1, args.latent_dim), logger)

    # for visualization
    if args.draw_arch:
        from gaswot.utils.genotype import draw_graph_G
        print(args.path_helper['graph_vis_path'])
        draw_graph_G(
            genotype_G,
            save=True,
            file_path=os.path.join(args.path_helper['graph_vis_path'],
                                   'nswot_gen'))
    fixed_z = torch.cuda.FloatTensor(
        np.random.normal(0, 1, (100, args.latent_dim)))

    fixed_z = torch.load('test_noise.pt')
    print('zzzzzz:', fixed_z)

    # test
    load_params(gen_net, gen_avg_param)
    inception_score, std, fid_score = validate(args, fixed_z, fid_stat,
                                               gen_net, writer_dict)
    logger.info(
        f'Inception score mean: {inception_score}, Inception score std: {std}, '
        f'FID score: {fid_score} || @ epoch {epoch}.')


if __name__ == '__main__':
    main()
