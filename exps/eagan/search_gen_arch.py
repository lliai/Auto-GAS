# Code borrowed from the "EAGAN" by marsggbo
# Original code: https://github.com/marsggbo/EAGAN

import os
from copy import deepcopy

import numpy as np
import search_cfg
import torch
import torch.nn as nn
from algorithms.search_algs import GanAlgorithm
from tqdm import tqdm
from trainer.trainer_generator import GenTrainer

import gaswot.archs as archs
import gaswot.dataset.datasets as datasets
from gaswot.archs.fully_super_network import simple_Discriminator
from gaswot.archs.super_network import Generator
from gaswot.utils.fid_score import (check_or_download_inception,
                                    create_inception_graph)
from gaswot.utils.inception_score import _init_inception
from gaswot.utils.utils import (count_parameters_in_MB, create_logger,
                                set_log_dir)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False


def main():
    args = search_cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
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
    gan_alg = GanAlgorithm(args)

    # import network from genotype
    basemodel_gen = Generator(args)
    gen_net = torch.nn.DataParallel(
        basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
    basemodel_dis = simple_Discriminator()
    dis_net = torch.nn.DataParallel(
        basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(
                    args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train
    # epoch number for dis_net
    args.max_epoch_D = args.max_epoch_G * args.n_critic
    if args.max_iter_G:
        args.max_epoch_D = np.ceil(args.max_iter_G * args.n_critic /
                                   len(train_loader))

    # set TensorFlow environment for evaluation (calculate IS and FID)
    _init_inception()

    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    if args.dataset.lower() == 'cifar10':
        fid_stat = './fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = './fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)
    gen_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gen_net.parameters()), args.g_lr,
        (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, dis_net.parameters()), args.d_lr,
        (args.beta1, args.beta2))

    # initial
    start_epoch = 0
    # set writer
    args.path_helper = set_log_dir('exps', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    # model size
    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))
    trainer_gen = GenTrainer(args, gen_net, dis_net, gen_optimizer,
                             dis_optimizer, train_loader, gan_alg, None)
    best_genotypes = None
    # search genarator

    for epoch in tqdm(
            range(int(start_epoch), int(args.epoch_generator)),
            desc='search genearator'):
        if epoch >= args.warmup and (epoch -
                                     args.warmup) % args.ga_interval == 0:
            best_genotypes = trainer_gen.search_evol_arch(epoch, fid_stat)

            for index, best_genotype in enumerate(best_genotypes):
                file_path = os.path.join(
                    args.path_helper['ckpt_path'],
                    'best_gen_{}_{}.npy'.format(str(epoch), str(index)))
                np.save(file_path, best_genotype)
    # save checkpoint
    checkpoint_file = os.path.join(args.path_helper['ckpt_path'],
                                   'gen_checkpoint')
    ckpt = {'epoch': epoch, 'weight': gen_net.state_dict()}
    torch.save(ckpt, checkpoint_file)


if __name__ == '__main__':
    main()
