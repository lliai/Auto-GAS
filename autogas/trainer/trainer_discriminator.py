# Code borrowed from the "EAGAN" by marsggbo
# Original code: https://github.com/marsggbo/EAGAN

import heapq
import logging
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from measures import compute_meco_dis
from tqdm import tqdm

import gaswot.utils.cfg as cfg
from gaswot.archs.fully_super_network import Discriminator
from gaswot.utils.fid_score import calculate_fid_given_paths
from gaswot.utils.inception_score import get_inception_score
from gaswot.utils.sort import CARS_NSGA
from gaswot.utils.utils import count_parameters_in_MB

logger = logging.getLogger(__name__)


class DisTrainer():

    def __init__(self, args, gen_net, dis_net, gen_optimizer, dis_optimizer,
                 train_loader, gan_alg, gen_genotype):
        self.args = args
        self.gen_net = gen_net
        self.dis_net = dis_net
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.train_loader = train_loader
        self.gan_alg = gan_alg
        self.gen_genotype = gen_genotype

        self.genotypes = np.stack(
            [gan_alg.search_dis() for i in range(args.num_individual)], axis=0)
        self.base_weights = None
        self.weights = [0 for i in range(args.num_individual)]
        for i in range(len(self.weights)):
            self.weights[i] = deepcopy(copy_weight(self.gen_net))

    def search_evol_arch(self, epoch, fid_stat):
        proxy_values, is_values, fid_values, params = np.zeros(
            len(self.genotypes)), np.zeros(len(self.genotypes)), np.zeros(
                len(self.genotypes)), np.zeros(len(self.genotypes))

        keep_N, selected_N = len(self.genotypes) // 2, self.args.num_selected
        for idx, genotype_D in enumerate(tqdm(self.genotypes)):

            is_value, is_std, fid_value, proxy_avg = self.validate(
                idx, fid_stat, genotype_D)
            proxy_values[idx] = proxy_avg

            param_szie = count_parameters_in_MB(
                Discriminator(self.args, genotype_D))
            is_values[idx] = is_value
            fid_values[idx] = fid_value
            params[idx] = param_szie
        logger.info(
            f'mean_IS_values: {np.mean(is_values)}, mean_FID_values: {np.mean(fid_values)},@ epoch {epoch}.'
        )
        obj = [is_values, fid_values, params]

        keep, selected = CARS_NSGA(proxy_values, obj,
                                   keep_N), CARS_NSGA(proxy_values, obj,
                                                      selected_N)
        for i in selected:
            logger.info(
                f'genotypes_{i}, IS_values: {is_values[i]}, FID_values: {fid_values[i]}, param_szie: {params[i]}, proxy_values: {proxy_values[i]}|| @ epoch {epoch}.'
            )

        keep_genotypes = self.genotypes[keep]
        offsprings = self.gen_offspring(keep_genotypes, epoch)
        selected_genotypes = self.genotypes[selected]
        self.base_weights = self.weights[selected[0]]
        self.genotypes = np.concatenate((keep_genotypes, offsprings), axis=0)
        return selected_genotypes

    def validate(self, idx, fid_stat, genotype_D):
        gen_net = self.gen_net.eval()
        gen_net.load_state_dict(self.weights[idx])
        # get fid and inception score
        fid_buffer_dir = os.path.join(self.args.path_helper['sample_path'],
                                      'fid_buffer')
        os.makedirs(fid_buffer_dir, exist_ok=True)
        eval_iter = self.args.num_eval_imgs // self.args.eval_batch_size
        img_list = list()
        for iter_idx in tqdm(range(eval_iter), desc='sample images'):
            z = torch.cuda.FloatTensor(
                np.random.normal(
                    0, 1, (self.args.eval_batch_size, self.args.latent_dim)))
            # generate a batch of images
            gen_imgs = gen_net(
                z, self.gen_genotype).mul_(127.5).add_(127.5).clamp_(
                    0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                       torch.uint8).numpy()
            for img_idx, img in enumerate(gen_imgs):
                file_name = os.path.join(fid_buffer_dir,
                                         f'iter{iter_idx}_b{img_idx}.png')
                imsave(file_name, img)
            img_list.extend(list(gen_imgs))

        logger.info('=> calculate infoScore')
        dis_net = self.dis_net.eval()

        args = cfg.parse_args()
        if args.dataset == 'cifar10':
            gf_dim_gas = args.latent_dim
        else:
            gf_dim_gas = args.gf_dim
        avg_proxy_score = compute_meco_dis(dis_net, gen_net, self.gen_genotype,
                                           genotype_D, z, args.img_size,
                                           gf_dim_gas,
                                           self.args.eval_batch_size)

        # get inception score
        logger.info('=> calculate inception score')
        mean, std = get_inception_score(img_list)
        # get fid score
        logger.info('=> calculate fid score')
        fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat],
                                              inception_path=None)

        print('IS(mean, std):', mean, std)
        print('FID:', fid_score)

        #return mean, std, fid_score
        return mean, std, fid_score, avg_proxy_score

    def gen_offspring(self, alphas, epoch, offspring_ratio=1.0):
        """Generate offsprings.
        :param alphas: Parameteres for populations
        :type alphas: nn.Tensor
        :param offspring_ratio: Expanding ratio
        :type offspring_ratio: float
        :return: The generated offsprings
        :rtype: nn.Tensor
        """
        n_offspring = int(offspring_ratio * alphas.shape[0])
        offsprings = []
        while len(offsprings) != n_offspring:
            rand = np.random.rand()
            if rand < 0.5:
                alphas_c = self.mutation(alphas[np.random.randint(
                    0, alphas.shape[0])])
            else:
                a, b = np.random.randint(0,
                                         alphas.shape[0]), np.random.randint(
                                             0, alphas.shape[0])
                while (a == b):
                    a, b = np.random.randint(
                        0, alphas.shape[0]), np.random.randint(
                            0, alphas.shape[0])
                alphas_c = self.crossover(alphas[a], alphas[b])
            if not self.gan_alg.judge_repeat_dis(alphas_c):
                offsprings.append(alphas_c)
        offsprings = np.stack(offsprings, axis=0)
        return offsprings

    def judge_repeat(self, alphas, new_alphas):
        """Judge if two individuals are the same.
        :param alphas_a: An individual
        :type alphas_a: nn.Tensor
        :param new_alphas: An individual
        :type new_alphas: nn.Tensor
        :return: True or false
        :rtype: nn.Tensor
        """
        diff = np.reshape(
            np.absolute(alphas - np.expand_dims(new_alphas, axis=0)),
            (alphas.shape[0], -1))
        diff = np.sum(diff, axis=1)
        return np.sum(diff == 0)

    def crossover(self, alphas_a, alphas_b):
        """Crossover for two individuals."""
        # alpha a
        new_alphas = alphas_a.copy()
        # alpha b
        layer = random.randint(0, 2)
        index = random.randint(0, 6)
        while (new_alphas[layer][index] == alphas_a[layer][index]):
            layer = random.randint(0, 2)
            index = random.randint(0, 6)
            new_alphas[layer][index] = alphas_b[layer][index]
            if index >= 1 and index < 3 and new_alphas[layer][
                    1] == 0 and new_alphas[layer][2] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
            elif index >= 3 and new_alphas[layer][3] == 0 and new_alphas[
                    layer][4] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
        return new_alphas

    def mutation(self, alphas_a, ratio=0.5):
        """Mutation for an individual"""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, 2)
        index = random.randint(0, 6)
        if index == 0:
            new_alphas[layer][index] = random.randint(1, 6)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(1, 6)
        elif index >= 1 and index < 3:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][1] == 0 and new_alphas[layer][2]
                   == 0) or (new_alphas[layer][index]
                             == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        elif index >= 3 and index < 5:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][3] == 0 and new_alphas[layer][4]
                   == 0) or (new_alphas[layer][index]
                             == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        if index == 5:
            new_alphas[layer][index] = random.randint(-1, 5)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(-1, 5)
        if index == 6:
            new_alphas[layer][index] = random.randint(0, 5)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 5)
        return new_alphas

    def select_best(self, epoch):
        values = []
        for genotype_G in self.genotypes:
            ssim_value, psnr_value = self.validate(genotype_G)
            #logger.info(f'ssim_value: {ssim_value}, psnr_value: {psnr_value}|| @ epoch {epoch}.')
            values.append(ssim_value)
        max_index = values.index(max(values))
        return self.genotypes[max_index]


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_weight(model):
    return model.state_dict()


class LinearLrDecay(object):

    def __init__(self, optimizer, start_lr, end_lr, decay_start_step,
                 decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * \
                (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr
