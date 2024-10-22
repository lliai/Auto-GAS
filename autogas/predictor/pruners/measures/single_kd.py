# Copyright (C) 2010-2021 Alibaba Group Holding Limited.
# =============================================================================

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import measure


@measure('nst', bn=True)
def compute_nst_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    nas_score_list = []

    def nst(fm):
        fm = fm.view(fm.shape[0], fm.shape[1], -1)
        fm = F.normalize(fm, dim=2)
        return fm.sum(-1).pow(2)

    with torch.no_grad():
        output, logits = net.forward_with_features(inputs)
        nas_score_list = [
            torch.sum(nst(f)).detach().cpu().numpy() for f in output[1:-1]
        ]

        avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score


@measure('sp1', bn=True)
def compute_sp1_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    nas_score_list = []

    def sp1(fm):
        fm = fm.view(fm.shape[0], -1)
        fm = torch.mm(fm, fm.t())
        return F.normalize(fm, p=2, dim=1)

    with torch.no_grad():
        output, logits = net.forward_with_features(inputs)
        nas_score_list = [
            torch.sum(sp1(f)).detach().cpu().numpy() for f in output[1:-1]
        ]

        avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score


@measure('at', bn=True)
def compute_at_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    nas_score_list = []

    def at(fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), 2)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)
        return am

    with torch.no_grad():
        output, logits = net.forward_with_features(inputs)
        nas_score_list = [
            torch.sum(at(f)).detach().cpu().numpy() for f in output[1:-1]
        ]

        avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score


@measure('sp2', bn=True)
def compute_sp2_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    nas_score_list = []

    def sp2(fm):
        fm = fm.view(fm.size(0), -1)
        fm = torch.mm(fm, fm.t())
        norm_G_s = F.normalize(fm, p=2, dim=1)
        return norm_G_s.pow(2).sum(dim=1)

    with torch.no_grad():
        output, logits = net.forward_with_features(inputs)
        nas_score_list = [
            torch.sum(sp2(f)).detach().cpu().numpy() for f in output[1:-1]
        ]

        avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score


@measure('pdist', bn=True)
def compute_pdist_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    nas_score_list = []

    def pdist(fm, squared=False, eps=1e-12):
        fm = fm.reshape(fm.size(0), -1)
        feat_square = fm.pow(2).sum(dim=1)
        feat_prod = torch.mm(fm, fm.t())
        feat_dist = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) -
                     2 * feat_prod).clamp(min=eps)
        if not squared:
            feat_dist = feat_dist.sqrt()
        feat_dist = feat_dist.clone()
        feat_dist[range(len(fm)), range(len(fm))] = 0

        return feat_dist

    with torch.no_grad():
        output, logits = net.forward_with_features(inputs)
        nas_score_list = [
            torch.sum(pdist(f)).detach().cpu().numpy() for f in output[1:-1]
        ]

        avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score


@measure('cc', bn=True)
def compute_cc_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    nas_score_list = []

    def cc(fm):
        P_order = 2
        gamma = 0.4
        fm = F.normalize(fm, p=2, dim=-1)
        fm = fm.reshape(fm.size(0), -1)
        sim_mat = torch.matmul(fm, fm.t())
        corr_mat = torch.zeros_like(sim_mat)
        for p in range(P_order + 1):
            corr_mat += math.exp(-2 * gamma) * (2 * gamma) ** p / \
                math.factorial(p) * torch.pow(sim_mat, p)
        return corr_mat

    with torch.no_grad():
        output, logits = net.forward_with_features(inputs)
        nas_score_list = [
            torch.sum(cc(f)).detach().cpu().numpy() for f in output[1:-1]
        ]

        avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score


@measure('ickd', bn=True)
def compute_ickd_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    nas_score_list = []

    def ickd(fm):
        bsz, ch = fm.shape[0], fm.shape[1]
        fm = fm.view(bsz, ch, -1)
        emd_s = torch.bmm(fm, fm.permute(0, 2, 1))
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)

        G_diff = emd_s
        loss = (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz)
        return loss

    with torch.no_grad():
        output, logits = net.forward_with_features(inputs)
        nas_score_list = [
            torch.sum(ickd(f)).detach().cpu().numpy() for f in output[1:-1]
        ]

        avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score
