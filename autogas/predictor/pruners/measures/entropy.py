# https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/nas/scores/compute_entropy.py MAE_DET

# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import numpy as np
import torch
from torch import nn

from . import measure


@measure('entropy', bn=True)
def compute_entropy_score(
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

    device = inputs.device
    dtype = torch.half if fp16 else torch.float32

    output_list = []

    def hook_fw_fn(module, input, output):
        output_list.append(output.detach())

    for name, module in net.named_modules():
        if 'conv1' in name:
            continue
        if isinstance(module, (nn.Conv2d)):
            module.register_forward_hook(hook_fw_fn)

    with torch.no_grad():
        for _ in range(repeat):
            # network_weight_gaussian_init(net)
            input = torch.randn(
                size=list(inputs.shape), device=device, dtype=dtype)

            # outputs, logits = net.forward_with_features(input)
            _ = net(input)

            for i, output in enumerate(output_list):
                nas_score = torch.log(output.std(
                ))  # + torch.log(torch.var(output / (output.std() + 1e-9)))
                nas_score_list.append(float(nas_score))

    avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score
