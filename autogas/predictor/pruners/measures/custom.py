import torch

from . import measure


def get_score(net, x, target, device, split_data):
    result_list = []

    def forward_hook(module, data_input, data_output):

        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        n_positive_eigvals = sum((torch.real(values) > 0).float())
        result = torch.min(torch.real(n_positive_eigvals * values))

        result_list.append(result)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        # y = net(x[st:en])
        y = net(x[st:en])
    results = torch.tensor(result_list)

    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    # v = torch.log(torch.abs(v))
    result_list.clear()
    return v.item()


@measure('custom', bn=True)
def compute_custom_score(net,
                         inputs,
                         targets,
                         split_data=1,
                         loss_fn=None,
                         u1=0,
                         u2=0,
                         b1=0):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    custom = get_score(net, inputs, targets, device, split_data=split_data)
    print(f'custom score:{custom}')
    return custom
