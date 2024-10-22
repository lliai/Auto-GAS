import numpy as np

from . import measure


@measure('grad_hamming', bn=True, mode='param')
def compute_grad_hamming(net, inputs, targets, mode, loss_fn, split_data=1):

    def counting_backward_hook(module, grad_input, grad_output):
        try:
            if isinstance(grad_input, tuple):
                grad_input = grad_input[0]
            grad_input = grad_input.view(grad_input.size(0), -1)
            x = (grad_input > 0).float()
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()
        except Exception as e:
            print(e)

    def hooklogdet(K):
        s, ld = np.linalg.slogdet(K)
        return ld

    def hookeig(K):
        s, ld = np.linalg.eig(K)
        return np.mean(s)

    for name, module in net.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_backward_hook(counting_backward_hook)

    s = []
    N = inputs.shape[0]
    for sp in range(split_data):
        net.zero_grad()
        net.K = np.zeros((N // split_data, N // split_data))
        #net.N = 0

        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs = net(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()
        # if net.N != 0:
        #    s.append(hooklogdet(net.K / net.N))
        s.append(hooklogdet(net.K))
        # s.append(hookeig(net.K))
    # grad_hamming = np.mean(s)
    grad_hamming = np.prod(s)

    return grad_hamming
