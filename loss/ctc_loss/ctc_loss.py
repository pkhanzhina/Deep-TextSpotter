import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
from torch.autograd import Variable
import itertools


def compute_alpha(out, label, bl):
    S = len(label) * 2 + 1
    T = out.shape[1]

    a = np.zeros((S, T))
    a[0, 0] = out[bl, 0]
    a[1, 0] = out[label[0], 0]
    c = np.sum(a[:, 0])
    if c > 0:
        a[:, 0] /= c

    for t in range(1, T):
        start = max(0, S - 2 * (T - t))
        end = min(2 * t + 2, S)
        for s in range(start, end):
            i = max(0, (s - 1) // 2)
            if s % 2 == 0:
                if s == 0:
                    a[s, t] = a[s, t - 1] * out[bl, t]
                else:
                    a[s, t] = (a[s, t - 1] + a[s - 1, t - 1]) * out[bl, t]
            elif s == 1 or label[i] == label[i - 1]:
                a[s, t] = (a[s, t - 1] + a[s - 1, t - 1]) * out[label[i], t]
            else:
                a[s, t] = (a[s, t - 1] + a[s - 1, t - 1] + a[s - 2, t - 1]) * out[label[i], t]
        c = np.sum(a[start:end, t])
        if c > 0:
            a[start:end, t] /= c
    return a


def ctc_loss(args):
    out, label, length, bl = args
    softmax_grad = np.zeros_like(out)
    if out.shape[1] == 0 or length == 0:
        return 0.0, softmax_grad

    L = 2 * len(label) + 1
    T = length
    out_unpadded = out[:, :T]

    a = compute_alpha(out_unpadded, label, bl)
    flipped_b = compute_alpha(np.fliplr(out_unpadded), label[::-1], bl)
    b = np.flipud(np.fliplr(flipped_b))

    ab = a * b
    lab = np.zeros(out_unpadded.shape)
    for s in range(L):
        if s % 2 == 0:
            lab[bl, :] += ab[s, :]
            ab[s, :] = ab[s, :] / out_unpadded[bl, :]
        else:
            i = max(0, (s - 1) // 2)
            lab[label[i], :] += ab[s, :]
            ab[s, :] = ab[s, :] / out_unpadded[label[i], :]
    lh = np.sum(ab, axis=0)
    loss = np.log(lh)
    loss[np.isnan(loss)] = 0
    loss = -np.sum(loss)
    softmax_grad[:, :T] = out_unpadded - lab / (out_unpadded * lh)
    # indices = np.isnan(softmax_grad)
    softmax_grad[np.isnan(softmax_grad)] = 0
    return loss, softmax_grad


class CTCLoss(nn.Module):
    def __init__(self, device, workers=4):
        super().__init__()
        self.pool = mp.Pool(workers)
        self.device = device

    def forward(self, params_list, seq_list, lengths, bl=0):
        dynamic_params = list(zip(params_list, seq_list, lengths, itertools.repeat(bl)))
        # results = [ctc_loss(params) for params in dynamic_params]
        results = self.pool.map(ctc_loss, dynamic_params)
        losses, self.grad = zip(*results)
        return np.mean(losses)

    def backward(self):
        return Variable(torch.Tensor(np.asarray(self.grad))).to(self.device), None


if __name__ == '__main__':
    pass