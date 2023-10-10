# _*_ coding: utf-8 _*_
import numpy as np


def pairwise_accuracy(la, lb, n_samples=200000):
    n = len(la)
    assert n == len(lb)
    total = 0
    count = 0
    for _ in range(n_samples):
        i = np.random.randint(n)
        j = np.random.randint(n)
        while i == j:
            j = np.random.randint(n)
        if la[i] >= la[j] and lb[i] >= lb[j]:
            count += 1
        if la[i] < la[j] and lb[i] < lb[j]:
            count += 1
        total += 1
    return float(count) / total


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = min(max(topk), output.shape[-1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiClassAverageMeter:

    """Multi Binary Classification Tasks"""

    def __init__(self, num_classes, balanced=False, **kwargs):

        super(MultiClassAverageMeter, self).__init__()
        self.num_classes = num_classes
        self.balanced = balanced

        self.counts = []
        for k in range(self.num_classes):
            self.counts.append(np.ndarray((2, 2), dtype=np.float32))

        self.reset()

    def reset(self):
        for k in range(self.num_classes):
            self.counts[k].fill(0)

    def add(self, outputs, targets):
        outputs = outputs.data.cpu().numpy()
        targets = targets.data.cpu().numpy()

        for k in range(self.num_classes):
            output = np.argmax(outputs[:, k, :], axis=1)
            target = targets[:, k]

            x = output + 2 * target
            bincount = np.bincount(x.astype(np.int32), minlength=2 ** 2)

            self.counts[k] += bincount.reshape((2, 2))

    def value(self):
        mean = 0
        for k in range(self.num_classes):
            if self.balanced:
                value = np.mean(
                    (
                        self.counts[k]
                        / np.maximum(np.sum(self.counts[k], axis=1), 1)[:, None]
                    ).diagonal()
                )
            else:
                value = np.sum(self.counts[k].diagonal()) / np.maximum(
                    np.sum(self.counts[k]), 1
                )

            mean += value / self.num_classes * 100.0
        return mean

