import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from collections import Counter
import random
from functools import reduce


# from https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/6
def get_module_by_name(module, access_string):
    """
    Get a module by its name.

    Args:
        module: The module to search in.
        access_string: The name of the module to get.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


TEST_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]
)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_imagenet_loader(path, batch_size, num_workers, shuffle=False, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    dataset = torchvision.datasets.ImageNet(
        path, split="val", transform=TEST_TRANSFORMS
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dataloader


def get_imagenet_folder_loader(
    path, batch_size, num_workers, shuffle=False, resize_crop=True, seed=0
):
    g = torch.Generator()
    g.manual_seed(seed)
    transforms = TEST_TRANSFORMS if resize_crop else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.ImageFolder(path, transform=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dataloader


# Source: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L420
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

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


def seed_everything(seed):
    torch.manual_seed(seed)

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", "y")


def get_key_for_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None


def tensor_map(func, iterable):
    return torch.stack([func(x) for x in iterable])
