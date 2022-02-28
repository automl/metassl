import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler


def get_train_valid_sampler(args, trainset):
    # Use always the same validation set!
    # generator = torch.Generator()
    # generator.manual_seed(0)

    dataset_percentage_usage = args.dataset_percentage_usage
    num_train = int(len(trainset) / 100 * dataset_percentage_usage)
    indices = list(range(num_train))
    split = int(np.floor(args.valid_size * num_train))

    if args.distributed:
        np.random.seed(args.seed)
        np.random.shuffle(indices)
    else:
        random_number_generator = np.random.default_rng(0)
        random_number_generator.shuffle(indices)

    if np.isclose(args.valid_size, 0.0):
        train_idx, valid_idx = indices, indices
    else:
        train_idx, valid_idx = indices[split:], indices[:split]

    #  print(valid_idx[:10])
    valid_sampler = SubsetRandomSampler(valid_idx)  # add generator?

    if args.distributed:
        # Not tested yet
        raise ValueError("TODO: Test distributed!")
        # train_sampler = torch.utils.data.distributed.DistributedSampler(torch.tensor(train_idx))
        # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        if not np.isclose(args.valid_size, 0.0):
            train_sampler = SubsetRandomSampler(train_idx)
        else:
            train_sampler = None

    return train_sampler, valid_sampler