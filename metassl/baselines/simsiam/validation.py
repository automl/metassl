# https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import nn
import numpy as np
from metassl.baselines.simsiam.get_sampler import get_train_valid_sampler


class KNNValidation(object):
    def __init__(self, args, model, K=1):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        self.args = args
        self.K = K

        # Normalization
        normalize_cifar10 = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )

        normalize_cifar100 = transforms.Normalize(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762)
        )
        if args.dataset == "cifar10":
            normalize = normalize_cifar10
        elif args.dataset == "cifar100":
            normalize = normalize_cifar100
        else:
            raise NotImplementedError

        base_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        if args.dataset == "cifar10":
            train_dataset = datasets.CIFAR10(root=args.data_root,
                                             train=True,
                                             download=True,
                                             transform=base_transforms)
            train_sampler, valid_sampler = get_train_valid_sampler(args, train_dataset)
            self.train_dataloader = DataLoader(train_dataset,
                                               batch_size=args.pt_batch_size,
                                               shuffle=False,
                                               sampler=train_sampler,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               drop_last=True)
            if np.isclose(args.valid_size, 0.0):
                test_dataset = datasets.CIFAR10(root=args.data_root,
                                               train=False,
                                               download=True,
                                               transform=base_transforms)

                self.eval_dataloader = DataLoader(test_dataset,
                                                 batch_size=args.pt_batch_size,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 drop_last=True)
            else:
                valid_dataset = datasets.CIFAR10(root=args.data_root,
                                                train=True,
                                                download=True,
                                                transform=base_transforms)

                self.eval_dataloader = DataLoader(valid_dataset,
                                                   batch_size=args.pt_batch_size,
                                                   shuffle=False,
                                                   sampler=valid_sampler,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True)
        elif args.dataset == "cifar100":
            train_dataset = datasets.CIFAR100(root=args.data_root,
                                             train=True,
                                             download=True,
                                             transform=base_transforms)
            train_sampler, valid_sampler = get_train_valid_sampler(args, train_dataset)
            self.train_dataloader = DataLoader(train_dataset,
                                               batch_size=args.pt_batch_size,
                                               shuffle=False,
                                               sampler=train_sampler,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               drop_last=True)
            if np.isclose(args.valid_size, 0.0):
                test_dataset = datasets.CIFAR100(root=args.data_root,
                                                train=False,
                                                download=True,
                                                transform=base_transforms)

                self.eval_dataloader = DataLoader(test_dataset,
                                                  batch_size=args.pt_batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True,
                                                  drop_last=True)
            else:
                valid_dataset = datasets.CIFAR100(root=args.data_root,
                                                 train=True,
                                                 download=True,
                                                 transform=base_transforms)

                self.eval_dataloader = DataLoader(valid_dataset,
                                                  batch_size=args.pt_batch_size,
                                                  shuffle=False,
                                                  sampler=valid_sampler,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True,
                                                  drop_last=True)
        else:
            raise NotImplementedError

    def _topk_retrieval(self):
        """Extract features from validation split and search on train split features."""
        n_data = self.train_dataloader.dataset.data.shape[0]
        feat_dim = self.args.feat_dim

        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        train_features = torch.zeros([feat_dim, n_data], device=self.device)
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_dataloader):
                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)

                # forward
                features = self.model(inputs)
                features = nn.functional.normalize(features)
                train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = features.data.t()

            train_labels = torch.LongTensor(self.train_dataloader.dataset.targets).cuda()

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.eval_dataloader):
                targets = targets.cuda(non_blocking=True)
                batch_size = inputs.size(0)
                features = self.model(inputs.to(self.device))

                dist = torch.mm(features, train_features)
                yd, yi = dist.topk(self.K, dim=1, largest=True, sorted=True)
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

                total += targets.size(0)
                correct += retrieval.eq(targets.data).sum().item()
        top1 = correct / total

        return top1

    def eval(self):
        return self._topk_retrieval()


