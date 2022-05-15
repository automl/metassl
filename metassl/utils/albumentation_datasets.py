import numpy as np
import torchvision
from PIL import Image

from metassl.utils.rand_augment import RandAugment, SmartAugment, TrivialAugment


class Cifar10AlbumentationsPT(torchvision.datasets.CIFAR10):
    # For ProbabilityAugment
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            image_a = self.transform(image=image)["image"]
            image_b = self.transform(image=image)["image"]
            image = [image_a, image_b]
        return image, label


class Cifar10AugmentationPT(torchvision.datasets.CIFAR10):
    # For RandAugment, SmartAugment, TrivialAugment, and SmartSamplingAugment
    def __init__(
        self,
        root="~/data/cifar10",
        train=True,
        download=True,
        transform=None,
        data_augmentation_mode="rand_augment",
        neps_hyperparameters=None,
    ):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.data_augmentation_mode = data_augmentation_mode
        self.neps_hyperparameters = neps_hyperparameters
        if data_augmentation_mode == "rand_augment":
            self.rand_augment = RandAugment(neps_hyperparameters=neps_hyperparameters)
        elif data_augmentation_mode == "smart_augment":
            self.smart_augment = SmartAugment(neps_hyperparameters=neps_hyperparameters)
        elif data_augmentation_mode == "trivial_augment":
            self.trivial_augment = TrivialAugment()
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        # NP > PIL
        image = Image.fromarray(image)

        # Data Augmentation
        if self.data_augmentation_mode == "rand_augment":
            image_a = self.rand_augment(image)
            image_b = self.rand_augment(image)
        elif self.data_augmentation_mode == "trivial_augment":
            image_a = self.trivial_augment(image)
            image_b = self.trivial_augment(image)
        elif self.data_augmentation_mode == "smart_augment":
            image_a = self.smart_augment(image)
            image_b = self.smart_augment(image)
        else:
            raise NotImplementedError

        # PIL > NP
        image_a = np.asarray(image_a, dtype="float64")
        image_b = np.asarray(image_b, dtype="float64")

        # CIFAR10 normalization
        means = ([0.4914, 0.4822, 0.4465],)
        stds = [0.2023, 0.1994, 0.2010]
        image_a /= np.float(255)
        image_b /= np.float(255)
        image_a = (image_a - means) / stds
        image_b = (image_b - means) / stds

        # TRANSPOSE
        image_a = np.transpose(image_a, axes=[2, 0, 1])
        image_b = np.transpose(image_b, axes=[2, 0, 1])

        image = [image_a, image_b]

        return image, label


class Cifar10AlbumentationsFT(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label
