
import numpy as np


import torchvision

from PIL import Image

from metassl.baselines.simsiam.augment import TrivialAugment, SmartSamplingAugment


class Cifar10AugmentPT(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None, augmentation_mode=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.trivialaugment = TrivialAugment()
        self.smartsamplingaugment = SmartSamplingAugment(max_epochs=800, current_epoch=800)
        self.augmentation_mode = augmentation_mode

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        # For ProbabilityAugment
        if self.transform is not None:
            image_a = self.transform(image=image)["image"]
            image_b = self.transform(image=image)["image"]
            image = [image_a, image_b]
        # For RandAugment, SmartAugment, TrivialAugment, and SmartSamplingAugment
        else:
            # NP > PIL
            image = Image.fromarray(image)

            # Data Augmentation
            if self.augmentation_mode == "trivialaugment":
                image_a = self.trivialaugment(image)
                image_b = self.trivialaugment(image)
            elif self.augmentation_mode == "smartsamplingaugment":
                image_a = self.smartsamplingaugment(image)
                image_b = self.smartsamplingaugment(image)
            else:
                raise NotImplementedError

            # PIL > NP
            image_a = np.asarray(image_a, dtype='float64')
            image_b = np.asarray(image_b, dtype='float64')

            # CIFAR10 normalization
            means = [0.4914, 0.4822, 0.4465],
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