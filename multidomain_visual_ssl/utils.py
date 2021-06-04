from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random

# for cifar10 (32x32)
class CifarPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        self.pair_transform = pair_transform
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

# for tiny imagenet (64x64)
class TinyImageNetPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                                saturation=0.4, hue=0.1)], 
                        p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(
                        64,
                        scale=(0.2, 1.0),
                        ratio=(0.75, (4 / 3)),
                        interpolation=Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ])
        self.pair_transform = pair_transform
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

# for stl10 (96x96)
class StlPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                                saturation=0.4, hue=0.1)], 
                        p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(
                        64,
                        scale=(0.2, 1.0),
                        ratio=(0.75, (4 / 3)),
                        interpolation=Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(70, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
                ])
        self.pair_transform = pair_transform
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

# for imagenet
class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
