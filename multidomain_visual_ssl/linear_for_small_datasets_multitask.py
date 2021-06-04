import argparse
from pathlib import Path
import distutils
from distutils import util

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torchvision

import utils

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = torchvision.models.resnet50(zero_init_residual=True)
        self.f1_c10, self.f1_stl, self.f1_tim, self.f2 = [], [], [], []
        for name, module in resnet50.named_children():
            if name == 'conv1':
                self.f1_c10.append(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
                self.f1_stl.append(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
                self.f1_tim.append(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
            elif name == 'bn1':
                self.f1_c10.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.f1_stl.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.f1_tim.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            elif name == 'relu':
                self.f1_c10.append(nn.ReLU(inplace=True))
                self.f1_stl.append(nn.ReLU(inplace=True))
                self.f1_tim.append(nn.ReLU(inplace=True))
            elif name == 'maxpool':
                self.f1_stl.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))
                self.f1_tim.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))
            elif not name == 'fc':
                self.f2.append(module)
        self.f1_c10 = nn.Sequential(*self.f1_c10)
        self.f1_stl = nn.Sequential(*self.f1_stl)
        self.f1_tim = nn.Sequential(*self.f1_tim)
        self.f2 = nn.Sequential(*self.f2)

    def forward(self, y1, y2, conditional, weak, dataset, num_replicas=None):
        # non weak conditional case
        embedding1, embedding2 = None, None
        #print(dataset)
        if (not weak) and conditional:
            if dataset == 'cifar':
                embedding1 = torch.flatten(self.f2(self.f1_c10(y1)), start_dim=1)
                embedding2 = torch.flatten(self.f2(self.f1_c10(y2)), start_dim=1)
            elif dataset == 'stl10':
                embedding1 = torch.flatten(self.f2(self.f1_stl(y1)), start_dim=1)
                embedding2 = torch.flatten(self.f2(self.f1_stl(y2)), start_dim=1)
            elif dataset == 'tiny_imagenet':
                embedding1 = torch.flatten(self.f2(self.f1_tim(y1)), start_dim=1)
                embedding2 = torch.flatten(self.f2(self.f1_tim(y2)), start_dim=1)
        else:
            if dataset == 'cifar':
                embedding1 = torch.flatten(self.f2(self.f1_c10(y1)), start_dim=1)
                embedding2 = torch.flatten(self.f2(self.f1_c10(y2)), start_dim=1)
            elif dataset == 'stl10':
                embedding1 = torch.flatten(self.f2(self.f1_stl(y1)), start_dim=1)
                embedding2 = torch.flatten(self.f2(self.f1_stl(y2)), start_dim=1)
            elif dataset == 'tiny_imagenet':
                embedding1 = torch.flatten(self.f2(self.f1_tim(y1)), start_dim=1)
                embedding2 = torch.flatten(self.f2(self.f1_tim(y2)), start_dim=1)
        return embedding1, embedding2


class Net(nn.Module):
    def __init__(self, num_class, pretrained_path, dataset, conditional, weak):
        super(Net, self).__init__()

        # encoder
        #class ARGS():
        #    def __init__(self):
        #        self.dataset = dataset
        #        self.method = None
        #        self.projector = '1'
        #        self.conditional = conditional
        #        self.weak = weak

        #args = ARGS()
        model = Model()
        self.f = model

        # original saved file with DataParallel
        state_dict = torch.load(pretrained_path, map_location='cpu')
        #if dataset == "cifar":
        #    self.f1 = model.f1_c10
        #    # Filter state dict that match the dataset
        #    new_state_dict = {k:v for k, v in state_dict.items() if "c10" in k}
        #elif dataset == "stl10":
        #    self.f1 = model.f1_stl
        #    new_state_dict = {k:v for k, v in state_dict.items() if "stl" in k}
        #elif dataset == "tiny_imagenet":
        #    self.f1 = model.f1_tim
        #    new_state_dict = {k:v for k, v in state_dict.items() if "tim" in k}
        #else:
        #    assert False
        #print(new_state_dict)
        #self.f1.load_state_dict(new_state_dict)
        #print(self.f1.state_dict().keys())
        #assert False
        #self.f2 = model.f2

        # create new OrderedDict that does not contain `module.` prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "projector" not in k: # do not include projector layers
                name = k[7:] # remove `module.` prefix
                new_state_dict[name] = v
        self.f.load_state_dict(new_state_dict)

        # A non-optimal way to remove the last projection layer
        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        #model_dict = self.f.state_dict()
        #del model_dict['projector.0.weight']
        #model_dict.update(new_state_dict)
        #print(model_dict.keys())
        #del self.f.state_dict()['projector.0.weight']
        #self.f.load_state_dict(model_dict)
        #print(self.f.state_dict().keys())
        #assert False

        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x, conditional, weak, dataset):

        # A not optimal way to fix the issue
        x, _ = self.f(x, x, conditional, weak, dataset)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

# train or test for one epoch
def train_val(net, data_loader, train_optimizer, conditional, weak, dataset):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data, conditional, weak, dataset)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}% model: {}'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100,
                                             model_path.split('/')[-1]))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--data', type=Path, metavar='DIR', help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset: cifar10 or tiny_imagenet or stl10')
    parser.add_argument('--model_path', type=str, default='checkpoint/resnet50.pth',
                        help='The base string of the pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=200, help='Number of sweeps over the dataset to train')


    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
   
    # Figure out conditional setting from model path (weak conditional / conditional / unconditional)
    model_path_list = model_path.split("_")
    conditional = True if model_path_list[model_path_list.index("cond") + 1] == "True" else False
    weak = True if model_path_list[model_path_list.index("weak") + 1] == "True" else False
    # minor fix for cifar10 name
    dataset = "cifar" if args.dataset == "cifar10" else args.dataset

    if dataset == 'cifar':
        train_data = CIFAR10(root=args.data, train=True,\
            transform=utils.CifarPairTransform(train_transform = True, pair_transform=False), download=True)
        test_data = CIFAR10(root=args.data, train=False,\
            transform=utils.CifarPairTransform(train_transform = False, pair_transform=False), download=True)
    elif dataset == 'stl10':
        train_data =  torchvision.datasets.STL10(root=args.data, split="train", \
            transform=utils.StlPairTransform(train_transform = True, pair_transform=False), download=True)
        test_data =  torchvision.datasets.STL10(root=args.data, split="test", \
            transform=utils.StlPairTransform(train_transform = False, pair_transform=False), download=True)
    elif dataset == 'tiny_imagenet':
        train_data = torchvision.datasets.ImageFolder(args.data / 'tiny-imagenet-200/train', \
                            utils.TinyImageNetPairTransform(train_transform=True, pair_transform=False))
        test_data = torchvision.datasets.ImageFolder(args.data / 'tiny-imagenet-200/val', \
                            utils.TinyImageNetPairTransform(train_transform = False, pair_transform=False))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = Net(num_class=len(train_data.classes), pretrained_path=model_path, \
                dataset=dataset, conditional=conditional, weak=weak).cuda()
    for param in model.f.parameters():
        param.requires_grad = False

    #if dataset == 'cifar':
    #    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(), ))
    #elif dataset == 'tiny_imagenet' or dataset == 'stl10':
    #    flops, params = profile(model, inputs=(torch.randn(1, 3, 64, 64).cuda(), ))
    #flops, params = clever_format([flops, params])
    #print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer, conditional, weak, dataset)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, conditional, weak, dataset)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

    # TODO save the results
