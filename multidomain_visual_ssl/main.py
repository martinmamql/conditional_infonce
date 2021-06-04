# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.distributed as dist
import utils

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.2, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
# for barlow twins
parser.add_argument('--lambd', default=3.9e-3, type=float, metavar='L',
                    help='weight on off-diagonal terms')
# for simclr
parser.add_argument('--temperature', default=0.1, type=float, metavar='T',
                    help='temperature in infonce loss')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--scale-loss', default=1 / 32, type=float,
                    metavar='S', help='scale the loss')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--dataset', default='imagenet', type=str, metavar='D',
                    help='dataset: cifar10 or imagenet or tiny_imagenet or stl10')
parser.add_argument('--method', default='barlow_twins', type=str, metavar='M',
                    help='method: barlow_twins or simclr or hsic')
parser.add_argument('--dist-address', default='58472', type=str, metavar='N',
                    help='address for distributed training')


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = 'tcp://{}:'.format(host_name) + args.dist_address
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:' + args.dist_address
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    save_name = args.method + '_' + args.dataset + '_bsz_' + str(args.batch_size) + \
                '_lr_' + str(args.learning_rate) + '_featdim_' + args.projector.split('-')[-1] + \
                '_temp_' + str(args.temperature) + '_epoch_' + str(args.epochs)
    save_name_stats = save_name + '_stats.txt'
    save_name_ckpt = save_name + '_checkpoint.pth'
    save_name_final = save_name + '_resnet50.pth'

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / save_name_stats, 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = Model(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=exclude_bias_and_norm,
                     lars_adaptation_filter=exclude_bias_and_norm)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / save_name_ckpt).is_file():
        ckpt = torch.load(args.checkpoint_dir / save_name_ckpt,
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0
    
    if args.dataset == 'imagenet':
        dataset = torchvision.datasets.ImageFolder(args.data / 'train', utils.Transform())
    elif args.dataset == 'tiny_imagenet':
        dataset = torchvision.datasets.ImageFolder(args.data / 'tiny-imagenet-200/train',\
                    utils.TinyImageNetPairTransform(train_transform=True, pair_transform=True))
    elif args.dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=args.data, train=True,\
                    transform=utils.CifarPairTransform(train_transform=True, pair_transform=True),\
                    download=True)
    elif args.dataset == 'stl10':
        dataset = torchvision.datasets.STL10(root=args.data, split="train+unlabeled", \
                    transform=utils.StlPairTransform(train_transform=True, pair_transform=True),\
                    download=True)
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2, temperature=args.temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                torch.distributed.reduce(loss.div_(args.world_size), 0)
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step, learning_rate=lr,
                                 loss=loss.item(),
                                 time=int(time.time() - start_time),
                                 dataset=args.dataset,
                                 method=args.method,
                                 batch_size=args.batch_size,
                                 proj_feat_dim=args.projector.split('-')[-1])
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / save_name_ckpt)
    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / save_name_final)


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.learning_rate * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def all_gather(tensor, expand_dim=0, num_replicas=None):
    """Gathers a tensor from other replicas, concat on expand_dim and return."""
    num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
    other_replica_tensors = [torch.zeros_like(tensor) for _ in range(num_replicas)]
    dist.all_gather(other_replica_tensors, tensor)
    return torch.cat([o.unsqueeze(expand_dim) for o in other_replica_tensors], expand_dim)

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.dataset == 'cifar10' or args.dataset == 'tiny_imagenet' or args.dataset == 'stl10':
            resnet50 = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone = []
            for name, module in resnet50.named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if args.dataset == 'cifar10':
                    if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                        self.backbone.append(module)
                elif args.dataset == 'tiny_imagenet' or args.dataset == 'stl10':
                    if not isinstance(module, nn.Linear):
                        self.backbone.append(module)
            self.backbone = nn.Sequential(*self.backbone)
        elif args.dataset == 'imagenet':
            self.backbone = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity()

        self.dataset = args.dataset

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        
        self.method = args.method
    # default: forward with Barlow Twins
    def forward(self, y1, y2, temperature=0.1, num_replicas=None):
        if self.dataset == 'cifar10' or self.dataset == 'tiny_imagenet' or self.dataset == 'stl10':
            embedding1 = self.projector(torch.flatten(self.backbone(y1), start_dim=1))
            embedding2 = self.projector(torch.flatten(self.backbone(y2), start_dim=1))
        elif self.dataset == 'imagenet':
            embedding1 = self.projector(self.backbone(y1))
            embedding2 = self.projector(self.backbone(y2))
            
        if self.method == 'barlow_twins' or self.method == 'hsic':
            # empirical cross-correlation matrix
            c = self.bn(embedding1).T @ self.bn(embedding2)

            # sum the cross-correlation matrix between all gpus
            c.div_(self.args.batch_size)
            torch.distributed.all_reduce(c)

            # use --scale-loss to multiply the loss by a constant factor
            # see the Issues section of the readme
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.args.scale_loss)
            if self.method == 'barlow_twins':
                off_diag = off_diagonal(c).pow_(2).sum().mul(self.args.scale_loss)
            elif self.method == 'hsic':
                off_diag = off_diagonal(c).add_(1).pow_(2).sum().mul(self.args.scale_loss)
            loss = on_diag + self.args.lambd * off_diag
            return loss
        elif self.method == 'simclr':
            """NT-XENT Loss from SimCLR

            :param embedding1: embedding of augmentation1
            :param embedding2: embedding of augmentation2
            :param temperature: nce normalization temp
            :param num_replicas: number of compute devices
            :returns: scalar loss
            :rtype: float32
            """
            batch_size = embedding1.shape[0]
            feature_size = embedding1.shape[-1]
            num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
            LARGE_NUM = 1e9

            # normalize both embeddings
            embedding1 = F.normalize(embedding1, dim=-1)
            embedding2 = F.normalize(embedding2, dim=-1)

            if num_replicas > 1 and self.training:
                # First grab the tensor from all other embeddings
                embedding1_full = all_gather(embedding1, num_replicas=num_replicas)
                embedding2_full = all_gather(embedding2, num_replicas=num_replicas)

                # fold the tensor in to create [B, F]
                embedding1_full = embedding1_full.reshape(-1, feature_size)
                embedding2_full = embedding2_full.reshape(-1, feature_size)

                # Create pseudo-labels using the current replica id & ont-hotting
                replica_id = dist.get_rank()
                labels = torch.arange(batch_size, device=embedding1.device) + replica_id * batch_size
                labels = labels.type(torch.int64)
                full_batch_size = embedding1_full.shape[0]
                masks = F.one_hot(labels, full_batch_size).to(embedding1_full.device)
                labels = F.one_hot(labels, full_batch_size * 2).to(embedding1_full.device)
            else:  # no replicas or we are in test mode; test set is same size on all replicas for now
                embedding1_full = embedding1
                embedding2_full = embedding2
                masks = F.one_hot(torch.arange(batch_size), batch_size).to(embedding1.device)
                labels = F.one_hot(torch.arange(batch_size), batch_size * 2).to(embedding1.device)

            # Matmul-to-mask
            logits_aa = torch.matmul(embedding1, embedding1_full.T) / temperature
            logits_aa = logits_aa - masks * LARGE_NUM
            logits_bb = torch.matmul(embedding2, embedding2_full.T) / temperature
            logits_bb = logits_bb - masks * LARGE_NUM
            logits_ab = torch.matmul(embedding1, embedding2_full.T) / temperature
            logits_ba = torch.matmul(embedding2, embedding1_full.T) / temperature

            # Use our standard cross-entropy loss which uses log-softmax internally.
            # Concat on the feature dimension to provide all features for standard softmax-xent
            loss_a = F.cross_entropy(input=torch.cat([logits_ab, logits_aa], 1),
                                     target=torch.argmax(labels, -1),
                                     reduction="none")
            loss_b = F.cross_entropy(input=torch.cat([logits_ba, logits_bb], 1),
                                     target=torch.argmax(labels, -1),
                                     reduction="none")
            loss = loss_a + loss_b
            return torch.mean(loss)

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1

if __name__ == '__main__':
    main()
