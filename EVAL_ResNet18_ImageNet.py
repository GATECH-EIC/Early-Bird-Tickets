from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
from filter import *
from scipy.ndimage import filters

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--data', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('--scratch',default='', type=str,
                    help='the PATH to the pruned model')
# filter
parser.add_argument('--filter', default='none', type=str, choices=['none', 'lowpass', 'highpass'])
parser.add_argument('--sigma', default=1.0, type=float, help='gaussian filter hyper-parameter')

# sparsity
parser.add_argument('--sparsity_gt', default=0, type=float, help='sparsity controller')
# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

gpu = args.gpu_ids
gpu_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for gpu_id in gpu_ids:
   id = int(gpu_id)
   if id > 0:
       args.gpu_ids.append(id)
if len(args.gpu_ids) > 0:
   torch.cuda.set_device(args.gpu_ids[0])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.Lambda(lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                           transforms.Lambda(lambda x: my_gaussian_filter_2(x, 1/args.sigma, args.filter) if args.filter == 'highpass' else x),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.Lambda(lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                           transforms.Lambda(lambda x: my_gaussian_filter_2(x, 1/args.sigma, args.filter) if args.filter == 'highpass' else x),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.Lambda(lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                           transforms.Lambda(lambda x: my_gaussian_filter_2(x, 1/args.sigma, args.filter) if args.filter == 'highpass' else x),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.Lambda(lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                           transforms.Lambda(lambda x: my_gaussian_filter_2(x, 1/args.sigma, args.filter) if args.filter == 'highpass' else x),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=16, pin_memory=True)

if args.dataset == 'imagenet':
    model = models.__dict__[args.arch](pretrained=False)
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.scratch:
    checkpoint = torch.load(args.scratch)
    if args.dataset == 'imagenet':
        model = models.__dict__[args.arch](pretrained=False, cfg=checkpoint['cfg'])
        model_ref = models.__dict__[args.arch](pretrained=False, cfg=checkpoint['cfg'])
        model_ref.load_state_dict(checkpoint['state_dict'])
    else:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
        model_ref = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
        model_ref.load_state_dict(checkpoint['state_dict'])
    for m0, m1 in zip(model.modules(), model_ref.modules()):
        if isinstance(m0, models.channel_selection):
            m0.indexes.data = m1.indexes.data.clone()

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def save_checkpoint(state, is_best, epoch, filepath):
    if epoch == 'init':
        filepath = os.path.join(filepath, 'init.pth.tar')
        torch.save(state, filepath)
    elif 'EB' in str(epoch):
        filepath = os.path.join(filepath, epoch+'.pth.tar')
        torch.save(state, filepath)
    else:
        filename = os.path.join(filepath, 'ckpt'+str(epoch)+'.pth.tar')
        torch.save(state, filename)
        # filename = os.path.join(filepath, 'ckpt.pth.tar')
        # torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))

if len(args.gpu_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              # .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    print('nooooo!')

history_score = np.zeros((args.epochs, 3))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(epoch):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        # pred = output.data.max(1, keepdim=True)[1]
        # train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        train_acc += prec1.item()
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = np.round(train_acc / len(train_loader), 2)

def test():
    model.eval()
    test_loss = 0
    test_acc = 0
    test_acc_5 = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()
        test_acc_5 += prec5.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy (Top-1): {}/{} ({:.2f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader)))
    print('\nTest set: Average loss: {:.4f}, Accuracy (Top-5): {}/{} ({:.2f}%)\n'.format(
        test_loss, test_acc_5, len(test_loader), test_acc_5 / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2), np.round(test_acc_5 / len(test_loader), 2)

prec1, prec5 = test()
print('Top-1: ', prec1)
print('Top-5: ', prec5)
