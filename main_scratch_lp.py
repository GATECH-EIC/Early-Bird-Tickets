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
from compute_flops import print_model_param_flops
from torch.optim import SGD
from qtorch.quant import *
from qtorch.optim import OptimLP
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.auto_low import sequential_lower

num_types = ["weight", "activate", "grad", "error", "momentum"]


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--data', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
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
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
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
parser.add_argument('--scratch',default='', type=str,
                    help='the PATH to the pruned model')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

# swa
parser.add_argument('--swa', default=False, help='SWALP start epoch')
parser.add_argument('--swa_start', type=int, default=160, metavar='N',
                    help='SWALP start epoch')
# parser.add_argument('--swa_lr', type=float, default=0.01, metavar='LR',
                    # help='SWALP learning rate (default: 0.01)')

# quantized parameters
for num in num_types:
    parser.add_argument('--wl-{}'.format(num), type=int, default=-1, metavar='N',
                        help='word length in bits for {}; -1 if full precision.'.format(num))
parser.add_argument('--rounding'.format(num), type=str, default='stochastic', metavar='S',
                    choices=["stochastic","nearest"],
                    help='rounding method for {}, stochastic or nearest'.format(num))

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

# prepare quantization functions
# using block floating point, allocating shared exponent along the first dimension
number_dict = dict()
for num in num_types:
    num_wl = getattr(args, "wl_{}".format(num))
    number_dict[num] = BlockFloatingPoint(wl=num_wl, dim=0)
    print("{:10}: {}".format(num, number_dict[num]))
quant_dict = dict()
for num in ["weight", "momentum", "grad"]:
    quant_dict[num] = quantizer(forward_number=number_dict[num],
                                forward_rounding=args.rounding)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
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
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'imagenet':
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=16, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=16, pin_memory=True)

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
# automatically insert quantization modules
model = sequential_lower(model, layer_types=["conv", "linear"],
                         forward_number=number_dict["activate"], backward_number=number_dict["error"],
                         forward_rounding=args.rounding, backward_rounding=args.rounding)
# removing the final quantization module
if args.arch == 'vgg':
    model.classifier = model.classifier[0] 
elif args.arch == 'resnet':
    model.fc = model.fc[0]

if args.dataset == 'imagenet':
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

def load_checkpoint(model, checkpoint_path):
   model_ckpt = torch.load(checkpoint_path, map_location='cpu')
   pretrained_dict = model_ckpt['state_dict']
   model_dict = model.state_dict()
   new_dict = {}
   for k in model_dict.keys():
       pre_k = 'module.' + k
       new_dict[k] = pretrained_dict[pre_k]
   print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
   model.load_state_dict(new_dict)
   print('load checkpoint!')
   return model

if args.scratch:
    try:
        checkpoint = torch.load(args.scratch)
        # print(checkpoint['state_dict'].keys())
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
        # automatically insert quantization modules
        model = sequential_lower(model, layer_types=["conv", "linear"],
                             forward_number=number_dict["activate"], backward_number=number_dict["error"],
                             forward_rounding=args.rounding, backward_rounding=args.rounding)
        # removing the final quantization module
        if args.arch == 'vgg':
            model.classifier = model.classifier[0] 
        elif args.arch == 'resnet':
            model.fc = model.fc[0]

        if args.dataset == 'imagenet':
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        model.load_state_dict(checkpoint['state_dict'])
        model._initialize_weights()
    except:
        checkpoint = torch.load(args.scratch, map_location='cpu')
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
        # automatically insert quantization modules
        model = sequential_lower(model, layer_types=["conv", "linear"],
                             forward_number=number_dict["activate"], backward_number=number_dict["error"],
                             forward_rounding=args.rounding, backward_rounding=args.rounding)
        # removing the final quantization module
        if args.arch == 'vgg':
            model.classifier = model.classifier[0] 
        elif args.arch == 'resnet':
            model.fc = model.fc[0]
        model = load_checkpoint(model, args.scratch)
        model._initialize_weights()
        # model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

# Build SWALP model
if args.swa:
    swa_model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    swa_n = 0
    swa_model.cuda()

if args.cuda:
    model.cuda()

optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# insert quantizations into the optimization loops
optimizer = OptimLP(optimizer,
                    weight_quant=quant_dict["weight"],
                    grad_quant=quant_dict["grad"],
                    momentum_quant=quant_dict["momentum"])

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

history_score = np.zeros((args.epochs, 4))

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

def test(model):
    model.eval()
    test_loss = 0
    test_acc = 0
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

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)

def save_checkpoint(state, is_best, filepath, is_swa):
    if is_swa:
        torch.save(state, os.path.join(filepath, 'swa.pth.tar'))
    else:
        torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec1 = test(model)
    history_score[epoch][2] = prec1
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save, is_swa=False)

    if args.swa and epoch >= args.swa_start:
        moving_average(swa_model, model, 1.0 / (swa_n + 1))
        swa_n += 1
        bn_update(train_loader, swa_model)
        prec1 = test(swa_model)
        history_score[epoch][3] = prec1
        save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'prec1': prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save, is_swa=True)
    else:
        history_score[epoch][3] = 0.0

print("Best accuracy: "+str(best_prec1))
history_score[-1][0] = best_prec1
if args.swa:
    history_score[-1][1] = prec1
    print('SWA accuracy: '+str(prec1))
np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
