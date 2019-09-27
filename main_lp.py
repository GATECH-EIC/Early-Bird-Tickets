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
from optim.sgd import SGD

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
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

# quantized parameters
parser.add_argument('--bits_A', default=8, type=int, help='input quantization bits')
parser.add_argument('--bits_W', default=8, type=int, help='weight quantization bits')
parser.add_argument('--bits_G', default=8, type=int, help='gradient quantization bits')
parser.add_argument('--bits_E', default=8, type=int, help='error quantization bits')
parser.add_argument('--bits_R', default=16, type=int, help='rand number quantization bits')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

class EarlyBird():
    def __init__(self, percent, epoch_keep=5):
        self.percent = percent
        self.epoch_keep = epoch_keep
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]

    def pruning(self, model, percent):
        total = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * percent)
        thre = y[thre_index]
        # print('Pruning threshold: {}'.format(thre))

        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        # print('Pre-processing Successful!')
        return mask

    def put(self, mask):
        if len(self.masks) < self.epoch_keep:
            self.masks.append(mask)
        else:
            self.masks.pop(0)
            self.masks.append(mask)

    def cal_dist(self):
        if len(self.masks) == self.epoch_keep:
            for i in range(len(self.masks)-1):
                mask_i = self.masks[-1]
                mask_j = self.masks[i]
                self.dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
            return True
        else:
            return False

    def early_bird_emerge(self, model):
        mask = self.pruning(model, self.percent)
        self.put(mask)
        flag = self.cal_dist()
        if flag == True:
            print(self.dists)
            for i in range(len(self.dists)):
                if self.dists[i] > 0.1:
                    return False
            return True
        else:
            return False

# quantize setting
def quant_summary(bits):
    if bits == -1:
        return "float"
    else:
        return "wage-{}".format(bits)

w_summary = quant_summary(args.bits_W)
a_summary = quant_summary(args.bits_A)
g_summary = quant_summary(args.bits_G)
e_summary = quant_summary(args.bits_E)
print('W:{}, A:{}, G:{}, E:{}'.format(w_summary, a_summary, g_summary, e_summary))

act_quantizer = lambda x: models.QA(x, args.bits_A)
weight_quantizer = lambda x, scale: models.QW(x, args.bits_W, scale)
grad_quantizer = lambda x, lr: models.QG(x, args.bits_G, args.bits_R, lr)
grad_clip = lambda x: models.C(x, args.bits_W)
if args.bits_W == -1: weight_quantizer = lambda x, scale: x
if args.bits_G == -1: grad_clip = lambda x: x; grad_quantizer = lambda x, lr: x * lr

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
else:
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

if args.arch.endswith('lp'):
    model = models.__dict__[args.arch](bits_A=args.bits_A, bits_E=args.bits_E, bits_W=args.bits_W, dataset=args.dataset, depth=args.depth)
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def save_checkpoint(state, is_best, epoch, filepath):
    if epoch == 'init':
        filepath = os.path.join(filepath, 'init.pth.tar')
        torch.save(state, filepath)
    else:
        filename = os.path.join(filepath, 'ckpt'+str(epoch)+'.pth.tar')
        torch.save(state, filename)
        # filename = os.path.join(filepath, 'ckpt.pth.tar')
        # torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))

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
else:
    save_checkpoint({'state_dict': model.state_dict()}, False, epoch='init', filepath=args.save)

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
        # quantize weight
        for name, param in model.named_parameters():
            param.data = weight_quantizer(param.data, model.weight_scale[name]).cuda()
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
        optimizer.step(grad_quantizer, grad_clip)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = np.round(train_acc / len(train_loader), 2)

def test():
    model.eval()
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        # quantize weight
        for name, param in model.named_parameters():
            param.data = weight_quantizer(param.data, model.weight_scale[name]).cuda()
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)

best_prec1 = 0.
early_bird_30 = EarlyBird(0.3)
early_bird_50 = EarlyBird(0.5)
early_bird_70 = EarlyBird(0.7)
for epoch in range(args.start_epoch, args.epochs):
    if early_bird_30.early_bird_emerge(model):
        print("[early_bird_30] Find EB!!!!!!!!!, epoch: "+str(epoch))
    if early_bird_50.early_bird_emerge(model):
        print("[early_bird_50] Find EB!!!!!!!!!, epoch: "+str(epoch))
    if early_bird_70.early_bird_emerge(model):
        print("[early_bird_70] Find EB!!!!!!!!!, epoch: "+str(epoch))
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec1 = test()
    history_score[epoch][2] = prec1
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, epoch, filepath=args.save)

print("Best accuracy: "+str(best_prec1))
history_score[-1][0] = best_prec1
np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
