import argparse
import numpy as np
import os,sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from compute_flops import print_model_param_nums, print_model_param_flops

import models
from qtorch.quant import *
from qtorch.optim import OptimLP
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.auto_low import sequential_lower

num_types = ["weight", "activate", "grad", "error", "momentum"]

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--data', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

# quantized parameters
for num in num_types:
    parser.add_argument('--wl-{}'.format(num), type=int, default=-1, metavar='N',
                        help='word length in bits for {}; -1 if full precision.'.format(num))
parser.add_argument('--rounding'.format(num), type=str, default='stochastic', metavar='S',
                    choices=["stochastic","nearest"],
                    help='rounding method for {}, stochastic or nearest'.format(num))

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.__dict__['vgg'](dataset=args.dataset, depth=args.depth)
# automatically insert quantization modules
model = sequential_lower(model, layer_types=["conv", "linear"],
                         forward_number=number_dict["activate"], backward_number=number_dict["error"],
                         forward_rounding=args.rounding, backward_rounding=args.rounding)
# removing the final quantization module
model.classifier = model.classifier[0]

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        # args.start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids).cuda()
            model.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              # .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
        exit()

print('original model param: ', print_model_param_nums(model))
print('original model flops: ', print_model_param_flops(model, 32, True))

if args.cuda:
    model.cuda()

total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = []
protective_thre = {}
bn_index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        bn_index += 1
        size = m.weight.data.shape[0]
        w = m.weight.data.abs().clone()
        # protective layer-wise threshold
        y, i = torch.sort(w)
        thre_index = int(size * 0.9) # protect top 10%
        protective_thre[bn_index] = y[thre_index]
        # remaining channels are needed to prune
        for item in y[:thre_index]:
            bn.append(item)

bn = torch.FloatTensor(bn)
p_flops = 0
y, i = torch.sort(bn)
# comparsion and permutation (sort process)
p_flops += total * np.log2(total) * 3
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []
bn_index = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        bn_index += 1
        weight_copy = m.weight.data.abs().clone()
        _mask = weight_copy.gt(thre.cuda()).float().cuda()
        # add protective channels
        protective_mask = weight_copy.gt(protective_thre[bn_index].cuda()).float().cuda()
        mask = torch.zeros(_mask.size())
        for i, (a, b) in enumerate(zip(_mask, protective_mask)):
            if a==1 or b==1:
                mask[i] = 1
        mask = mask.cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        if int(torch.sum(mask)) > 0:
            cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

# apply mask flops
# p_flops += total

# compare two mask distance
p_flops += 2 * total #(minus and sum)

pruned_ratio = pruned/total

print('  + Memory Request: %.2fKB' % float(total * 32 / 1024 / 8))
print('  + Flops for pruning: %.2fM' % (p_flops / 1e6))

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
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

def test(model):
    """
    Load the model.

    Args:
        model: (todo): write your description
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'imagenet':
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
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=16, pin_memory=True)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    test_acc = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_acc, len(test_loader), test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)

acc = test(model)

# Make real prune
print(cfg)
newmodel = models.__dict__['vgg'](dataset=args.dataset, depth=args.depth, cfg=cfg)
# automatically insert quantization modules
newmodel = sequential_lower(newmodel, layer_types=["conv", "linear"],
                         forward_number=number_dict["activate"], backward_number=number_dict["error"],
                         forward_rounding=args.rounding, backward_rounding=args.rounding)
# removing the final quantization module
newmodel.classifier = newmodel.classifier[0]

if len(args.gpu_ids) > 0:
    newmodel = torch.nn.DataParallel(newmodel, device_ids=args.gpu_ids)
if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc))

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        if torch.sum(end_mask) == 0:
            continue
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        if torch.sum(end_mask) == 0:
            continue
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        # random set for test
        # new_end_mask = np.asarray(end_mask.cpu().numpy())
        # new_end_mask = np.append(new_end_mask[int(len(new_end_mask)/2):], new_end_mask[:int(len(new_end_mask)/2)])
        # idx1 = np.squeeze(np.argwhere(new_end_mask))

        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

# print(newmodel)
model = newmodel
# test(model)
param = print_model_param_nums(model)
flops = print_model_param_flops(model.cpu(), 32, True)
with open(savepath, "w") as fp:
    fp.write("new model param: \n"+str(param)+"\n")
    fp.write("new model flops: \n"+str(flops)+"\n")
print('new model param: ', param)
print('new model flops: ', flops)