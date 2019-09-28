# Early-Bird-Tickets

This is PyTorch implementation of [Drawing Early-Bird Tickets: Toward More Efficient Training of Deep Networks](https://arxiv.org/abs/1909.11957)

## Introduction

(Frankle & Carbin, 2019) shows that there exist winning tickets (small but critical subnetworks) for dense, randomly initialized networks, that can be trained alone to achieve comparable accuracies to the latter in a similar number of iterations. However, the identification of these winning tickets still requires the costly train-prune-retrain process, limiting their practical benefits. In this paper, we discover for the first time that the winning tickets can be identified at the very early training stage, which we term as early-bird (EB) tickets, via low-cost training schemes (e.g., early stopping and low-precision training) at large learning rates. Our finding of EB tickets is consistent with recently reported observations that the key connectivity patterns of neural networks emerge early. Furthermore, we propose a mask distance metric that can be used to identify EB tickets with low computational overhead, without needing to know the true winning tickets that emerge after the full training. Finally, we leverage the existence of EB tickets and the proposed mask distance to develop efficient training methods, which are achieved by first identifying EB tickets via low-cost schemes, and then continuing to train merely the EB tickets towards the target accuracy. Experiments based on various deep networks and datasets validate: 1) the existence of EB tickets, and the effectiveness of mask distance in efficiently identifying them; and 2) that the proposed efficient training via EB tickets can achieve up to 4.7x energy savings while maintaining comparable or even better accuracy, demonstrating a promising and easily adopted method for tackling cost-prohibitive deep network training.

## Prerequisites
The code has the following dependencies:

- python 3.7
- pytorch 1.1.0
- torchvision 0.3.0
- Pillow (PIL) 5.4.1
- scipy 1.2.1

## Core Training Options
- `dataset`: which dataset you want to use CIFAR10/100 by default
- `data`: If you want to use ImageNet, plz specified the path to raw data
- `batch-size`: all exps use 256 by default in paper
- `epochs`: total epochs, 160 in total
- `schedule`: at which points the learning rate degraded, use [80, 120] by default
- `lr`: initial learning rate, 0.1 by default
- `save`: save checkpoints to the specific directory
- `arch`: which model you want to use, support vgg and resnet now
- `depth`: model depth
- `filter`: apply filter to dataset, default is none
- `sparsify_gt`: sparify the dataset with given percentage
- `gpu_ids`: multi-gpus is supported

## Usage

### Standard train for identifying early-bird tickets
* e.g., for VGG16 performed on CIFAR-100

````
CUDA_VISIBLE_DEVICES=0 nohup python main.py \
    --dataset cifar100 \
    --arch vgg \
    --depth 16 \
    --lr 0.1 \
    --epochs 160 \
    --schedule 80 120 \
    --batch-size 256 \
    --test-batch-size 256 \
    --save ./baseline/vgg16-cifar100 \
    --momentum 0.9 \
    --sparsity-regularization \
    --filter none \
    --sigma 1 \
    --sparsity_gt 0 \
    >> ./baseline/vgg16-cifar100.out &
````

* real prune the saved checkpoints

````
python vggprune.py \
--dataset cifar100 \
--test-batch-size 256 \
--depth 16 \
--percent 0.3 \
--model ./baseline/vgg16-cifar100/EB-30-35.pth.tar \
--save ./baseline/vgg16-cifar100/pruned_3035_0.3 \
--gpu_ids 0
````

### Retrain to restore accuracy
* e.g., for VGG16 performed on CIFAR-100 (finetune)

````
nohup python main_c.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.1 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/vgg16-cifar100/retrain_1035_0.1 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/vgg16-cifar100/pruned_1035_0.1/pruned.pth.tar \
--gpu_ids 0 \
--start_epoch 35 &

````

* e.g., for VGG16 performed on CIFAR-100 (from scratch)

````
nohup python main_scratch.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.1 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/vgg16-cifar100/retrain_1035_0.1_scratch \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/vgg16-cifar100/pruned_1035_0.1/pruned.pth.tar \
--gpu_ids 0 &

````