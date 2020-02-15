# Early-Bird-Tickets

[![ICLR2020: spotlight](https://img.shields.io/badge/ICLR2020-spotlight-success)](https://openreview.net/group?id=ICLR.cc/2020/Conference#accept-spotlight)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

This is PyTorch implementation of [Drawing Early-Bird Tickets: Toward More Efficient Training of Deep Networks](https://arxiv.org/abs/1909.11957)

ICLR 2020 spotlight oral paper

## Table of Content
<!-- [TOC] -->
<div class="toc">
<ul>
<li><a href="#early-bird-tickets">Early-Bird-Tickets</a><ul>
<li><a href="#table-of-content">Table of Content</a></li>
<li><a href="#introduction">Introduction</a></li>
<li><a href="#early-bird-tickets_1">Early-Bird Tickets</a><ul>
<li><a href="#existence-of-early-bird-tickets">Existence of Early-Bird Tickets</a></li>
<li><a href="#identify-early-bird-tickets">Identify Early-Bird Tickets</a></li>
<li><a href="#efficient-training-via-early-bird-tickets">Efficient Training via Early-Bird Tickets</a></li>
</ul>
</li>
<li><a href="#basic-usage">Basic Usage</a><ul>
<li><a href="#prerequisites">Prerequisites</a></li>
<li><a href="#core-training-options">Core Training Options</a></li>
<li><a href="#standard-train-for-identifying-early-bird-tickets">Standard Train for Identifying Early-Bird Tickets</a></li>
<li><a href="#retrain-to-restore-accuracy">Retrain to Restore Accuracy</a></li>
<li><a href="#low-precision-search-and-retrain">Low Precision Search and Retrain</a></li>
</ul>
</li>
<li><a href="#imagenet-experiments">ImageNet Experiments</a><ul>
<li><a href="#resnet18-on-imagenet">ResNet18 on ImageNet</a></li>
<li><a href="#resnet50-on-imagenet">ResNet50 on ImageNet</a></li>
</ul>
</li>
<li><a href="#citation">Citation</a></li>
<li><a href="#acknowledgement">Acknowledgement</a></li>
</ul>
</li>
</ul>
</div>

## Introduction

- **[Lottery Ticket Hypothesis](https://openreview.net/forum?id=rJl-b3RcF7)**: (Frankle & Carbin, 2019) shows that there exist winning tickets (small but critical subnetworks) for dense, randomly initialized networks, that can be trained alone to achieve comparable accuracies to the latter in a similar number of iterations.

- **Limitation**: However, the identification of these winning tickets still requires the costly train-prune-retrain process, limiting their practical benefits.

- **Our Contributions**:
    + We discover for the first time that the winning tickets can be identified at the very early training stage, which we term as early-bird (EB) tickets, via low-cost training schemes (e.g., early stopping and low-precision training) at large learning rates. Our finding of EB tickets is consistent with recently reported observations that the key connectivity patterns of neural networks emerge early.
    + Furthermore, we propose a mask distance metric that can be used to identify EB tickets with low computational overhead, without needing to know the true winning tickets that emerge after the full training.
    + Finally, we leverage the existence of EB tickets and the proposed mask distance to develop efficient training methods, which are achieved by first identifying EB tickets via low-cost schemes, and then continuing to train merely the EB tickets towards the target accuracy.

Experiments based on various deep networks and datasets validate: 1) the existence of EB tickets, and the effectiveness of mask distance in efficiently identifying them; and 2) that the proposed efficient training via EB tickets can achieve up to 4.7x energy savings while maintaining comparable or even better accuracy, demonstrating a promising and easily adopted method for tackling cost-prohibitive deep network training.

## Early-Bird Tickets

### Existence of Early-Bird Tickets
To articulate the Early-Bird (EB) tickets phenomenon: the winning tickets can be drawn very early in training, we perform ablation simulation using two representative deep models (VGG16 and PreResNet101) on two popular datasets (CIFAR10 and CIFAR100). Specifically, we follow the main idea of [(Frankle & Carbin, 2019)](https://openreview.net/forum?id=rJl-b3RcF7) but instead prune networks trained at earlier points to see if reliable tickets can be drawn. We adopt the same channel pruning in [(Liu et al., 2017)](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) as pruning techniuqes for all experiments since it aligns with our end goal of efficient trianing. Below figure demonstrates the existence of EB tickets (p = 30% means 30% weights are pruned, hollow star means retraining accuracy of subnetwork drawn from checkpoint with best accuracy in search stage).

![](./assets/eb-retrain.png)

### Identify Early-Bird Tickets
we visialize distance evolution process among the tickets drawn from each epoch.
Below figure plots the pairwise mask distance matrices (160 x 160) of the VGG16 and PreResNet101 experiments on CIFAR100 at different pruning ratio `p`, where `(i, j)-th` element in a matrix denotes the mask distance between epochs `i` and `j` in that corresponding experiment. A lower distance (close to 0) indicates a smaller mask distance and is colored warmer.

<!-- ![](./assets/overlap.png) -->
<div align=center>
    <img src="./assets/overlap.png" width = "800" alt="overlap"  />
</div>

Our observation that the ticket masks quickly become stable and hardly changed in early training stages supports drawing EB tickets. We therefore measure the mask distance consecutive epochs, and draw EB tickets when such distance is smaller than a threshold.
Practically, to improve the reliability of EB tickets, we will stop to draw EB tickets when the last five recorded mask distances are all smaller than given threshold.

### Efficient Training via Early-Bird Tickets
Instead of adopting a three-step routine of 1) training a dense model, 2) pruning it and 3) then retraining the pruned model to restore performance, and these three steps can be iterated, we leverage the existence of EB tickets to develop EB Train scheme which replaces the aforementioned steps 1 and 2 with a lower-cost step of detecting the EB tickets.
<!-- ![](./assets/eb-train.png) -->
<div align=center>
    <img src="./assets/eb-train.png" width = "700" alt="eb-train"  />
</div>
<br>

## Basic Usage
### Prerequisites
The code has the following dependencies:

- python 3.7
- pytorch 1.1.0
- torchvision 0.3.0
- Pillow (PIL) 5.4.1
- scipy 1.2.1
- [qtorch](https://github.com/Tiiiger/QPyTorch) 0.1.1 (for low precision)
- GCC >= 4.9 on linux (for low precision)

### Core Training Options
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


### Standard Train for Identifying Early-Bird Tickets
Example: Reproduce early-bird (EB) tickets on CIFAR-100

* Step1: Standard train to find EB tickets at different pruning ratio. Note that one can directly stop training after identifying the emergence of EB tickets while we keep training here to compare among underlying subnetworks drawn at different training stages.
````
bash ./scripts/standard-train/search.sh
````

* Step2: Conduct real prune for the saved checkpoints (checkpoints containing EB tickets are represented as `EB-{pruning ratio}-{drawing epoch}.pth.tar` format).
````
bash ./scripts/standard-train/prune.sh
````

* Optional: Pairwise mask distance matrix visualization.
````
bash ./scripts/standard-train/mask_distance.sh
````

After calculating mask distance matrix (automatically save as `overlap-0.5.npy`), u can call `plot_overlap.py` to draw figures.

### Retrain to Restore Accuracy
Example: Retrain drawn EB tickets (e.g., VGG16 for CIFAR-100) to restore accuracy

* Finetune EB tickets from emergence epoch. Note we keep sparsity regularization for underlying iterative pruning.
````
bash ./scripts/standard-train/retrain_continue.sh
````

* Retrain re-initialized EB tickets from scratch (refer to `EB Train (re-init)` in Sec. 4.3 of paper).
````
bash ./scripts/standard-train/retrain_scratch.sh
````

### Low Precision Search and Retrain
We perform low precision method [SWALP](https://arxiv.org/pdf/1904.11943.pdf) to both the search and retrian stages (refer to `EB Train LL` in Sec. 4.3 of paper). Below is the guidance taking VGG16 performed on CIFAR-10 as an example:

* Step1:  Standard train to find EB tickets at different pruning ratio.
````
bash ./scripts/low-precision/search.sh
````

* Step 2: Conduct real prune for the saved checkpoints.
````
bash ./scripts/low-precision/prune.sh
````

* Step 3: Finetune EB tickets from emergence epoch.
````
bash ./scripts/low-precision/retrain_continue.sh
````

* Comparison example
<div align=left>
    <img src="./assets/vgg-result.png" width = "800" alt="eb-train"  />
</div>

## ImageNet Experiments
All pretrained checkpoints of different pruning ratio have been collected in [Google Drive](https://drive.google.com/open?id=1aTl47KR8oaHkOxqKOUDgQyI96dW7JgHJ). To evaluate the inference accuracy of test set, we provide evaluation scripts ( `EVAL_ResNet18_ImageNet.py` and `EVAL_ResNet50_ImageNet.py` ) and corresponding commands shown below for your convenience.

````
bash ./scripts/resnet18-imagenet/evaluation.sh
bash ./scripts/resnet50-imagenet/evaluation.sh
````

### ResNet18 on ImageNet
* Step1:  Standard train to find EB tickets at different pruning ratio.
````
bash ./scripts/resnet18-imagenet/search.sh
````

* Step 2: Conduct real prune for the saved checkpoints.
````
bash ./scripts/resnet18-imagenet/prune.sh
````

* Step 3: Finetune EB tickets from emergence epoch.
````
bash ./scripts/resnet18-imagenet/retrain_continue.sh
````

* comparison results
<div align=left>
    <img src="./assets/resnet18-result.png" width = "700" alt="eb-train"  />
</div>

### ResNet50 on ImageNet
* Step1: Standard train to find EB tickets at different pruning ratio.
````
bash ./scripts/resnet50-imagenet/search.sh
````

* Step 2: Conduct real prune for the saved checkpoints.
````
bash ./scripts/resnet50-imagenet/prune.sh
````

* Step 3: Finetune EB tickets from emergence epoch.
````
bash ./scripts/resnet50-imagenet/retrain_continue.sh
````


## Citation

If you find this code is useful for your research, please cite:
````
@inproceedings{
you2020drawing,
title={Drawing Early-Bird Tickets: Toward More Efficient Training of Deep Networks},
author={Haoran You and Chaojian Li and Pengfei Xu and Yonggan Fu and Yue Wang and Xiaohan Chen and Yingyan Lin and Zhangyang Wang and Richard G. Baraniuk},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJxsrgStvr}
}
````

## Acknowledgement
* Code is inspired by [Rethink-Network-Pruning](https://github.com/Eric-mingjie/rethinking-network-pruning).
* Thank for the constructive suggestions of [Yifan](https://github.com/yueruchen), [Tianlong](https://github.com/TianlongChenTAMU) and [Zhihan](https://github.com/Zhihan-Lu).
* Thank [Yifan](https://github.com/yueruchen) for helping accelerate ImageNet training.
