CUDA_VISIBLE_DEVICES=0 python resprune.py \
--dataset imagenet \
--data /data3/imagenet-data/raw-data \
--arch resnet18 \
--test-batch-size 128 \
--depth 18 \
--percent 0.3 \
--model ./EBTrain-ImageNet/ResNet18/EB-30-11.pth.tar \
--save ./EBTrain-ImageNet/ResNet18/pruned_3011_0.3 \