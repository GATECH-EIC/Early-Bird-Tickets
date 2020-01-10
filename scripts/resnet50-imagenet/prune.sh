python resprune_50.py \
--dataset imagenet \
--data /data3/imagenet-data/raw-data \
--arch resnet50_official \
--test-batch-size 128 \
--depth 50 \
--percent 0.7 \
--model ./EBTrain-ImageNet/ResNet50/EB-70-8.pth.tar \
--save ./EBTrain-ImageNet/ResNet50/pruned_7008_0.7