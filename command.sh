# search (e.g. for vgg16@cifar100)
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

# prune (30%, 50%, 70%)
python vggprune.py \
--dataset cifar100 \
--test-batch-size 256 \
--depth 16 \
--percent 0.3 \
--model ./baseline/vgg16-cifar100/EB-30-35.pth.tar \
--save ./baseline/vgg16-cifar100/pruned_3035_0.3 \
--gpu_ids 0

# retrain (finetune)
python main_c.py \
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
--start_epoch 35

# retrain (re-init)
python main_scratch.py \
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
--gpu_ids 0

######## ImageNet resnet18
# search
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset imagenet \
--data /data1/ILSVRC2017/ILSVRC/Data/CLS-LOC \
--arch resnet18 \
--depth 18 \
--lr 0.1 \
--epochs 90 \
--schedule 30 60 \
--batch-size 256 \
--test-batch-size 64 \
--save ./EBTrain-ImageNet/ResNet18 \
--momentum 0.9 \
--sparsity-regularization

# prune
CUDA_VISIBLE_DEVICES=5 python resprune.py \
--dataset imagenet \
--data /data3/imagenet-data/raw-data \
--arch resnet18 \
--test-batch-size 128 \
--depth 18 \
--percent 0.3 \
--model ./EBTrain-ImageNet/ResNet18/EB-30-11.pth.tar \
--save ./EBTrain-ImageNet/ResNet18/pruned_3011_0.3 \

# retrain
CUDA_VISIBLE_DEVICES=3 python main_c.py \
--dataset imagenet \
--data /data3/imagenet-data/raw-data \
--arch resnet18 \
--depth 18 \
--lr 0.1 \
--epochs 90 \
--schedule 30 60 \
--batch-size 128 \
--test-batch-size 64 \
--save ./EBTrain-ImageNet/ResNet18/retrain_1011_0.1 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./EBTrain-ImageNet/ResNet18/pruned_1011_0.1/pruned.pth.tar \
--start-epoch 11

######## ImageNet resnet50
# search
python -m torch.distributed.launch main_resnet50.py \
--dataset imagenet \
--data /data3/imagenet-data/raw-data \
--arch resnet50_official \
--depth 50 \
--lr 0.1 \
--epochs 90 \
--schedule 30 60 \
--batch-size 256 \
--test-batch-size 64 \
--save ./EBTrain-ImageNet/ResNet50 \
--momentum 0.9 \
--sparsity-regularization \
--gpu_ids 0,1,2,3

# prune
python resprune_50.py \
--dataset imagenet \
--data /data3/imagenet-data/raw-data \
--arch resnet50_official \
--test-batch-size 128 \
--depth 50 \
--percent 0.7 \
--model ./EBTrain-ImageNet/ResNet50/EB-70-8.pth.tar \
--save ./EBTrain-ImageNet/ResNet50/pruned_7008_0.7

# retrain
python -m torch.distributed.launch main_resnet50.py \
--dataset imagenet \
--data /data3/imagenet-data/raw-data \
--arch resnet50_prune \
--depth 50 \
--lr 0.1 \
--epochs 90 \
--schedule 30 60 \
--batch-size 256 \
--test-batch-size 128 \
--save ./EBTrain-ImageNet/ResNet50/retrain_7008_0.7 \
--scratch ./EBTrain-ImageNet/ResNet50/pruned_7008_0.7/pruned.pth.tar \
--momentum 0.9 \
--gpu_ids 4,5,6,7 \
--port 14000

