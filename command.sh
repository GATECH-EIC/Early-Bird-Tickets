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

# retrain (re-init)
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
