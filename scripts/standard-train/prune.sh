python vggprune.py \
--dataset cifar100 \
--test-batch-size 256 \
--depth 16 \
--percent 0.3 \
--model ./baseline/vgg16-cifar100/EB-30-35.pth.tar \
--save ./baseline/vgg16-cifar100/pruned_3035_0.3 \
--gpu_ids 0