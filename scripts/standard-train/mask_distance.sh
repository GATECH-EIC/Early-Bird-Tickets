CUDA_VISIBLE_DEVICES=0 python mask_cr.py \
--dataset cifar100 \
--start_epoch 1 \
--end_epoch 160 \
--depth 16 \
--arch vgg \
--percent 0.5 \
--save ./baseline/vgg16-cifar100 \
--save_1 ./baseline/vgg16-cifar100