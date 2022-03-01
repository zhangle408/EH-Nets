#!/usr/bin/env bash


#---in PyramidNet--------------------------------------------
python train_cifar.py \
--arch=resnet_cifar_56 \
--dataset=cifar10 \
--epochs=400 \
--start_epoch=0 \
--batch_size=128 \
--DCT_root \
--DCT_flag \
--warmup=10 \
--balance_weight=0.0001 \
--lr=0.1 \
--compund_level=1
