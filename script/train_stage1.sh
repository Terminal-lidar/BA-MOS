#!/bin/bash

DatasetPath=/data/
ArchConfig=./train_yaml/ddp_mos_coarse_stage.yml
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml
LogPath=./log/Train-Stage1

export CUDA_VISIBLE_DEVICES=0,1 && torchrun --nproc_per_node=1 \
                                           ./train.py -d $DatasetPath \
                                                      -ac $ArchConfig \
                                                      -dc $DataConfig \
                                                      -l $LogPath