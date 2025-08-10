#!/bin/bash

DatasetPath=/data/
ArchConfig=./train_yaml/mos_pointrefine_stage.yml
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml
LogPath=./log/Train-Stage2
FirstStageModelPath=/path

export CUDA_VISIBLE_DEVICES=1 && python train_2stage.py -d $DatasetPath \
                                                        -ac $ArchConfig \
                                                        -dc $DataConfig \
                                                        -l $LogPath \
                                                        -p $FirstStageModelPath