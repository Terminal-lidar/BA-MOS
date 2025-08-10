#!/bin/bash

DatasetPath=/data/
ModelPath=/YourPath/
SavePath=./log/predictions/
SPLIT=test

# If you want to use stage2, set pointrefine on
export CUDA_VISIBLE_DEVICES=3 && python3 infer.py -d $DatasetPath \
                                                  -m $ModelPath \
                                                  -l $SavePath \
                                                  -s $SPLIT \
                                                  --pointrefine \
                                                  --movable # Whether to save the label of movable objects
