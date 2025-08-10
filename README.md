# BA-MOS
This repository contains the official implementation of the paper:
> **BA-MOS: LiDAR Point Cloud Moving Object Segmentation Based on Boundary Awareness and 3D Feature Refinement**\
> Mu Zhou*, Rui Xiang, Wei He, Nan Du, Liangbo Xie\
> (Accepted by *Acta Optica Sinica*)
## Abstract:
<p align="justify">
To address the problem of low segmentation accuracy of moving objects in LiDAR point clouds under complex moving environments caused by blurred object boundaries, information loss, and false detections of static objects, this paper proposes a moving object segmentation method based on boundary awareness and 3D feature refinement. Specifically, a dual-branch projection network is first employed, with the motion branch playing a dominant role, jointly extracting and fusing information from a movable semantic branch to effectively capture moving object features. Second, boundary information is extracted and deeply fused at multiple encoding levels, significantly reducing boundary blur and enhancing object discriminability. Finally, a moving object correction method is introduced, which incorporates multi-scale channel aggregation sparse convolution based on 3D point clouds and a movability constraint. This method refines local point cloud features and effectively suppresses false detections of static objects, thereby improving segmentation accuracy. Experimental results show that the proposed method achieves an Intersection over Union (IoU) 78.7% for moving segmentation on the SemanticKITTI-MOS dataset and further validates its effectiveness and scene adaptability through real-world measurement data.
</p>

## Framework:
### Stage1:
<img width="6281" height="2806" alt="框架" src="https://github.com/user-attachments/assets/ef1f1ac9-ee2f-46d0-92dc-a3dce176a720" />

### Stage2:
<img width="5025" height="1650" alt="二阶段1" src="https://github.com/user-attachments/assets/46fae294-be6c-400b-82ef-611b07fd9418" />


## Installation
Our environment: Ubuntu 20.04, CUDA 12.2, NVIDIA A40 GPU x 2

Create a conda env with
```bash
conda env create -f environment.yml
```
### TorchSparse
The stage2 training requires installing `torchsparse`. The command is as follows:
```bash
# Install TorchSparse follow https://github.com/mit-han-lab/torchsparse
sudo apt install libsparsehash-dev 
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

## Dataset:
**Download SemanticKITTI** from [official website](http://www.semantic-kitti.org/dataset.html)\
To get the residual maps as the input of the model during training, run `utils/auto_gen_residual_images.py`.
The standard data format is as follows:
```bash
DATAROOT
└── sequences
    ├── 00
    │   ├── poses.txt
    │   ├── calib.txt
    │   ├── labels
    │   ├── residual_images_1
    │   ├── residual_images_2
    │   ├── residual_images_3
    │   ├── residual_images_4
    │   ├── residual_images_5
    │   ├── residual_images_6
    │   ├── residual_images_7
    │   ├── residual_images_8
    │   └── velodyne
   ...
```

## Usage:
### Train
#### Train on Stage 1:
```bash
./scripts/train_stage1.sh
```
#### Train on Stage 2:
Upon finishing the first-stage training, proceed with the second-stage training. Training can only be performed on a single GPU.
```bash
./scripts/train_stage2.sh
```
### Infer and Eval
- Infer on SemanticKITTI datasets by:
```bash
./scripts/valid.sh
```
- Eval for valid sequences:\
The evaluation of the datasets in our experiments is conducted using the [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api) provided by the SemanticKITTI benchmark.

## Acknowledgments
We would like to thank the developers of the following open-source projects for providing valuable code and tools that supported this work: [MF-MOS](https://github.com/SCNU-RISLAB/MF-MOS.git), [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext.git), 
and [SPVNAS](https://github.com/mit-han-lab/spvnas.git). Thanks the contributors of these repositories!
