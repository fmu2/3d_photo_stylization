# Stylized 3D Photo from a Single Image

## Introduction

PyTorch implementation for 3D photo stylization from a single image. Code has been tested on python 3.7, pytorch 1.7.1, CUDA 10.1 and Ubuntu 16.04. This stable branch is for running ablation study and creating demos.

## Overview

Our implementation follows a multi-stage pipeline.

- **(TODO) Monocular depth estimation.** Given an RGB image, we run LeReS for depth and shape estimation. More precisely, the output of this step is a coarse depth map up to unknown scale at the same resolution as the input, along with the estimated field of view (FOV) under the pinhole camera model.

- **(TODO) Point cloud inpainting.** We apply the method of Shih et al. (casually known as 3DPhoto) to convert the input image and its estimated depth into a complete 3D point cloud. Specifically, 3DPhoto refines the depth map by sharpening depth discontinuities, lifts RGB pixels into a layered depth image (LDI), and performs context-aware color and depth inpainting to fill in the dissoccluded regions. The output of this step is an inpainted LDI. It is trivial to convert the LDI into a 3D point cloud given camera FOV.

- **Learning encoder-decoder model for image reconstruction.** As in most image style transfer methods, we first train an encoder-decoder model for the task of image reconstruction. This model will later serve as the backbone for style transfer. Run `train_inpaint.py` to train the encoder-decoder model.

- **Stylization via feature modulation.** Once trained, the encoder-decoder model is kept frozen. A stylization module learns to blend the desired style into the content image by modulating the encoder output. Run `train_stylize.py` to train the stylization module.

## Quick Start

### Preliminaries

- Clone this repo
```shell
git clone https://github.com/fmu2/3d_photo_stylization.git
cd 3d_photo_stylization
```

- Create conda environment and install required packages
```shell
conda create -n 3d_photo_stylization python=3.7
conda activate 3d_photo_stylization
conda install -c pytorch pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0
conda install -c conda-forge tensorboardx
conda install -c anaconda scipy
conda install -c anaconda pyyaml
```

- Compile C++ and CUDA extension
```shell
python setup_render.py build_ext --inplace
python setup_pointnet2.py build_ext --inplace
```

- Prepare datasets
Download the pre-processed COCO2014 dataset [here](https://drive.google.com/drive/folders/1qtv-fBrCJXT93w0QoFbt8YqeI09pS8EM?usp=sharing) and the WikiArt dataset [here](https://drive.google.com/drive/folders/1cwjSJaNJyRqQI-2wmFWNwgxkN6pUOy6k?usp=sharing). Unzip the files.
```shell
cd coco_pcd
tar -xvzf train.tar.gz
tar -xvzf val.tar.gz
cd ..
cd wikiart
tar -xvzf train.tar.gz
tar -xvzf val.tar.gz
cd ..
```

Images from the COCO2014 training split are resized to 448 x 448 and converted into LDIs following the aforementioned steps. Each LDI (or 3D point cloud) contains approximately 300K points. We use 80000 LDIs for training and the remaining for validation. The raw WikiArt images are extremely large. We down-sample them to 512 x 512 and keep the original training and validation splits.

- Recommended folder structure
```
3d_photo_stylization
│   README.md
│   ...    
│
└───data
│   └───coco_pcd/
│   │	  └───train/
│   │   │   └───ldi/
│   │   │   │   └───ldi/
│   │   │   │   │   │   COCO_train2014_000000XXXXXX.mat
│   │   │   │   │   │   COCO_train2014_000000XXXXXX.mat
│   │   │   │   │   │   ...
│   │   └───val/
│   │   │   └───ldi/
│   │   │   │   └───ldi/
│   │   │   │   │   │   COCO_train2014_000000XXXXXX.mat
│   │   │   │   │   │   COCO_train2014_000000XXXXXX.mat
│   │   │   │   │   │   ...
│   └───wikiart/
│   │   └───train/
│   │   │   │   XXX.jpg
│   │   │   │   XXX.jpg
│   │   │   │   ...
│   │   └───val/
│   │   │   │   XXX.jpg
│   │   │   │   XXX.jpg
│   │   │   │   ...
└───configs
│   │	  └───ablation/
│   │   │   │   inpaint3d_r3_pixel_content.yaml
│   │   │   │   inpaint3d_r3_pixel_content_match.yaml
│   │   │   │   ...
└───log
│   ...
```

### Training
To train the encoder-decoder network for image reconstruction, run
```
python train_inpaint.py -d data/coco_pcd -c configs/{file_name}.yaml -n {job_name} -g {gpu_id}
```
The latest model checkpoint `inpaint-last.pth` and the config file `inpaint-config.yaml` will be saved in `log/{job_name}`.

To train the stylization module for style transfer, run
```
python train_stylize.py -d data/coco_pcd -s data/wikiart -c configs/{file_name}.yaml -n {job_name} -g {gpu_id}
```
Note that the job name needs to exactly match that of a pre-trained image reconstruction model. The latest model checkpoint `stylize-last.pth` and the config file `stylize-config.yaml` will be saved in `log/{job_name}`.

### Evaluation

## Contact
[Fangzhou Mu](http://pages.cs.wisc.edu/~fmu/) (fmu2@wisc.edu)

## Related Code Repos

Our code relies heavily on the following repos.
* LeReS <https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS>
* 3d-photo-inpainting <https://github.com/vt-vl-lab/3d-photo-inpainting>
* Pointnet2.PyTorch <https://github.com/sshaoshuai/Pointnet2.PyTorch>

Our code is inspired by the following repos.
* 3d-ken-burns <https://github.com/sniklaus/3d-ken-burns>
* softmax-splatting <https://github.com/sniklaus/softmax-splatting>
* deep_gcns_torch <https://github.com/lightaime/deep_gcns_torch>
* pytorch-AdaIN <https://github.com/naoto0804/pytorch-AdaIN>
* LinearStyleTransfer <https://github.com/sunshineatnoon/LinearStyleTransfer>
* AdaAttN <https://github.com/Huage001/AdaAttN>
* partialconv <https://github.com/NVIDIA/partialconv>
* stylescene <https://github.com/hhsinping/stylescene>
* synsin <https://github.com/facebookresearch/synsin>

## References
```
@inproceedings{Wei2021CVPR,
	author = {Yin, Wei and Zhang, Jianming and Wang, Oliver and Niklaus, Simon and Mai, Long and Chen, Simon and Shen, Chunhua},
  title = {Learning to Recover 3D Scene Shape from a Single Image},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
}

@inproceedings{Shih3DP20,
  author = {Shih, Meng-Li and Su, Shih-Yang and Kopf, Johannes and Huang, Jia-Bin},
  title = {3D Photography using Context-aware Layered Depth Inpainting},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```