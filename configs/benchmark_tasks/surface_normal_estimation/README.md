# Surface Normal Estimation

This code for the surface normal task lives [here](https://github.com/imisra/semantic-segmentation-pytorch).

It is a PyTorch codebase built on top of the [Semantic Segmentation toolbox by MIT](https://github.com/CSAILVision/semantic-segmentation-pytorch).
It is used for the Surface normal estimation task on the NYUv2 dataset in our work on [Scaling and Benchmarking Self-Supervised Visual Representation Learning
](https://arxiv.org/abs/1905.01235)

## Installation

The code has been tested with PyTorch 0.4.1. It has no additional requirements over the base segmentation toolbox code.

## Surface Normal Estimation Background
We follow the methodology from [Wang et al.](https://arxiv.org/abs/1411.4958) and use the surface normals computed by [Ladicky et al.](https://inf.ethz.ch/personal/pomarc/pubs/LadickyECCV14.pdf). Below, we provide a brief overview of this method.

The surface normal is a unit norm 3 dimensional vector (x, y, z) at each pixel location.
In this codebase, we first discretize this prediction problem so we can perform a classification task at each pixel.

### Quantize surface normals to train
The steps followed for creating the *training labels* are
- Perform k-means clustering of the surface normals into K=40 clusters. This is stored in the file `vocab40.mat`.
- At each pixel, assign the surface normal to one of the K clusters. This becomes the classification label for that pixel. This step is performed in the script `make_SN_labels.py`.
We use these training labels to train a segmentation network using `train_generic.py`. This network minimizes a standard cross-entropy classification loss.

### Convert to continuous to evaluate
We follow these steps to evaluate the predictions of a model.
- The network predicts a K-dimensional probability distribution at each pixel.
- We compute a weighted sum (using the above probabilities) of the surface normal cluster centers. This (after normalization) is the continuous surface normal output at the pixel location.
- The continuous surface normals are then evaluated following the metrics stated in [Fouhey et al.](http://www.cs.cmu.edu/~dfouhey/3DP/dfouhey_primitives.pdf). The details can be found in the file `normal_evaluation_utils.py`.

## Data
- The surface normal data (including the ground truth normal, train/test split) can be downloaded [here](https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/nyuv2_surfacenormal_metadata.zip). This contains a `pklz` (gzip compressed pickle) file containing the surface normal data (courtesy of Lubor Ladicky), the train/test `json` files needed to train models, `vocab40.mat` (courtesy of Xiaolong Wang) that has the k-means clusters of the surface normals.
- The NYUv2 dataset images can be downloaded from the [NYU website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). We use the 1449 images (795 images for training and 654 images for testing).
- [Optional] Download the PyTorch models used in our self-supervised paper. These are needed to reproduce the results in our work.

### Preprocessing and generating labels
- Download the NYUv2 dataset and the surface normal data.
- Add the paths to the script `make_SN_labels.py`.
- Run the script `python3 make_SN_labels.py` to generate the surface normal label files (png images used in the codebase)
- Change the paths in `dataset_generic.py` to point to the NYUv2 images and the surface normal label (png) files as well as the surface normal .

## Training and Testing
- Run the script `scripts/train_finetune_models.sh` to finetune models or `scripts/train_scratch_models.sh` to train models from scratch. The comments in these files show the settings used in our paper.
- Run the script `scripts/test_models.sh` to evaluate the model. Note that the surface normal evaluation needs about 150GB RAM (this can be made memory efficient, but is left as a TODO).

*Note*: There can be a difference of about 1 point in the median error. For example, in the paper we report (Table 7, row 4) a median error of 13.1. This is an average across runs and individual runs can vary from 12.4 to 14.

## References
If you find the code useful, please cite the following papers - 

Results from this code were used in our work:
```
@article{goyal2019self,
 title={{Scaling and Benchmarking Self-Supervised Visual Representation Learning}},
 author={Priya Goyal and Dhruv Mahajan and Abhinav Gupta and Ishan Misra},
 journal={arXiv preprint arXiv:1905.01235},
 year={2019}
}
```
Ground truth from:
```
@inproceedings{ladicky2014discriminatively,
 title={Discriminatively trained dense surface normal estimation},
 author={Ladick{\`y}, L and Zeisl, Bernhard and Pollefeys, Marc},
 booktitle={ECCV},
 volume={2},
 number={3},
 pages={4},
 year={2014}
}
```
Surface Normal task and metrics:
```
@inproceedings{fouhey2013data,
 title={Data-driven 3D primitives for single image understanding},
 author={Fouhey, David F and Gupta, Abhinav and Hebert, Martial},
 booktitle={ICCV},
 year={2013}
}
@inproceedings{wang2015designing,
  title={Designing deep networks for surface normal estimation},
  author={Wang, Xiaolong and Fouhey, David and Gupta, Abhinav},
  booktitle={CVPR},
  year={2015}
}
```
We build upon the codebase by:
```
@article{zhou2018semantic,
 title={Semantic understanding of scenes through the ade20k dataset},
 author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Xiao, Tete and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
 journal={International Journal on Computer Vision},
 year={2018}
}
```
