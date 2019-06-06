# FAIR Self-Supervision Benchmark Model Zoo

We provide all the models used in our [paper](https://arxiv.org/abs/1905.01235). Further, we also provide the input files to vary the problem complexity of Jigsaw and Colorization approaches by varying the number of permutations and number of color bins respectively.

## Models

### AlexNet

- ImageNet-1K supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/caffenet_bvlc_in1k_supervised.npy
- Places-205 supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/caffenet_bvlc_places205_supervised.pkl
- Jigsaw ImageNet-1K self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/alexnet_jigsaw_in1k_pretext.pkl
- Jigsaw ImageNet-22K self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/alexnet_jigsaw_in22k_pretext.pkl
- Jigsaw YFCC100M self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/alexnet_jigsaw_yfcc100m_pretext.pkl
- Colorization ImageNet-1K self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/alexnet_colorization_in1k_pretext.pkl
- Colorization ImageNet-22K self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/alexnet_colorization_in22k_pretext.pkl
- Colorization YFCC100M self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/alexnet_colorization_yfcc100m_pretext.pkl

### ResNet-50
- ImageNet-1K supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_in1k_supervised.pkl
- Places-205 supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_places205_supervised.pkl
- Jigsaw ImageNet-1K self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_jigsaw_in1k_pretext.pkl
- Jigsaw ImageNet-22K self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_jigsaw_in22k_pretext.pkl
- Jigsaw YFCC100M self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_jigsaw_yfcc100m_pretext.pkl
- Jigsaw ImageNet-22K self-supervised for object detection: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_jigsaw_detection_in22k_pretext.pkl
- Colorization ImageNet-1K self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_colorization_in1k_pretext.pkl
- Colorization ImageNet-22K self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_colorization_in22k_pretext.pkl
- Colorization ImageNet-22K self-supervised for Low-shot training: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_colorization_lowShot_in22k_pretext.pkl
- Colorization YFCC100M self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_colorization_yfcc100m_pretext.pkl

## Object Detection Models and Proposals

- ImageNet-1K supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/detection/resnet50_in1k_supervised.pkl
- Places-205 supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/detection/resnet50_places205_supervised.pkl
- Jigsaw ImageNet-1K self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/detection/resnet50_jigsaw_in1k_pretext.pkl
- Jigsaw ImageNet-22K self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/detection/resnet50_jigsaw_in22k_pretext.pkl
- Jigsaw YFCC100M self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/detection/resnet50_jigsaw_yfcc100m_pretext.pkl

- Selective Search Proposals VOC2007 trainval: https://dl.fbaipublicfiles.com/detectron/selective_search_proposals/selective_search_msra_voc_2007_trainval.pkl
- Selective Search Proposals VOC2007 test: https://dl.fbaipublicfiles.com/detectron/selective_search_proposals/selective_search_msra_voc_2007_test.pkl
- Selective Search Proposals VOC2012 trainval: https://dl.fbaipublicfiles.com/detectron/selective_search_proposals/selective_search_msra_voc_2012_trainval.pkl

## PyTorch Models for Visual Navigation

- ImageNet-1K supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_in1k_supervised_gibson.dat
- Jigsaw ImageNet-22K self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_jigsaw_in22k_pretext_gibson.dat
- Jigsaw YFCC100M self-supervised: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_in1k_supervised_gibson.dat

## Surface Normal Estimation Data

- Ground Truth surface normals and their metadata: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/nyuv2_surfacenormal_metadata.zip

## Jigsaw Permutations
All permutations below are for 9 patches:

- 100 permutations: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/jigsaw_permutations/naroozi_perms_100_patches_9_max.npy
- 701 permutations: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/jigsaw_permutations/naroozi_perms_701_patches_9_max.npy
- 2000 permutations: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/jigsaw_permutations/hamming_perms_2000_patches_9_max_avg.npy
- 5000 permutations: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/jigsaw_permutations/hamming_perms_5000_patches_9_max_avg.npy
- 10000 permutations: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/jigsaw_permutations/hamming_perms_10000_patches_9_max_avg.npy

## Colorization color bins and priors

Number of color bins vary based on the bin size used. We generate the color bins and priors for various bin sizes:

- Bin size 10 colors: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/colorization_bins_priors/priors_train_imagenet1k_bin10.npy
- Bin size 10 color priors: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/colorization_bins_priors/pts_in_hull_train_imagenet1k_bin10.npy

- Bin size 15 colors: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/colorization_bins_priors/priors_train_imagenet1k_bin15.npy
- Bin size 15 colors priors: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/colorization_bins_priors/pts_in_hull_train_imagenet1k_bin15.npy

- Bin size 20 colors: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/colorization_bins_priors/priors_train_imagenet1k_bin20.npy
- Bin size 20 colors priors: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/colorization_bins_priors/pts_in_hull_train_imagenet1k_bin20.npy

- Bin size 5 colors: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/colorization_bins_priors/priors_train_imagenet1k_bin5.npy
- Bin size 5 colors priors: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/colorization_bins_priors/pts_in_hull_train_imagenet1k_bin5.npy

- Original work Bin size 10 colors: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/colorization_bins_priors/zhang_prior_probs.npy
- Original work Bin size 10 colors priors: https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/colorization_bins_priors/zhang_pts_in_hull.npy

