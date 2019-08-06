# Using FAIR Self-Supervision Benchmark

We provide a brief tutorial for running various evaluations using various tasks (benchmark/legacy) on various datasets.

- For installation, please refer to [`INSTALL.md`](INSTALL.md).
- For setting up various datasets, please refer to [this README](self_supervision_benchmark/data/README.md).

## Feature Extraction

We believe self-supervision as a way to learn feature representation and to evaluate the quality of representation learned, we train Linear SVMs on frozen feature representations extracted from various layers of the model (AlexNet/ResNet-50). We provide various YAML configuration files to reproduce results in our paper for models AlexNet/ResNet-50 and feature source (random/supervised training/self-supervised task either jigsaw or colorization). Below are the commands for extracting features and for reproducing results in our [paper](https://arxiv.org/abs/1905.01235).

#### 1. Extracting features for VOC2007 Linear SVM training

All relevant configuration files are available in `configs/benchmark_tasks/image_classification/voc07`. To extract features for training (test) dataset, we use the `data_type` as `train (test)` and `output_file_prefix` as `trainval (test)`. We also generate the input `TRAIN.DATA_FILE`, `TRAIN.LABELS_FILE`, `TEST.DATA_FILE` and `TEST.LABELS_FILE` using the scripts [here](extra_scripts/README.md). Below is an example usage:

- For random features

```bash
python tools/extract_features.py \
    --config_file configs/benchmark_tasks/image_classification/voc07/caffenet_bvlc_random_extract_features.yaml \
    --data_type train \
    --output_file_prefix trainval \
    --output_dir /tmp/ssl-benchmark-output/extract_features/random \
    TRAIN.DATA_FILE /tmp/ssl-benchmark-output/voc07/train_images.npy \
    TRAIN.LABELS_FILE /tmp/ssl-benchmark-output/voc07/train_labels.npy
```

- For weights initialization (supervised/self-supervised) weights

Specify the weights you want to use as `TEST.PARAMS_FILE` input. An example is shown below:

```bash
python tools/extract_features.py \
    --config_file configs/benchmark_tasks/image_classification/voc07/caffenet_bvlc_supervised_extract_features.yaml \
    --data_type train \
    --output_file_prefix trainval \
    --output_dir /tmp/ssl-benchmark-output/extract_features/weights_init \
    TEST.PARAMS_FILE https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/caffenet_bvlc_in1k_supervised.npy \
    TRAIN.DATA_FILE /tmp/ssl-benchmark-output/voc07/train_images.npy \
    TRAIN.LABELS_FILE /tmp/ssl-benchmark-output/voc07/train_labels.npy
```

#### 2. Extracting features for COCO2014 Linear SVM training

All relevant configuration files are available in `configs/benchmark_tasks/image_classification/coco2014`. To extract features for training (test) dataset, we use the `data_type` as `train (test)` and `output_file_prefix` as `trainval (test)`. We also generate the input `TRAIN.DATA_FILE`, `TRAIN.LABELS_FILE`, `TEST.DATA_FILE` and `TEST.LABELS_FILE` using the scripts [here](extra_scripts/README.md). Below is an example usage:

- For weights initialization (supervised/self-supervised) weights

Specify the weights you want to use as `TEST.PARAMS_FILE` input. An example is shown below:

```bash
python tools/extract_features.py \
    --config_file configs/benchmark_tasks/image_classification/coco2014/caffenet_bvlc_supervised_extract_features.yaml \
    --data_type train \
    --output_file_prefix trainval \
    --output_dir /tmp/ssl-benchmark-output/extract_features/weights_init \
    TEST.PARAMS_FILE https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/caffenet_bvlc_in1k_supervised.npy \ TRAIN.DATA_FILE /tmp/ssl-benchmark-output/coco/train_images.npy \
    TRAIN.LABELS_FILE /tmp/ssl-benchmark-output/coco/train_labels.npy
```

#### 3. Extracting features for Places205 Low-shot Linear SVM training
For Places-205, given the dataset size if 2.48M, we only train Linear SVM classifiers in case of low-shot image classification. We provide the scripts to generate the low-shot samples [here](extra_scripts/README.md). All relevant configuration files are available in `configs/benchmark_tasks/low_shot_image_classification/places205`. To extract features for training (test) dataset, we use the `data_type` as `train (test)` and `output_file_prefix` as `trainval (test)`. Below is an example usage:

- For weights initialization (supervised/self-supervised) weights

Specify the weights you want to use as `TEST.PARAMS_FILE` input. An example is shown below:

```bash
python tools/extract_features.py \
    --config_file configs/benchmark_tasks/low_shot_image_classification/places205/resnet50_supervised_low_shot_extract_features.yaml \ --data_type train \
    --output_file_prefix trainval \
    --output_dir /tmp/ssl-benchmark-output/extract_features/weights_init \
    TEST.PARAMS_FILE https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_in1k_supervised.pkl \
    TRAIN.DATA_FILE /tmp/ssl-benchmark-output/places205/train_images_sample4_k1.npy \
    TRAIN.LABELS_FILE /tmp/ssl-benchmark-output/places205/train_labels_sample4_k1.npy
```

## Running various Benchmark tasks
We describe example commands for running 9 benchmark tasks:

#### Benchmark task 1: Places205 Image Classification
We train linear classifiers on feature representations from various layers of the model (AlexNet/ResNet-50). The configuration files are provided in `configs/benchmark_tasks/image_classification/places205`. We provide various YAML configuration files to reproduce results in our paper for models AlexNet/ResNet-50 and feature source (random/supervised training/self-supervised task either jigsaw or colorization). We also generate the input `TRAIN.DATA_FILE`, `TRAIN.LABELS_FILE`, `TEST.DATA_FILE` and `TEST.LABELS_FILE` using the scripts [here](extra_scripts/README.md). An example usage is shown below:

```bash
python tools/train_net.py \
    --config_file configs/benchmark_tasks/image_classification/places205/caffenet_bvlc_supervised_finetune_linear.yaml \
    TRAIN.DATA_FILE /tmp/ssl-benchmark-output/places205/train_images.npy \
    TRAIN.LABELS_FILE /tmp/ssl-benchmark-output/places205/train_labels.npy \
    TEST.DATA_FILE /tmp/ssl-benchmark-output/places205/val_images.npy \
    TEST.LABELS_FILE /tmp/ssl-benchmark-output/places205/val_labels.npy
```

#### Benchmark task 2: Training Object detection model with frozen backbone on VOC07 and VOC07+12 data
First, install the [Detectron](https://github.com/facebookresearch/Detectron) following the detectron Install commands [here](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md). We freeze the backbone of the model using `FREEZE_AT=4` and use a slightly longer training schedule for both supervised and self-supervised baselines. All other hyperparameters are same as the Detectron settings. The configuration files for Fast R-CNN and Faster R-CNN on VOC2007 and VOC07+12 datasets are provided in `configs/benchmark_tasks/object_detection_frozen/`. We also provide the model weights converted to be compatible with Detectron blob names and input (see [this](extra_scripts/README.md). You can use model you want to use from the config file and download the model to local disk and use the command below for training.

- **Step 1:** Rename Jigsaw model weights

For this, follow the instructions [here](https://github.com/facebookresearch/fair_self_supervision_benchmark/tree/master/extra_scripts#rename-jigsaw-model-blob-names).

- **Step 2:** Convert model weights to Detectron compatible weights

For this, follow the instructions [here](https://github.com/facebookresearch/fair_self_supervision_benchmark/tree/master/extra_scripts#pickle-jigsaw-model-for-detection).

- **Step 3:** Train detection model using the command below:

```bash
# NOTE: this train_net.py is the Detectron codebase train_net.py
python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/benchmark_tasks/object_detection_frozen/voc07/e2e_faster_rcnn_R-50-C4_trainval-50-FPN.yaml \
    OUTPUT_DIR /tmp/detectron-output \
    TRAIN.WEIGHTS <downloaded_weights.pkl>
```

#### Benchmark task 3: Training Linear SVM on VOC07 and COCO2014 datasets

We train linear SVM classifiers on feature representations from various layers of the model (AlexNet/ResNet-50). We provide various YAML configuration files to reproduce results in our paper for models AlexNet/ResNet-50 and feature source (random/supervised training/self-supervised task either jigsaw or colorization). Training SVM involves following steps:

- **Step 1: Prepare Data files**

For a given dataset, prepare the input data file by following the instructions described [here](extra_scripts/README.md).

- **Step 2: Extract Features**

We then extract features from various layers of the model (AlexNet/ResNet-50) for both train/test partition of the target dataset (VOC07/COCO2014). The YAML configuration files for feature extraction are provided under `configs/benchmark_tasks/image_classification/voc07` and `configs/benchmark_tasks/image_classification/coco2014`. For extracting features, see the instructions provided at the beginning of this README.

- **Step 3: Train SVM**

Once we have the features extracted, we can train linear SVM on it. For the demonstration purpose, we chose AlexNet model supervised baseline and `conv1` layer features. Following is the sample command:

```bash
python tools/svm/train_svm_kfold.py \
    --data_file /tmp/ssl-benchmark-output/extract_features/weights_init/voc07/trainval_conv1_s4k19_resize_features.npy \
    --targets_data_file /tmp/ssl-benchmark-output/extract_features/weights_init/voc07/trainval_conv1_s4k19_resize_targets.npy \
    --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
    --output_path /tmp/ssl-benchmark-output/voc07_svm/svm_conv1/
```

- **Step 4: Test SVM**

Once we have the SVM trained, we can test the SVM classifier to get the mAP. Here's a sample command:

```bash
python tools/svm/test_svm.py \
  --data_file /tmp/ssl-benchmark-output/extract_features/weights_init/voc07/test_conv1_s4k19_resize_features.npy \
  --targets_data_file /tmp/ssl-benchmark-output/extract_features/weights_init/voc07/test_conv1_s4k19_resize_targets.npy \
  --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
  --output_path /tmp/ssl-benchmark-output/voc07_svm/svm_conv1/
```

#### Benchmark Task 4: Low-shot Image Classification using Linear SVM on VOC07 and Places205 dataset
We train linear SVM classifiers for low-shot image classification on feature representations from various layers of the model (ResNet-50) for VOC07 and Places205 datasets. We provide various YAML configuration files to reproduce results in our paper for models AlexNet/ResNet-50 and feature source (random/supervised training/self-supervised task either jigsaw or colorization). Following are the steps for low-shot image classification:

- **Step 1: Prepare Low-shot Data samples**

For a given dataset (VOC07 / Places205), we generate 5 independent sample for various low-shot values. The scripts and instructions to generate these samples are provided [here](extra_scripts/README.md).

- **Step 2: Extract features**

We then extract features for all generated low-shot samples. The features are extracted from various layers of the model (ResNet-50) for both train/test partition of the target dataset (VOC07/COCO2014). The YAML configuration files for feature extraction are provided under `configs/benchmark_tasks/low_shot_image_classification/voc07` and `configs/benchmark_tasks/low_shot_image_classification/places205`. For extracting features, see the instructions provided at the beginning of this README.

- **Step 3: Train SVM**

Once the features are extracted, the SVM can be trained using the following command. For demonstration purpose, we chose ResNet50 model (Jigsaw pre-trained), features on VOC07 dataset and `conv1` layer.

```bash
python tools/svm/train_svm_low_shot.py \
    --data_file /tmp/ssl-benchmark-output/extract_features/trainval_res_conv1_bn_s0_resize_features.npy \
    --targets_data_file /tmp/ssl-benchmark-output/extract_features/low_shot/labels/trainval_res_conv1_bn_s0_resize_targets_sample1_k1.npy  \
    --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
    --output_path /tmp/ssl-benchmark-output/voc07_svm_low_shot/svm_conv1/
```

- **Step 4: Testing SVM**

Once we have the SVM trained, we can test the SVM classifier to get the mAP. The SVM is tested for all k-values and all independent samples. For the testing purpose, we always test on the full test set.

```bash
python tools/svm/test_svm_low_shot.py \
    --data_file /tmp/ssl-benchmark-output/extract_features/test_res_conv1_bn_s0_resize_features.npy \
    --targets_data_file /tmp/ssl-benchmark-output/extract_features/test_res_conv1_bn_s0_resize_targets.npy \
    --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
    --output_path /tmp/ssl-benchmark-output/voc07_svm_low_shot/svm_conv1/ \
    --k_values "1,2,4,8,16,32,64,96" \
    --sample_inds "0,1,2,3,4"
```

- **Step 5: Gathering stats - mean/min/max/std**

Next we report various stats across 5 independent samples. For this, we provide the script and the command below. The `output_path` should be the same as the previous step. One can vary the `sample_inds` to gather stats over any independent sample(s).

```bash
python tools/svm/aggregate_low_shot_svm_stats.py \
    --output_path /tmp/ssl-benchmark-output/voc07_svm_low_shot/svm_conv1/ \
    --k_values "1,2,4,8,16,32,64,96" \
    --sample_inds "0,1,2,3,4"
```

#### Benchmark Task 5: Visual Navigation
Please see the README [here](configs/benchmark_tasks/visual_navigation/README.md) for instructions.

#### Benchmark Task 6: Surface Normal Estimation
Please see the README [here](configs/benchmark_tasks/surface_normal_estimation/README.md) for instructions.

## Running various Legacy tasks
We describe example commands for running 9 benchmark tasks:

#### Legacy task 1: Training ImageNet-1K linear classifier

We train linear classifiers on feature representations from various layers of the model (AlexNet/ResNet-50). The configuration files are provided in `configs/legacy_tasks/imagenet_linear_tune`. We provide various YAML configuration files to reproduce results in our paper for models AlexNet/ResNet-50 and feature source (random/supervised training/self-supervised task either jigsaw or colorization). We also generate the input `TRAIN.DATA_FILE`, `TRAIN.LABELS_FILE`, `TEST.DATA_FILE` and `TEST.LABELS_FILE` using the scripts [here](extra_scripts/README.md). An example usage is shown below:

```bash
python tools/train_net.py \
    --config_file configs/legacy_tasks/imagenet_linear_tune/caffenet_bvlc_supervised_finetune_linear.yaml \
    TRAIN.PARAMS_FILE https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/caffenet_bvlc_places205_supervised.pkl \ TRAIN.DATA_FILE /tmp/ssl-benchmark-output/imagenet1k/train_images.npy \
    TRAIN.LABELS_FILE /tmp/ssl-benchmark-output/imagenet1k/train_labels.npy \
    TEST.DATA_FILE /tmp/ssl-benchmark-output/imagenet1k/val_images.npy \
    TEST.LABELS_FILE /tmp/ssl-benchmark-output/imagenet1k/val_labels.npy
```

#### Legacy task 2: Training Object detection model with full fine-tuning
First, install the Detectron following the detectron Install commands [here](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md). We use a slightly longer training schedule for both supervised and self-supervised baselines. All other hyperparameters are same as the Detectron settings. The configuration files for Fast R-CNN and Faster R-CNN on VOC2007 and VOC07+12 datasets are provided in `configs/legacy_tasks/object_detection_full_finetune/`. We also provide the model weights converted to be compatible with Detectron blob names and input (see [this](extra_scripts/README.md). You can use model you want to use from the config file and download the model to local disk and use the command below for training.

```bash
python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/legacy_tasks/object_detection_full_finetune/voc07/e2e_faster_rcnn_R-50-C4_trainval-50-FPN.yaml \
    OUTPUT_DIR /tmp/detectron-output \
    TRAIN.WEIGHTS <downloaded_weights.pkl>
```

#### Legacy task 3: VOC07 full fine-tuning
In this task the model is initialized from the weights (supervised or self-supervised) and the full network is fine-tuned. This task helps measuring how good the weights initialization is and not the quality of feature representations (hence a legacy task). We provide various YAML configuration files under `configs/legacy_tasks/voc07_full_finetune` to reproduce results in our paper for models AlexNet/ResNet-50 and feature source (random/supervised training/self-supervised task either jigsaw or colorization). We also generate the input `TRAIN.DATA_FILE`, `TRAIN.LABELS_FILE`, `TEST.DATA_FILE` and `TEST.LABELS_FILE` using the scripts [here](extra_scripts/README.md). An example command is shown below:

```bash
python tools/train_net.py \
    --config_file configs/legacy_tasks/voc07_full_finetune/alexnet_colorization_voc07_finetune_full.yaml \
    TRAIN.PARAMS_FILE https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/alexnet_colorization_in22k_pretext.pkl \ TRAIN.DATA_FILE /tmp/ssl-benchmark-output/voc07/train_images.npy \
    TRAIN.LABELS_FILE /tmp/ssl-benchmark-output/voc07/train_labels.npy \
    TEST.DATA_FILE /tmp/ssl-benchmark-output/voc07/test_images.npy \
    TEST.LABELS_FILE /tmp/ssl-benchmark-output/voc07/test_labels.npy
```


