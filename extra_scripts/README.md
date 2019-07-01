# Helpful scripts for data preparation and model pickling

We provide various helpful script to convert your data to the input format used by this codebase, sampling low-shot data, scripts to vary problem complexity for Jigsaw and Colorization problems and finally to convert models to use with Detectron or from Caffe2 to PyTorch 1.0.

## Preparing Data input files

Below are the example commands to prepare input data files for various datasets.

### Preparing COCO data files
We assume you have the data in the format as described [here](../self_supervision_benchmark/data/README.md).

```bash
python extra_scripts/create_coco_data_files.py \
    --json_annotations_dir /path/to/coco_json_annotations/coco \
    --output_dir /tmp/ssl-benchmark-output/coco/ \
    --train_imgs_path /path/to/coco/coco_train2014 \
    --val_imgs_path /path/to/coco/coco_val2014
```

### Preparing VOC data files
We assume you have the data in the format as described [here](../self_supervision_benchmark/data/README.md).

- For VOC2007 dataset:
```bash
python extra_scripts/create_voc_data_files.py \
    --data_source_dir /path/to/VOC2007/ \
    --output_dir /tmp/ssl-benchmark-output/voc07/
```

- For VOC2012 dataset:

```bash
python extra_scripts/create_voc_data_files.py \
    --data_source_dir /path/to/VOC2012/ \
    --output_dir /tmp/ssl-benchmark-output/voc12/
```

### Preparing ImageNet-1K data files
We assume you have the data in the format as described [here](../self_supervision_benchmark/data/README.md).

```bash
python extra_scripts/create_imagenet_data_files.py \
    --data_source_dir /path/to/imagenet_full_size/ \
    --output_dir /tmp/ssl-benchmark-output/imagenet1k/
```

### Preparing Places-205 data files
We assume you have the data in the format as described [here](../self_supervision_benchmark/data/README.md).

```bash
python extra_scripts/create_imagenet_data_files.py \
    --data_source_dir /path/to/places205/ \
    --output_dir /tmp/ssl-benchmark-output/places205/
```

## Varying Problem Complexity
We explore problem complexity as an axis of scaling in our [paper](https://arxiv.org/abs/1905.01235).

### Generating Jigsaw Permutations for varying problem complexity

For the problem of Jigsaw, we vary the number of permutations used
to solve the jigsaw task. In our paper, the permutations used `∈` [100, 701, 2000, 5000, 10000]. We provide these permutations files for download [here](../MODEL_ZOO.md). To generate the permutations, use the command below:

```bash
python extra_scripts/generate_jigsaw_permutations.py \
    --output_dir /tmp/ssl-benchmark-output/jigsaw_perms/ \
    -- N 2000
```

### Generating Colorization color bins and priors for varying problem complexity

For the problem of Image Colorization, we vary the number of color bins and their corresponding priors. To generate the color bins from a dataset and the color priors, use
the command below (example for the ImageNet-1K dataset, chose any dataset if you want):

```bash
python extra_scripts/generate_colorization_bins_priors.par \
    --type train \
    --output_path /tmp/ssl-benchmark-output/color_bins/ \
    --data_file /tmp/ssl-benchmark-output/imagenet1k/train_images.npy
```

## Low-shot data sampling
Low-shot image classification is one of the benchmark tasks in our [paper](https://arxiv.org/abs/1905.01235). We perform low-shot sampling on the PASCAL VOC and Places-205 datasets.

### Generating VOC2007 Low-shot samples

We train on trainval split of VOC2007 dataset which has 5011 images and 20 classes.
Hence the labels are of shape 5011 x 20. We generate 5 independent samples (for a given low-shot value `k`) by essentially generating 5 independent target files. For each class, we randomly
pick the positive `k` samples and `19 * k` negatives. Rest of the samples are ignored. We perform low-shot image classification on various different layers on the model (AlexNet, ResNet-50). The targets `targets_data_file` is usually obtained by extracting features for a given layer. For information on how to extract features, see [this](../GETTING_STARTED.md). Below command generates 5 samples for various `k` values:

```bash
python extra_scripts/create_voc_low_shot_samples.py \
    --targets_data_file /path/to/numpy_targets.npy \  
    --output_path /tmp/ssl-benchmark-output/voc07/low_shot/labels/ \
    --k_values "1,2,4,8,16,32,64,96" \
    --num_samples 5
```

### Generating Places-205 Low-shot samples

We generate 5 independent samples (for a given low-shot value `k`). For each class, we randomly pick the positive `k` samples and `204 * k` negatives. Rest of the samples are ignored. We perform low-shot image classification on various different layers on the model (AlexNet, ResNet-50). The `images_data_file` and `targets_data_file` are as obtained above in the data preparation step. Below command generates 5 samples for various `k` values:

```bash
python extra_scripts/create_places_low_shot_samples.py \
    --images_data_file /tmp/ssl-benchmark-output/places205/train_images.npy \
    --targets_data_file /tmp/ssl-benchmark-output/places205/train_labels.npy \
    --output_path /tmp/ssl-benchmark-output/places205/low_shot/ \
    --k_values "1,2,4,8,16,32,64,96" \
    --num_samples 5
```

## Converting model to Detectron format for object detection

Object Detection is one the benchmark tasks in our [paper](https://arxiv.org/abs/1905.01235). We convert the self-supervision models to the model that is compatible with Detectron. This involves: (1) Pickling jigsaw model to rename the blobs compatible with Detection model blobs (2) remove the momentum blobs for various model parameters, (3) converting the Spatial Batch Norm layers to Affine scale and bias layers, (4) absorbing the std deviation processing on the input as is done in Jigsaw pretext task.

### Rename Jigsaw model blob names
Jigsaw model training involves a 9-tower Siamese-CNN. The weights are shared among the 9 towers and the blobs of a tower has a suffix `s{i}` denoting the tower number. We remove these suffixes from the blob names so the model is compatible with Detectron blob names. You can pick any Jigsaw model as given in the [Model Zoo](../MODEL_ZOO.md), download the model to your local disk and use the command below to convert it.

```bash
python extra_scripts/pickle_jigsaw_model.py \
    --model_file </path/to/model.pkl> \
    --output_file /tmp/ssl-benchmark-output/convert_jigsaw_model.pkl
```

### Pickle jigsaw model for detection

We now take the above converted model and perform remaining steps to make the model
compatible with object detection training.

```bash
python extra_scripts/pickle_caffe2_detection.py \
    --c2_model /tmp/ssl-benchmark-output/convert_jigsaw_model.pkl \
    --output_file /tmp/ssl-benchmark-output/detection_model.pkl \
    --absorb_std
```

## Using PyTorch models

If you have a PyTorch model that you want to use benchmark suite on, you can easily/quickly convert your model to Caffe2 using the below command and run any benchmark task afterwards.

0. Jigsaw model

```bash
python extra_scripts/pickle_pytorch_to_caffe2.py \
    --pth_model </path/to/model.pth> \
    --output_model /tmp/ssl-benchmark-output/c2_jigsaw_model.pkl \
    --jigsaw True
```

0. Colorization and Supervised model

```bash
python extra_scripts/pickle_pytorch_to_caffe2.py \
    --c2_model </path/to/model.pth> \
    --output_model /tmp/ssl-benchmark-output/c2_model.pkl
```

## Converting model to PyTorch 1.0

If you are interested in converting the Caffe2 model to a PyTorch compatible model,
you can use the commands below. We provide instruction for converting a colorization pretext model, jigsaw pretext model and supervised training model.

0. Jigsaw model

```bash
python extra_scripts/pickle_caffe2_to_pytorch.py \
    --c2_model </path/to/model.pkl> \
    --output_model /tmp/ssl-benchmark-output/pth_jigsaw_model.dat \
    --jigsaw True
```

0. Colorization and Supervised model

```bash
python extra_scripts/pickle_caffe2_to_pytorch.py \
    --c2_model </path/to/model.pkl> \
    --output_model /tmp/ssl-benchmark-output/pth_model.dat
```

