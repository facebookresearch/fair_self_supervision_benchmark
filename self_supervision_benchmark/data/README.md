# Datasets setup

In this README, we describe the expected data format for various tasks. We provide support for the following datasets:

- [ImageNet-1K](http://www.image-net.org/)
- [Places205](http://places.csail.mit.edu/downloadData.html)
- [COCO2014](http://cocodataset.org/#download)
- [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/)
- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/)

Our tasks assume the data input to have to have two numpy files: (1) file containing images information (2) file containing corresponding labels for images. We provide [scripts](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/extra_scripts/README.md) that can be used to prepare these two files for a dataset of choice.

## COCO dataset
We assume the downloaded data to look like:

```
coco
|_ coco_train2014
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ coco_val2014
|_ ...
|_ annotations
   |_ instances_train2014.json
   |_ ...
```

The **annotations** are available for download with Detectron [here](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/data/README.md#coco-minival-annotations).


## PASCAL VOC dataset
We assume the downloaded data to look like:

```
VOC<year>
|_ JPEGImages
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ annotations
|  |_ voc_<year>_train.json
|  |_ voc_<year>_val.json
|  |_ ...
|_ VOCdevkit<year>
```

## ImageNet-1K dataset
We assume the downloaded data to look like:

```
imagenet_full_size
|_ train
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
|_ val
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
```


## Places-205 dataset
We assume the downloaded data to look like below. We take the original train/val split .csv file and convert the data to the format below.

```
places
|_ train
|  |_ <abbey>
|  |  |_<im-1-name>.jpg
|  |  |_...
|  |  |_<im-N-name>.jpg
|  |_ ...
|  |_ <yard>
|  |  |_<im-1-name>.jpg
|  |  |_...
|  |  |_<im-M-name>.jpg
|_ val
|  |  |_<im-1-name>.jpg
|  |  |_...
|  |  |_<im-N-name>.jpg
|  |_ ...
|  |_ <yard>
|  |  |_<im-1-name>.jpg
|  |  |_...
|  |  |_<im-M-name>.jpg
```

