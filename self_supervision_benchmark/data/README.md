### Supported Datasets

We provide support for the following datasets used in the benchmark tasks.

0. ImageNet-1K
1. Places205
2. COCO2014
3. VOC2007

*NOTE:* for places205 dataset, we assume the data to be available as well just
like ImageNet-1k which follows the structure:

```
<root_folder>/<train or val>/<class_name>/image.jpg
```
Inside fbcode, we have places205 dataset inside everstore to load from. See https://fb.quip.com/2BH7AoRi7uVg for where the dataset lives originally on gfsai.
