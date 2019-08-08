### COCO2014 Image Classification Task

# What?
For the COCO2014 dataset image classification task, we train SVM classifiers on frozen feature representations of various layers of a model.

0. Models: AlexNet, ResNet-50
1. Pretext: Supervised, Jigsaw

# How?
0. **Step 0: Data preparation:**
You will need the COCO 2014 annotations and images available at http://cocodataset.org. The exact instructions for doing so are available at https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/data/README.md#setting-up-datasets. Once you have the dataset setup correctly, you can proceed to prepare the data files that can be used as input for the next steps. Please follow the steps [`here`](../../../.../extra_scripts/README.md).

1. **Step 1: Feature extraction**:
We first extract the features for various layers of models mentioned above.
We provide scripts for feature extraction as the yaml config files.

2. **Step 2: SVM training:**
We provide an example bash script to train linear SVM classifiers for the frozen
extracted features. You can adapt the script

3. **Step 3: SVM testing:**
We provide the bash scripts to test the SVM models on the test dataset.
