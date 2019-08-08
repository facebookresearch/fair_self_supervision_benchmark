# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This script is used to load the image data files (data/labels) at the beginning
of the dataloader. During the dataloader run, we pass the minibatch images
information to the loader.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import logging
import numpy as np
import os

from self_supervision_benchmark.core.config import config as cfg
import self_supervision_benchmark.data.dataset_catalog as dataset_catalog

# create the logger
logger = logging.getLogger(__name__)

DATA_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
DATA_STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)

class ImageDataset():

    def __init__(self, split):
        self._split = split
        self._image_paths = None
        self._image_classes = None
        # train = 1, test = 0 -> this is used in multi-processing step
        self._split_num = int(split == 'train')
        self._get_data()

    def get_db_size(self):
        db_size = self._image_paths.shape[0]
        return db_size

    def get_minibatch_path_indexes(self, indices):
        image_paths = [self._image_paths[idx] for idx in indices]
        labels = [self._image_classes[idx] for idx in indices]
        split_list = [self._split_num] * len(indices)
        return image_paths, labels, split_list

    def _get_data(self):
        # this will load the imagePaths and image class
        data_file, labels_file = dataset_catalog.get_data_labels_file(
            cfg.DATASET, self._split
        )
        logger.info('loading data from: {}'.format(data_file))
        logger.info('loading labels from: {}'.format(labels_file))
        self._image_paths = np.load(data_file, encoding='latin1')
        self._image_classes = np.load(labels_file, encoding='latin1')
        logger.info('============={} images: {} labels: {}============='.format(
            self._split, self._image_paths.shape, self._image_classes.shape))

class NpyDataset():
    def __init__(self, split):
        self._split = split
        # train = 1, test = 0 -> this is used in multi-processing step
        self._split_num = int(split == 'train')
        self._get_data()
        self._image_paths = None
        self._image_classes = None

    def get_db_size(self):
        db_size = int(self.data.shape[0] / cfg.TEST.TEN_CROP_N)
        return db_size

    def get_minibatch_path_indexes(self, indices):
        data = [self.data[idx*cfg.TEST.TEN_CROP_N:(idx+1)*cfg.TEST.TEN_CROP_N] for idx in indices]
        data = np.vstack(data).astype(np.float32)
        data = preprocess_imgs(data)
        # data = convert_to_batch(data)
        lbls = [1 for _ in indices]
        lbls = np.array(lbls).astype(np.int32)
        return data, lbls, lbls.copy()

    def _get_data(self):
        # this will load the imagePaths and image class
        logger.info('loading data /home/yihuihe/outs/')
        self.data = parts_read()

def preprocess_imgs(imgs):
    #N 64 64 3
    imgs = imgs / 255.
    imgs = (imgs - DATA_MEAN) / DATA_STD
    return imgs.transpose((0, 3, 1, 2))

def convert_to_batch(data):
    data = np.concatenate(
        [arr[np.newaxis] for arr in data]).astype(np.float32)
    return np.ascontiguousarray(data)

def parts_read(name='imgs', maxparts=10000000, sel=None):
    start = 0
    end = 0

    cnt = 0
    #preallocation
    dir = cfg.EXTRACT.IMGS_DIR #'/home/yihuihe/outs/'
    for idx in range(maxparts):
        prefix = '{}_{}_part{}_{}.npy'.format('trainval', 'res4_5_branch2c_bn_s0', idx, name)
        prefix = os.path.join(dir, prefix)
        logger.info('preallocation ' + prefix)
        if idx == 0:
            feat = np.load(prefix)
        else:
            if not os.path.exists(prefix):
                cnt = idx
                break
    wholeshape = list(feat.shape)
    wholeshape[0] = wholeshape[0] * cnt
    feats = np.ndarray(wholeshape, dtype=np.uint8)

    for idx in range(maxparts):
        prefix = '{}_{}_part{}_{}.npy'.format('trainval', 'res4_5_branch2c_bn_s0', idx, name)
        prefix = os.path.join(dir, prefix)
        logger.info(prefix)
        try:
            feat = np.load(prefix)
            end += len(feat)
            feats[start:end] = feat
            start = end
        except Exception as e:
            print(e)
            break
    feats = feats[:end]
    return feats
