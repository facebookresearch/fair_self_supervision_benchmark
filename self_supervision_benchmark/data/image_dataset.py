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
