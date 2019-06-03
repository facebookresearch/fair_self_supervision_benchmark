# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This script is used to create the low-shot data for Places205 svm trainings.
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import logging
import numpy as np
import os
import random
import sys

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def find_num_positives(input_targets):
    num_classes = max(input_targets) + 1
    num_pos = []
    for idx in range(num_classes):
        pos = len(np.where(input_targets == idx)[0])
        num_pos.append(pos)
    return num_pos


def sample_data(input_images, input_targets, num):
    sample_imgs, sample_lbls = [], []
    num_classes = max(input_targets) + 1
    logger.info('sampling for num: {}'.format(num))
    for idx in range(num_classes):
        sample = input_images[np.where(input_targets == idx)]
        # import pdb; pdb.set_trace()
        subset_imgs = random.sample(list(sample), num)
        sample_lbls.extend([idx] * num)
        sample_imgs.extend(subset_imgs)
    return sample_imgs, sample_lbls


def generate_independent_sample(opts):
    logger.info('Reading images and labels file...')
    assert os.path.exists(opts.targets_data_file), "Target file not found. Abort"
    assert os.path.exists(opts.images_data_file), "Images file not found. Abort"
    targets = np.load(opts.targets_data_file, encoding='latin1')
    images = np.load(opts.images_data_file, encoding='latin1')
    logger.info('images: {} labels: {}'.format(images.shape, targets.shape))

    # get the maximum and minumum number of positives per class
    logger.info('getting number of positives per class information...')
    num_pos = find_num_positives(targets)
    logger.info('min #num_pos: {}, max #num_pos: {}'.format(
        min(num_pos), max(num_pos)))

    k_values = [int(val) for val in opts.k_values.split(",")]

    # start sampling now
    for idx in range(opts.num_samples):
        for k in k_values:
            if (k > min(num_pos)):
                logger.info('Skip k: {} min #pos: {}'.format(k, min(num_pos)))
                continue
            logger.info('sampling: {} time for k: {}'.format(idx + 1, k))
            out_imgs, out_lbls = sample_data(images, targets, k)
            suffix = 'sample{}_k{}'.format(idx + 1, k)
            out_img_file = os.path.join(
                opts.output_path, 'train_images_{}.npy'.format(suffix))
            out_lbls_file = os.path.join(
                opts.output_path, 'train_labels_{}.npy'.format(suffix))
            logger.info('Saving imgs file: {} {}'.format(out_img_file, len(out_imgs)))
            logger.info('Saving lbls file: {} {}'.format(out_lbls_file, len(out_lbls)))
            np.save(out_img_file, out_imgs)
            np.save(out_lbls_file, out_lbls)
    logger.info('Done!')


def main():
    parser = argparse.ArgumentParser(description='Sample Low-shot data for Places')
    parser.add_argument('--images_data_file', type=str, default=None,
                        help="Numpy file containing image")
    parser.add_argument('--targets_data_file', type=str, default=None,
                        help="Numpy file containing image labels")
    parser.add_argument('--output_path', type=str, default=None,
                        help="path where low-shot samples should be saved")
    parser.add_argument('--k_values', type=str, default="1,2,4,8,16,32,64,96",
                        help="Low-shot k-values for svm testing.")
    parser.add_argument('--num_samples', type=int, default=5,
                        help="Number of independent samples.")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    opts = parser.parse_args()
    generate_independent_sample(opts)


if __name__ == '__main__':
    main()
