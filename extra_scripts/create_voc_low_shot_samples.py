# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This script is used to create the low-shot data for VOC svm trainings.
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


def sample_symbol(input_targets, output_target, symbol, num):
    logger.info('Sampling symbol: {} for num: {}'.format(symbol, num))
    num_classes = input_targets.shape[1]
    for idx in range(num_classes):
        symbol_data = np.where(input_targets[:, idx] == symbol)[0]
        sampled = random.sample(list(symbol_data), num)
        for index in sampled:
            output_target[index, idx] = symbol
    return output_target


def generate_independent_sample(opts):
    assert os.path.exists(opts.targets_data_file), "Target file not found. Abort"
    targets = np.load(opts.targets_data_file, encoding='latin1')
    k_values = [int(val) for val in opts.k_values.split(",")]
    # the way sample works is: for each independent sample, and a given k value
    # we create a matrix of the same shape as given targets file. We initialize
    # this matrix with -1 (ignore label). We then sample k positive and
    # (num_classes-1) * k negatives.
    # N x 20 shape
    num_classes = targets.shape[1]
    for idx in range(opts.num_samples):
        for k in k_values:
            logger.info('Sampling: {} time for k-value: {}'.format(idx + 1, k))
            output = np.ones(targets.shape, dtype=np.int32) * -1
            output = sample_symbol(targets, output, 1, k)
            output = sample_symbol(targets, output, 0, (num_classes - 1) * k)
            prefix = opts.targets_data_file.split('/')[-1].split('.')[0]
            output_file = os.path.join(
                opts.output_path, '{}_sample{}_k{}.npy'.format(prefix, idx + 1, k)
            )
            logger.info('Saving file: {}'.format(output_file))
            np.save(output_file, output)
    logger.info('Done!!')


def main():
    parser = argparse.ArgumentParser(description='Sample Low shot data for VOC')
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
