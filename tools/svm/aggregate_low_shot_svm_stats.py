# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
Aggregate the stats over various independent samples for low-shot svm training.
Stats computed: mean, max, min, std

Relevant transfer tasks: Low-shot Image Classification VOC07 and Places205 low
shot samples.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import argparse
import logging
import numpy as np
import os
import sys

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def save_stats(output_dir, stat, output):
    out_file = os.path.join(output_dir, 'test_ap_{}.npy'.format(stat))
    logger.info('Saving {} to: {} {}'.format(stat, out_file, output.shape))
    np.save(out_file, output)


def aggregate_stats(opts):
    k_values = [int(val) for val in opts.k_values.split(",")]
    sample_inds = [int(val) for val in opts.sample_inds.split(",")]
    logger.info('Aggregating stats for k-values: {} and sample_inds: {}'.format(
        k_values, sample_inds))

    output_mean, output_max, output_min, output_std = [], [], [], []
    for k_idx in range(len(k_values)):
        k_low = k_values[k_idx]
        k_val_output = []
        for inds in range(len(sample_inds)):
            sample_idx = sample_inds[inds]
            file_name = 'test_ap_sample{}_k{}.npy'.format(sample_idx + 1, k_low)
            filepath = os.path.join(opts.output_path, file_name)
            if os.path.exists(filepath):
                k_val_output.append(np.load(filepath, encoding='latin1'))
            else:
                logger.info('file does not exist: {}'.format(filepath))
        # import pdb; pdb.set_trace()
        k_val_output = np.concatenate(k_val_output, axis=0)
        k_low_max = np.max(k_val_output, axis=0).reshape(-1, k_val_output.shape[1])
        k_low_min = np.min(k_val_output, axis=0).reshape(-1, k_val_output.shape[1])
        k_low_mean = np.mean(k_val_output, axis=0).reshape(-1, k_val_output.shape[1])
        k_low_std = np.std(k_val_output, axis=0).reshape(-1, k_val_output.shape[1])
        output_mean.append(k_low_mean)
        output_min.append(k_low_min)
        output_max.append(k_low_max)
        output_std.append(k_low_std)

    output_mean = np.concatenate(output_mean, axis=0)
    output_min = np.concatenate(output_min, axis=0)
    output_max = np.concatenate(output_max, axis=0)
    output_std = np.concatenate(output_std, axis=0)

    save_stats(opts.output_path, 'mean', output_mean)
    save_stats(opts.output_path, 'min', output_min)
    save_stats(opts.output_path, 'max', output_max)
    save_stats(opts.output_path, 'std', output_std)

    argmax_cls = np.argmax(output_mean, axis=1)
    argmax_mean, argmax_min, argmax_max, argmax_std = [], [], [], []
    for idx in range(len(argmax_cls)):
        argmax_mean.append(100.0 * output_mean[idx, argmax_cls[idx]])
        argmax_min.append(100.0 * output_min[idx, argmax_cls[idx]])
        argmax_max.append(100.0 * output_max[idx, argmax_cls[idx]])
        argmax_std.append(100.0 * output_std[idx, argmax_cls[idx]])
    for idx in range(len(argmax_max)):
        logger.info('mean/min/max/std: {} / {} / {} / {}'.format(
            round(argmax_mean[idx], 2), round(argmax_min[idx], 2),
            round(argmax_max[idx], 2), round(argmax_std[idx], 2),
        ))
    logger.info('All done!!')


def main():
    parser = argparse.ArgumentParser(description='Low shot SVM model test')
    parser.add_argument('--output_path', type=str, default=None,
                        help="Numpy file containing test AP result files")
    parser.add_argument('--k_values', type=str, default=None,
                        help="Low-shot k-values for svm testing. Comma separated")
    parser.add_argument('--sample_inds', type=str, default=None,
                        help="sample_inds for which to test svm. Comma separated")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    logger.info(opts)
    aggregate_stats(opts)


if __name__ == '__main__':
    main()
