# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
A helper script to generate N permutations for M patches. The permutations are
selected based on hamming distance.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import itertools
import logging
import numpy as np
import os
import sys
from scipy.spatial.distance import cdist

# initiate the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Permutations for patches')
    parser.add_argument('--N', type=int, default=1000,
                        help='Number of permuations')
    parser.add_argument('--M', type=int, default=9,
                        help='Number of patches to permute')
    parser.add_argument('--method', type=str, default='max_avg',
                        choices=['max_avg', 'max_min'],
                        help='hamming distance : max, avg')
    parser.add_argument('--min_distance', type=float, default=2.0 / 9.0,
                        help='min distance of permutations in final set')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='output dir')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # now generate data permutation for num_perms, num_patches and save them.
    # The algorithm followed is same as in https://arxiv.org/pdf/1603.09246.pdf
    # Algorithm 1 on page 12.
    logger.info("Generating all perms....")
    num_perms, num_patches = args.N, args.M
    all_perms = np.array(list(itertools.permutations(list(range(num_patches)))))
    total_perms = all_perms.shape[0]

    logger.info("Selecting perms from set of {} perms".format(total_perms))
    for idx in range(num_perms):
        if idx == 0:
            j = np.random.randint(total_perms)  # uniformaly sample first perm
            selected_perms = all_perms[j].reshape([1, -1])
        else:
            selected_perms = np.concatenate(
                [selected_perms, all_perms[j].reshape([1, -1])], axis=0)
        all_perms = np.delete(all_perms, j, axis=0)
        # compute the hamming distance now between the remaining and selected
        D = cdist(selected_perms, all_perms, metric='hamming')
        if args.method == 'max_avg':
            D = D.mean(axis=0)
            j = D.argmax()
        elif args.method == 'max_min':
            min_to_selected = D.min(axis=0)
            j = min_to_selected.argmax()
            if min_to_selected.min() < args.min_distance:
                logger.info("min distance {} < threshold {}".format(
                    min_to_selected.min(), args.min_distance))
        elif args.method == 'avg':
            logger.info('not implemented yet')
        if (idx + 1) % 100 == 0:
            logger.info('selected_perms: {} -> {}'.format(
                (idx + 1), selected_perms.shape))

    dists_sel = cdist(selected_perms, selected_perms, metric='hamming')
    non_diag_elements = dists_sel[np.where(np.eye(dists_sel.shape[0]) != 1)]
    mean_dist = non_diag_elements.mean() * selected_perms.shape[1]
    min_dist = non_diag_elements.min() * selected_perms.shape[1]
    logger.info('Permutation stats: avg dist {}; min dist {}'.format(
        mean_dist, min_dist))

    perm_file = os.path.join(
        args.output_dir,
        'hamming_perms_{}_patches_{}_{}.npy'.format(args.N, args.M, args.method)
    )
    logger.info('Writing permutations to: {}'.format(perm_file))
    logger.info('permutations shape: {}'.format(selected_perms.shape))
    np.save(perm_file, selected_perms)
    logger.info('Done!')


if __name__ == '__main__':
    main()
