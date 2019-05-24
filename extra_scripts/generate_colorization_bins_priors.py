# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
A helper script to quantize the color space into bins and generating the priors
for the bins. These are used for training the colorization models.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import argparse
import cv2
import logging
import numpy as np
import os
import sys

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def convert2lab(img):
    # img is [0, 255] , HWC, BGR format, uint8 type
    assert len(img.shape) == 3, 'Image should have dim H x W x 3'
    assert img.shape[2] == 3, 'Image should have dim H x W x 3'
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # 8-bit image range -> L [0, 255], A [0, 255], B [0, 255]. Rescale it to:
    # L [0, 100], A [-128, 127], B [-128, 127]
    img_lab = img_lab.astype(np.float32)
    img_lab[:, :, 0] = (img_lab[:, :, 0] * (100.0 / 255.0)) - 50.0
    img_lab[:, :, 1:] = img_lab[:, :, 1:] - 128.0
    return img_lab


# Image should be in format HWC. Resize the entire image to the given size
# without worrying about the aspect ratio.
def global_resize(size, image, order='HWC'):
    img = cv2.resize(image, (size, size))
    return img


def HWC2CHW(image):
    if len(image.shape) == 3:
        return image.swapaxes(1, 2).swapaxes(0, 1)
    elif len(image.shape) == 4:
        # NHWC -> NCHW
        return image.swapaxes(2, 3).swapaxes(1, 2)
    else:
        logger.error('The image does not have correct dimensions')
        return None


def generate_prior_hull_files(xedges, gamut, opts):
    bins = [xedges, xedges]
    normalized_gamut = gamut / np.sum(gamut)
    pos_bins = np.where(normalized_gamut > 0.0)
    # the bins that have at least one pixel belonging to
    logger.info('Number of positive bins: {}'.format(pos_bins[0].shape))

    # get the positively occuring bins and save them
    x_pos, y_pos = list(bins[0][pos_bins[0]]), list(bins[1][pos_bins[1]])
    x_pos, y_pos = np.array(x_pos).reshape(-1, 1), np.array(y_pos).reshape(-1, 1)
    pts_in_hull = np.concatenate([x_pos, y_pos], axis=1)

    output_file = os.path.join(
        opts.output_path,
        'pts_in_hull_{}_{}_bin{}.npy'.format(opts.type, opts.dataset, opts.bin_size)
    )
    logger.info('Saving color bins to: {}'.format(output_file))
    np.save(output_file, pts_in_hull)

    # get the binds priors and save them
    priors = normalized_gamut[pos_bins]
    priors_file = os.path.join(
        opts.output_path,
        'priors_{}_{}_bin{}.npy'.format(opts.type, opts.dataset, opts.bin_size)
    )
    logger.info('Saving priors file: {} {}'.format(priors_file, priors.shape))
    np.save(priors_file, priors)
    logger.info('DONE!!!')


def compute_color_bins_frequency(opts):
    assert os.path.exists(opts.data_file), "Data file not found. Abort!"
    image_paths = np.load(opts.data_file, encoding='latin1')

    logger.info("Bin size: {}".format(opts.bin_size))
    xedges = np.arange(-128, 128, opts.bin_size)
    yedges = np.arange(-128, 128, opts.bin_size)
    histogram_bins = (xedges, yedges)

    final_gamut = None
    num_images = len(image_paths)
    logger.info('number of images: {}'.format((num_images)))
    logger.info('Generating color bins now....')
    for idx in range(num_images):
        img_path = image_paths[idx]
        try:
            img = cv2.imread(img_path)
            img = global_resize(opts.global_resize_img, img, order='HWC')
            img = convert2lab(img)
            img = HWC2CHW(img)[1:, :, :]
            x1, y1 = img[0, :], img[1, :]
            x1, y1 = x1.reshape(-1), y1.reshape(-1)
            H, edge1, edge2 = np.histogram2d(x1, y1, bins=histogram_bins)
            if final_gamut is None:
                final_gamut = H
            else:
                final_gamut += H
        except Exception:
            logger.info('difficulty getting image: {}'.format(img_path))
            pass
        if idx % 1000 == 0:
            logger.info('at: [{}/{}]'.format(idx, int(num_images)))
    logger.info('Final gamut shape: {}'.format(final_gamut.shape))
    return xedges, final_gamut


def main():
    parser = argparse.ArgumentParser(description='Generate color bins and priors')
    parser.add_argument('--dataset', type=str, default="imagenet",
                        help='What data is being used for color bins')
    parser.add_argument('--type', type=str, default="train",
                        help='data partition used for color bins')
    parser.add_argument('--data_file', type=str, default=None,
                        help='File containing info. about images')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path where bins and priors will be saved')
    parser.add_argument('--bin_size', type=int, default=10,
                        help='Bin size to use: 5/10/20')
    parser.add_argument('--global_resize_img', type=int, default=256,
                        help="Globally resize image before creating color bins")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    xedges, gamut = compute_color_bins_frequency(opts)
    generate_prior_hull_files(xedges, gamut, opts)


if __name__ == '__main__':
    main()
