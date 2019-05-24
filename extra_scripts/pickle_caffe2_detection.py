# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
Script for converting Caffe2 models into the the simple state dict format
used by Detectron in Caffe2.

Written by: Kaiming He
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import errno
import logging
import numpy as np
import os
import six
import sys
from six.moves import cPickle as pickle
from uuid import uuid4

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def save_object(obj, file_name, pickle_format=2):
    """
    Save a Python object by pickling it.
    Credits (copy here to avoid detectron dependency):
    https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/io.py#L39
    """
    file_name = os.path.abspath(file_name)
    # Avoid filesystem race conditions (particularly on network filesystems)
    # by saving to a random tmp file on the same filesystem, and then
    # atomically rename to the target filename.
    tmp_file_name = file_name + ".tmp." + uuid4().hex
    try:
        with open(tmp_file_name, 'wb') as f:
            pickle.dump(obj, f, pickle_format)
            f.flush()  # make sure it's written to disk
            os.fsync(f.fileno())
        os.rename(tmp_file_name, file_name)
    finally:
        # Clean up the temp file on failure. Rather than using os.path.exists(),
        # which can be unreliable on network filesystems, attempt to delete and
        # ignore os errors.
        try:
            os.remove(tmp_file_name)
        except EnvironmentError as e:  # parent class of IOError, OSError
            if getattr(e, 'errno', None) != errno.ENOENT:  # We expect ENOENT
                logger.info("Could not delete temp file %r",
                    tmp_file_name, exc_info=True)
                # pass through since we don't want the job to crash


def load_object(file_name):
    """
    Credits (copy here to avoid detectron dependency):
    https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/io.py#L72
    """
    with open(file_name, 'rb') as f:
        if six.PY2:
            return pickle.load(f)
        else:
            return pickle.load(f, encoding='latin1')


# decompose a bn layer into two AffineChannel layers
def decompose_spatial_bn_layers(c2cls_weights):
    finished_bn_blobs = []
    blobs = sorted(c2cls_weights['blobs'].keys())
    for blob in blobs:
        idx = blob.find('_bn_')
        if idx > -1:
            bn_layername = blob[:idx]
            if bn_layername not in finished_bn_blobs:
                finished_bn_blobs.append(bn_layername)
                logger.info('Converting BN: {}'.format(bn_layername))

                name_s = bn_layername + '_bn_s'  # gamma
                name_b = bn_layername + '_bn_b'  # beta
                name_rm = bn_layername + '_bn_rm'  # running mean
                name_rv = bn_layername + '_bn_riv'  # running var

                scale = c2cls_weights['blobs'][name_s]
                bias = c2cls_weights['blobs'][name_b]
                bn_mean = c2cls_weights['blobs'][name_rm]
                bn_var = c2cls_weights['blobs'][name_rv]

                bn_std = np.sqrt(bn_var + 1e-5)

                # rewrite
                del c2cls_weights['blobs'][name_rm]
                del c2cls_weights['blobs'][name_rv]

                # first affine layer:
                # y = (x - mean) / std
                # must be: y = s * x + b
                c2cls_weights['blobs'][bn_layername + '_bn_norm_s'] = 1 / bn_std
                c2cls_weights['blobs'][bn_layername + '_bn_norm_b'] = -bn_mean / bn_std

                # second affine layer
                c2cls_weights['blobs'][name_s] = scale  # bn_s
                c2cls_weights['blobs'][name_b] = bias  # bn_b


def remove_spatial_bn_layers(c2cls_weights):
    finished_bn_blobs = []
    blobs = sorted(c2cls_weights['blobs'].keys())
    for blob in blobs:
        idx = blob.find('_bn_')
        if idx > -1:
            bn_layername = blob[:idx]
            if bn_layername not in finished_bn_blobs:
                finished_bn_blobs.append(bn_layername)
                logger.info('Converting BN: {}'.format(bn_layername))

                name_s = bn_layername + '_bn_s'  # gamma
                name_b = bn_layername + '_bn_b'  # beta
                name_rm = bn_layername + '_bn_rm'  # running mean
                name_rv = bn_layername + '_bn_riv'  # running var

                scale = c2cls_weights['blobs'][name_s]
                bias = c2cls_weights['blobs'][name_b]
                bn_mean = c2cls_weights['blobs'][name_rm]
                bn_var = c2cls_weights['blobs'][name_rv]

                std = np.sqrt(bn_var + 1e-5)
                new_scale = scale / std
                new_bias = bias - bn_mean * scale / std

                # rewrite
                del c2cls_weights['blobs'][name_rm]
                del c2cls_weights['blobs'][name_rv]

                c2cls_weights['blobs'][name_s] = new_scale
                c2cls_weights['blobs'][name_b] = new_bias


def absorb_std_to_conv1(c2cls_weights):
    # ---------------------
    # absorb input std into conv1_w
    # in benchmark caffe2 pretext models: (raw) img: [0, 255]
    # data = (img / 255 - mean) / std
    # conv1 = data * conv1_w = (img / (255std) - mean / std) * conv1_w
    # -------
    # in Detectron: (raw) img: [0, 255]
    # data = img - (255mean)
    # conv1 = data * conv1_w'
    # ==> conv1_w' = conv1_w / (255std)
    # -------

    DATA_STD = [0.225, 0.224, 0.229]  # B, G, R in benchmark caffe2

    w = c2cls_weights['blobs']['conv1_w']  # (64, 3, 7, 7)
    assert (w.shape == (64, 3, 7, 7))

    # Note: the input conv1_w to this function is always BGR
    for i in range(3):
        w[:, i, :, :] /= (DATA_STD[i] * 255)
    logger.info('DATA_STD absorbed in conv1_w!')


def remove_non_param_fields(c2cls_weights):
    fields_to_del = ['epoch', 'model_iter', 'lr']
    for field in fields_to_del:
        if field in c2cls_weights['blobs']:
            del c2cls_weights['blobs'][field]


def remove_momentum(c2cls_weights):
    for k in list(c2cls_weights['blobs']):
        if k.endswith('_momentum'):
            del c2cls_weights['blobs'][k]


def load_and_convert_caffe2_cls_model(args):
    c2cls_weights = load_object(args.c2_model_file_name)
    remove_non_param_fields(c2cls_weights)

    if not args.keep_momentum:
        remove_momentum(c2cls_weights)

    if not args.decompose_bn:
        # convert bn
        remove_spatial_bn_layers(c2cls_weights)
    else:
        # decompose bn
        decompose_spatial_bn_layers(c2cls_weights)

    # absorb std (input to this should be BGR)
    if args.absorb_std:
        absorb_std_to_conv1(c2cls_weights)

    return c2cls_weights


def main():
    parser = argparse.ArgumentParser(
        description='Dump weights from a Caffe2 classification model')
    parser.add_argument('--c2_model', dest='c2_model_file_name', default=None,
        type=str, help='Pretrained network weights file path')
    parser.add_argument('--output_file', dest='out_file_name', default=None,
        type=str, help='Output file path')
    parser.add_argument('--decompose_bn', dest='decompose_bn', default=False,
        action="store_true", help='Decompose bn into two affine layers')
    parser.add_argument('--absorb_std', dest='absorb_std', default=False,
        action="store_true", help='Absorb input std into conv1_w')
    parser.add_argument('--keep_momentum', dest='keep_momentum', default=False,
        action='store_true', help='Keep blobs named X + "_momentum"')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    logger.info(args)
    assert os.path.exists(args.c2_model_file_name), \
        'Weights file does not exist'
    weights = load_and_convert_caffe2_cls_model(args)

    save_object(weights, args.out_file_name)
    logger.info('Wrote blobs to {}:'.format(args.out_file_name))
    logger.info(sorted(weights['blobs'].keys()))


if __name__ == '__main__':
    main()
