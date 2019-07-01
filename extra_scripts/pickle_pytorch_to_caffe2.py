# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
Given a ResNet-50 pretext model, we convert it to the weights that can be used
in Pytorch 1.0. This script provides an example conversion to the model
https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/resnet.py

For any other conversion, modify the script to suit your needs.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import argparse
import errno
import logging
import os
import pickle
import re
import six
import sys
import torch
from collections import OrderedDict
from uuid import uuid4

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def _rename_basic_resnet_weights(layer_keys):
    # Make C2 compatible - architecture:
    # https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/self_supervision_benchmark/modeling/supervised/resnet_supervised_finetune_full.py     # noqa B950
    layer_keys = [k.replace(".downsample.1.", ".branch1_bn.") for k in layer_keys]
    layer_keys = [k.replace(".downsample.0.", ".branch1.") for k in layer_keys]
    layer_keys = [k.replace(".conv1.", ".branch2a.") for k in layer_keys]
    layer_keys = [k.replace(".bn1.", ".branch2a_bn.") for k in layer_keys]
    layer_keys = [k.replace(".conv2.", ".branch2b.") for k in layer_keys]
    layer_keys = [k.replace(".bn2.", ".branch2b_bn.") for k in layer_keys]
    layer_keys = [k.replace(".conv3.", ".branch2c.") for k in layer_keys]
    layer_keys = [k.replace(".bn3.", ".branch2c_bn.") for k in layer_keys]
    layer_keys = [k.replace("layer1.", "res2.") for k in layer_keys]
    layer_keys = [k.replace("layer2.", "res3.") for k in layer_keys]
    layer_keys = [k.replace("layer3.", "res4.") for k in layer_keys]
    layer_keys = [k.replace("layer4.", "res5.") for k in layer_keys]

    layer_keys = [k.replace("bn1.", "conv1_bn.") for k in layer_keys]
    layer_keys = [k.replace("conv1_", "res.conv1_") for k in layer_keys]

    layer_keys = [k.replace( "_bn.weight", "_bn.scale") for k in layer_keys]
    layer_keys = [k.replace("_bn.scale", "_bn.s") for k in layer_keys]
    layer_keys = [k.replace("_bn.running_mean", "_bn.rm") for k in layer_keys]
    layer_keys = [k.replace("_bn.running_var", "_bn.riv") for k in layer_keys]
    layer_keys = [k.replace(".weight", ".w") for k in layer_keys]
    layer_keys = [k.replace(".bias", ".b") for k in layer_keys]

    layer_keys = [k.replace("_bn", ".bn") for k in layer_keys]
    layer_keys = [k.replace(".", "_") for k in layer_keys]

    return layer_keys

# res2_0_branch1_w, res3_0_branch1_w

def _rename_weights_for_resnet(weights):
    original_keys = sorted(weights.keys())
    layer_keys = sorted(weights.keys())

    # performs basic renaming: . -> _ , etc
    layer_keys = _rename_basic_resnet_weights(layer_keys)
    key_map = {k: v for k, v in zip(original_keys, layer_keys)}

    logger.info("Remapping PyTorch weights")
    max_pth_key_size = max([len(k) for k in original_keys])

    new_weights = OrderedDict()
    for k in original_keys:
        v = weights[k]
        if 'fc' in k:
            continue
        w = v.data.cpu().numpy()
        logger.info("PyTorch name: {: <{}} mapped name: {}".format(
            k, max_pth_key_size, key_map[k]))
        new_weights[key_map[k]] = w
    logger.info('Number of blobs: {}'.format(len(new_weights)))
    return new_weights


def _load_pth_pickled_weights(file_path):
    checkpoint = torch.load(file_path)
    if "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
    else:
        weights = checkpoint
    return weights


def convert_bgr2rgb(state_dict):
    w = state_dict['conv1_w']  # (64, 3, 7, 7)
    assert (w.shape == (64, 3, 7, 7))
    w = w[:, ::-1, :, :]
    state_dict['conv1_w'] = w.copy()
    logger.info('BGR ===> RGB for conv1_w.')
    return state_dict


def save_object(obj, file_name, pickle_format=2):
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
        logger.info('Saved: {}'.format(file_name))
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


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to C2")
    parser.add_argument('--pth_model', type=str, default=None,
                        help='Path to PyTorch RN-50 model')
    parser.add_argument('--output_model', type=str, default=None,
                        help='Path to save C2 RN-50 model')
    parser.add_argument('--arch', type=str, default="R-50",
                        help='R-50 | R-101 | R-152')
    parser.add_argument('--bgr2rgb', dest='bgr2rgb', default=False,
                        help='Revert bgr order to rgb order')
    args = parser.parse_args()

    # load the pytorch model first
    state_dict = _load_pth_pickled_weights(args.pth_model)

    # depending on the image reading library, we convert the weights to be
    # compatible order. The default order of caffe2 weights is BGR (openCV).
    if args.bgr2rgb:
        state_dict = convert_bgr2rgb(state_dict)

    blobs = _rename_weights_for_resnet(state_dict)
    blobs = dict(blobs=blobs)
    logger.info('Saving converted weights to: {}'.format(args.output_model))
    save_object(blobs, args.output_model)
    logger.info('Done!!')


if __name__ == '__main__':
    main()
