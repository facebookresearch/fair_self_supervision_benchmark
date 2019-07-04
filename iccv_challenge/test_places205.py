# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
SVM test for image classification.

Relevant transfer tasks: Image Classification VOC07 and COCO2014.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import argparse
import json
import logging
import numpy as np
import os
import sys

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def load_json(file_path):
    assert os.path.exists(file_path), "{} does not exist".format(file_path)
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    img_ids = sorted(list(data.keys()))
    img_labels = []
    for item in img_ids:
        img_labels.append(int(data[item]))
    return img_ids, img_labels


def test_places205(targets, predictions):
    num_imgs = len(targets)
    pred_labels = []
    for j in range(num_imgs):
        pred_labels.append(int(int(targets[j]) == int(predictions[j])))
    correct = sum(pred_labels)
    accuracy = (float(correct) / num_imgs) * 100.0
    logger.info('Top-1 Accuracy: {}'.format(accuracy))
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Places205 model test')
    parser.add_argument('--predictions', type=str, default=None,
                        help="Json file containing ground predictions")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    logger.info(opts)
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    targets_file = os.path.join(curr_dir, 'places205', 'val_targets.json')
    _, targets = load_json(targets_file)
    _, predictions = load_json(opts.predictions)
    assert len(targets) == len(predictions), "Predictions do not match targets length"
    accuracy = test_places205(targets, predictions)


if __name__ == '__main__':
    main()
