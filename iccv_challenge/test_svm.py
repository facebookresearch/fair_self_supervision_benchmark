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
import logging
import numpy as np
import os
import sys


# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def calculate_ap(rec, prec):
    """
    Computes the AP under the precision recall curve.
    """
    rec, prec = rec.reshape(rec.size, 1), prec.reshape(prec.size, 1)
    z, o = np.zeros((1, 1)), np.ones((1, 1))
    mrec, mpre = np.vstack((z, rec, o)), np.vstack((z, prec, z))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = 0
    for i in indices:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def get_precision_recall(targets, preds):
    """
    [P, R, score, ap] = get_precision_recall(targets, preds)
    Input    :
        targets  : number of occurrences of this class in the ith image
        preds    : score for this image
    Output   :
        P, R   : precision and recall
        score  : score which corresponds to the particular precision and recall
        ap     : average precision
    """
    # binarize targets
    targets = np.array(targets > 0, dtype=np.float32)
    tog = np.hstack((
        targets[:, np.newaxis].astype(np.float64),
        preds[:, np.newaxis].astype(np.float64)
    ))
    ind = np.argsort(preds)
    ind = ind[::-1]
    score = np.array([tog[i, 1] for i in ind])
    sortcounts = np.array([tog[i, 0] for i in ind])

    tp = sortcounts
    fp = sortcounts.copy()
    for i in range(sortcounts.shape[0]):
        if sortcounts[i] >= 1:
            fp[i] = 0.
        elif sortcounts[i] < 1:
            fp[i] = 1.
    P = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
    numinst = np.sum(targets)
    R = np.cumsum(tp) / numinst
    ap = calculate_ap(R, P)
    return P, R, score, ap


def load_json(file_path, ground_truth=True):
    import json
    assert os.path.exists(file_path), "{} does not exist".format(file_path)
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    img_ids = list(data.keys())
    cls_names = list(data[img_ids[0]].keys())
    if ground_truth:
        output = np.empty((len(img_ids), len(cls_names)), dtype=np.int32)
    else:
        output = np.empty((len(img_ids), len(cls_names)), dtype=np.float64)
    for idx in range(len(img_ids)):
        for cls_idx in range(len(cls_names)):
            output[idx][cls_idx] = data[img_ids[idx]][cls_names[cls_idx]]
    return output, img_ids, cls_names


def test_svm(targets, predictions):
    num_classes = targets.shape[1]
    ap_matrix = np.zeros((num_classes, 1))
    for cls in range(num_classes):
        logger.info('Testing model for cls: {}'.format(cls))
        prediction = predictions[:, cls]
        cls_labels = targets[:, cls]
        # meaning of labels in VOC/COCO original loaded target files:
        # label 0 = not present, set it to -1 as svm train target
        # label 1 = present. Make the svm train target labels as -1, 1.
        evaluate_data_inds = (targets[:, cls] != -1)
        eval_preds = prediction[evaluate_data_inds]
        eval_cls_labels = cls_labels[evaluate_data_inds]
        eval_cls_labels[np.where(eval_cls_labels == 0)] = -1
        P, R, score, ap = get_precision_recall(eval_cls_labels, eval_preds)
        ap_matrix[cls][0] = ap
    logger.info('Mean AP: {}'.format(np.mean(ap_matrix, axis=0)))


def main():
    parser = argparse.ArgumentParser(description='SVM model test')
    parser.add_argument('--predictions', type=str, default=None,
                        help="Json file containing ground predictions")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    logger.info(opts)
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    targets_file = os.path.join(curr_dir, 'voc2007', 'test_images.json')
    targets, _, _ = load_json(targets_file)
    predictions, _, _ = load_json(opts.predictions, ground_truth=False)
    assert targets.shape == predictions.shape, "Predictions do not match targets shape"
    test_svm(targets, predictions)


if __name__ == '__main__':
    main()
