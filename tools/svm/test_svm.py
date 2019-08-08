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
import pickle
import six
import sys

import svm_helper

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_chosen_costs(opts, num_classes):
    costs_list = svm_helper.parse_cost_list(opts.costs_list)
    train_ap_matrix = np.zeros((num_classes, len(costs_list)))
    for cls in range(num_classes):
        for cost_idx in range(len(costs_list)):
            cost = costs_list[cost_idx]
            _, ap_out_file = svm_helper.get_svm_train_output_files(
                cls, cost, opts.output_path
            )
            train_ap_matrix[cls][cost_idx] = float(
                np.load(ap_out_file, encoding='latin1')[0]
            )
    argmax_cls = np.argmax(train_ap_matrix, axis=1)
    chosen_cost = [costs_list[idx] for idx in argmax_cls]
    logger.info('chosen_cost: {}'.format(chosen_cost))
    np.save(
        os.path.join(opts.output_path, 'crossval_ap.npy'),
        np.array(train_ap_matrix)
    )
    np.save(
        os.path.join(opts.output_path, 'chosen_cost.npy'),
        np.array(chosen_cost)
    )
    logger.info('saved crossval_ap AP to file: {}'.format(
        os.path.join(opts.output_path, 'crossval_ap.npy')))
    logger.info('saved chosen costs to file: {}'.format(
        os.path.join(opts.output_path, 'chosen_cost.npy')))
    return np.array(chosen_cost)


def test_svm(opts):
    assert os.path.exists(opts.data_file), "Data file not found. Abort!"
    features, targets = svm_helper.load_input_data(
        opts.data_file, opts.targets_data_file
    )
    # normalize the features: N x 9216 (example shape)
    features = svm_helper.normalize_features(features)
    num_classes = targets.shape[1]
    logger.info('Num classes: {}'.format(num_classes))

    # get the chosen cost that maximizes the cross-validation AP per class
    costs_list = get_chosen_costs(opts, num_classes)

    ap_matrix = np.zeros((num_classes, 1))
    for cls in range(num_classes):
        cost = costs_list[cls]
        logger.info('Testing model for cls: {} cost: {}'.format(cls, cost))
        model_file = os.path.join(
            opts.output_path, 'cls' + str(cls) + '_cost' + str(cost) + '.pickle'
        )
        with open(model_file, 'rb') as fopen:
            if six.PY2:
                model = pickle.load(fopen)
            else:
                model = pickle.load(fopen, encoding='latin1')
        prediction = model.decision_function(features)
        cls_labels = targets[:, cls]
        # meaning of labels in VOC/COCO original loaded target files:
        # label 0 = not present, set it to -1 as svm train target
        # label 1 = present. Make the svm train target labels as -1, 1.
        evaluate_data_inds = (targets[:, cls] != -1)
        eval_preds = prediction[evaluate_data_inds]
        eval_cls_labels = cls_labels[evaluate_data_inds]
        eval_cls_labels[np.where(eval_cls_labels == 0)] = -1
        P, R, score, ap = svm_helper.get_precision_recall(
            eval_cls_labels, eval_preds
        )
        ap_matrix[cls][0] = ap
    logger.info('Mean AP: {}'.format(np.mean(ap_matrix, axis=0)))
    np.save(os.path.join(opts.output_path, 'test_ap.npy'), np.array(ap_matrix))
    logger.info('saved test AP to file: {}'.format(
        os.path.join(opts.output_path, 'test_ap.npy')))


def main():
    parser = argparse.ArgumentParser(description='SVM model test')
    parser.add_argument('--data_file', type=str, default=None,
                        help="Numpy file containing image features and labels")
    parser.add_argument('--targets_data_file', type=str, default=None,
                        help="Numpy file containing image labels")
    parser.add_argument('--costs_list', type=str, default="0.01,0.1",
                        help="comma separated string containing list of costs")
    parser.add_argument('--output_path', type=str, default=None,
                        help="path where trained SVM models are saved")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    logger.info(opts)
    test_svm(opts)


if __name__ == '__main__':
    main()
