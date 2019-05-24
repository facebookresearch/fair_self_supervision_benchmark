# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
SVM test for low shot image classification.

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
import pickle
import six
import sys

import svm_helper

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def test_svm_low_shot(opts):
    k_values = [int(val) for val in opts.k_values.split(",")]
    sample_inds = [int(val) for val in opts.sample_inds.split(",")]
    logger.info('Testing svm for k-values: {} and sample_inds: {}'.format(
        k_values, sample_inds))

    assert os.path.exists(opts.data_file), "Data file not found. Abort!"
    # we test the svms on the full test set. Given the test features and the
    # targets, we test it for various k-values (low-shot), cost values and
    # 5 independent samples.
    features, targets = svm_helper.load_input_data(
        opts.data_file, opts.targets_data_file
    )
    # normalize the features: N x 9216 (example shape)
    features = svm_helper.normalize_features(features)

    # parse the cost values for training the SVM on
    costs_list = svm_helper.parse_cost_list(opts.costs_list)
    logger.info('Testing SVM for costs: {}'.format(costs_list))

    # classes for which SVM testing should be done
    num_classes, cls_list = svm_helper.get_low_shot_svm_classes(
        targets, opts.dataset
    )

    # create the output for per sample, per k-value and per cost.
    sample_ap_matrices = []
    for _ in range(len(sample_inds)):
        ap_matrix = np.zeros((len(k_values), len(costs_list)))
        sample_ap_matrices.append(ap_matrix)

    # the test goes like this: For a given sample, for a given k-value and a
    # given cost value, we evaluate the trained svm model for all classes.
    # After computing over all classes, we get the mean AP value over all
    # classes. We hence end up with: output = [sample][k_value][cost]
    for inds in range(len(sample_inds)):
        sample_idx = sample_inds[inds]
        for k_idx in range(len(k_values)):
            k_low = k_values[k_idx]
            suffix = 'sample{}_k{}'.format(sample_idx + 1, k_low)
            for cost_idx in range(len(costs_list)):
                cost = costs_list[cost_idx]
                local_cost_ap = np.zeros((num_classes, 1))
                for cls in cls_list:
                    logger.info('Test sample/k_value/cost/cls: {}/{}/{}/{}'.format(
                        sample_idx + 1, k_low, cost, cls))
                    model_file = svm_helper.get_low_shot_output_file(
                        opts, cls, cost, suffix)
                    with open(model_file, 'rb') as fopen:
                        if six.PY2:
                            model = pickle.load(fopen)
                        else:
                            model = pickle.load(fopen, encoding='latin1')
                    prediction = model.decision_function(features)
                    eval_preds, eval_cls_labels = svm_helper.get_cls_feats_labels(
                        cls, prediction, targets, opts.dataset
                    )
                    P, R, score, ap = svm_helper.get_precision_recall(
                        eval_cls_labels, eval_preds
                    )
                    local_cost_ap[cls][0] = ap
                mean_cost_ap = np.mean(local_cost_ap, axis=0)
                sample_ap_matrices[inds][k_idx][cost_idx] = mean_cost_ap
            out_k_sample_file = os.path.join(
                opts.output_path,
                'test_ap_sample{}_k{}.npy'.format(sample_idx + 1, k_low)
            )
            save_data = sample_ap_matrices[inds][k_idx]
            save_data = save_data.reshape((1, -1))
            np.save(out_k_sample_file, save_data)
            logger.info('Saved sample test k_idx AP to file: {} {}'.format(
                out_k_sample_file, save_data.shape))
    logger.info('All done!!')


def main():
    parser = argparse.ArgumentParser(description='Low shot SVM model test')
    parser.add_argument('--data_file', type=str, default=None,
                        help="Numpy file containing image features and labels")
    parser.add_argument('--targets_data_file', type=str, default=None,
                        help="Numpy file containing image labels")
    parser.add_argument('--costs_list', type=str, default="0.01,0.1",
                        help="comma separated string containing list of costs")
    parser.add_argument('--output_path', type=str, default=None,
                        help="path where trained SVM models are saved")
    parser.add_argument('--k_values', type=str, default=None,
                        help="Low-shot k-values for svm testing. Comma separated")
    parser.add_argument('--sample_inds', type=str, default=None,
                        help="sample_inds for which to test svm. Comma separated")
    parser.add_argument('--dataset', type=str, default="voc",
                        help='voc | places')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    logger.info(opts)
    test_svm_low_shot(opts)


if __name__ == '__main__':
    main()
