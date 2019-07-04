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
import numpy as np
import os
import sys
import zipfile
from uuid import uuid4


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
        # print('Testing model for cls: {}'.format(cls))
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
    mean_ap = np.mean(ap_matrix, axis=0)
    return mean_ap


def main():
    parser = argparse.ArgumentParser(description='SVM model test')
    parser.add_argument('--predictions', type=str, default=None,
                        help="Zip file containing low-shot predictions")
    parser.add_argument('--k_values', type=str, default="1,2,4,8,16,32,64,96",
                        help="Low-shot k-values for svm testing. Comma separated")
    parser.add_argument('--sample_inds', type=str, default="0,1,2,3,4",
                        help="sample_inds for which to test svm. Comma separated")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    print(opts)
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    targets_file = os.path.join(curr_dir, 'voc2007', 'test_targets.json')
    targets, _, _ = load_json(targets_file)

    target_dir = "/tmp/{}".format(uuid4().hex)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    print('Output dir: {}'.format(target_dir))
    with zipfile.ZipFile(opts.predictions, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    k_values = [int(val) for val in opts.k_values.split(",")]
    sample_inds = [int(val) for val in opts.sample_inds.split(",")]
    print('Testing svm for k-values: {} and sample_inds: {}'.format(
        k_values, sample_inds))

    output_mean, output_max, output_min, output_std = [], [], [], []
    for k_idx in range(len(k_values)):
        k_low = k_values[k_idx]
        k_val_output = []
        for inds in range(len(sample_inds)):
            sample_idx = sample_inds[inds]
            suffix = 'sample{}_k{}'.format(sample_idx + 1, k_low)
            print('Suffix: {}'.format(suffix))
            out_k_sample_file = os.path.join(
                target_dir,
                'test_sample{}_k{}_json_preds.json'.format(sample_idx + 1, k_low)
            )
            predictions, _, _ = load_json(out_k_sample_file, ground_truth=False)
            assert targets.shape == predictions.shape, "Predictions do not match targets shape"
            ap = test_svm(targets, predictions)
            k_val_output.append(ap)
        k_val_output = np.concatenate(k_val_output, axis=0)
        k_low_max = np.max(k_val_output, axis=0)
        k_low_min = np.min(k_val_output, axis=0)
        k_low_mean = np.mean(k_val_output, axis=0)
        k_low_std = np.std(k_val_output, axis=0)
        output_mean.append(k_low_mean)
        output_min.append(k_low_min)
        output_max.append(k_low_max)
        output_std.append(k_low_std)

    for idx in range(len(output_mean)):
        print('mean/min/max/std: {} / {} / {} / {}'.format(
            round(output_mean[idx], 2), round(output_min[idx], 2),
            round(output_max[idx], 2), round(output_std[idx], 2),
        ))
    final_mean = np.mean(output_mean, axis=0)
    print('final_mean: {}'.format(final_mean))
    print('All done!!')


if __name__ == '__main__':
    main()
