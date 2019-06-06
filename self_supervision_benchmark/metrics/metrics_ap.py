# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This script is used to compute the TopK metrics reported during various transfer
tasks. Relevant transfer tasks: Full finetuning on VOC/COCO datasets.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import numpy as np
import datetime
import logging
from sklearn import metrics

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.metrics import metrics_helper
from self_supervision_benchmark.utils import helpers

from caffe2.python import workspace

# create the logger
logger = logging.getLogger(__name__)


class APMetricsCalculator():

    def __init__(self, model, split, batch_size, prefix):
        self.model = model
        self.split = split
        self.prefix = prefix
        self.batch_size = batch_size
        self.predictions = []
        self.ground_truth = []
        self.mAP = -1.0
        self.best_mAP = -1.0
        self.reset()

    def reset(self):
        # this should clear out all the metrics computed so far except the
        # best_topN metrics
        logger.info('Resetting {} metrics...'.format(self.split))
        self.split_loss = 0.0
        self.predictions = []
        self.ground_truth = []
        self.split_N = 0
        self.mAP = -1.0

    def finalize_metrics(self):
        self.split_loss /= self.split_N
        self.mAP = compute_mAP(self.predictions, self.ground_truth)
        if self.mAP > self.best_mAP:
            self.best_mAP = self.mAP

    def get_computed_metrics(self):
        json_stats = {}
        if self.split == 'train':
            json_stats['train_mAP'] = round(self.mAP, 3)
            json_stats['train_best_mAP'] = round(self.best_mAP, 3)
        elif self.split in ['test', 'val']:
            json_stats['test_mAP'] = round(self.mAP, 3)
            json_stats['test_best_mAP'] = round(self.best_mAP, 3)
        return json_stats

    # this is valid only for the test data
    def log_best_model_metrics(self, model_iter, total_iters):
        best_model_metrics_str = self.get_best_model_metrics_str()
        print(best_model_metrics_str.format(
            model_iter + 1, total_iters, self.best_mAP
        ))

    def compute_and_log_epoch_best_metric(self, model_iter):
        if self.mAP > self.best_mAP:
            self.best_mAP = self.mAP
            print('\n* Best model: mAP: {:7.3f}\n'.format(self.best_top1))
        # best_topN is calculated only when training happens
        final_metrics = self.get_epoch_metrics_log_str()
        print(final_metrics.format(
            model_iter + 1, cfg.SOLVER.NUM_ITERATIONS, self.mAP)
        )

    def get_split_mAP(self):
        # split_mAP = None
        mAP, predictions, truth = compute_multi_device_mAP(split=self.split)
        self.predictions.append(predictions)
        self.ground_truth.append(truth)
        self.mAP = compute_mAP(self.predictions, self.ground_truth)
        return mAP

    def calculate_and_log_train_iter_metrics(
        self, curr_iter, timer, rem_iters, total_iters, mb_size=None,
        db_loader=None, input_db=None
    ):
        self.lr = float(workspace.FetchBlob(self.prefix + '/lr'))
        split_loss = metrics_helper.sum_multi_device_blob('loss')
        self.split_loss += (split_loss * self.batch_size)
        self.split_N += self.batch_size
        mAP = self.get_split_mAP()
        if (curr_iter + 1) % cfg.LOGGER_FREQUENCY == 0:
            eta_seconds = timer.average_time * rem_iters
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            log_str = self.get_iter_metrics_log_str(self.split)
            print(log_str.format(
                eta, self.lr, curr_iter + 1, total_iters, timer.diff,
                timer.average_time, split_loss, mAP, mb_size
            ))

    def calculate_and_log_test_iter_metrics(
        self, curr_iter, timer, rem_iters, total_iters, mb_size=None,
        db_loader=None, input_db=None
    ):
        self.lr = cfg.SOLVER.BASE_LR
        self.split_N += self.batch_size
        mAP = self.get_split_mAP()
        if (curr_iter + 1) % cfg.LOGGER_FREQUENCY == 0:
            eta_seconds = timer.average_time * rem_iters
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            log_str = self.get_iter_metrics_log_str(self.split)
            print(log_str.format(
                eta, self.lr, curr_iter + 1, total_iters, timer.diff,
                timer.average_time, mAP, self.mAP
            ))

    def get_best_model_metrics_str(self):
        best_model_metrics_str = '\n* Best val model [{}|{}]:\t'
        best_model_metrics_str += ' mAP: {:7.3f}'
        return best_model_metrics_str

    def get_epoch_metrics_log_str(self):
        final_output_str = '\n* Finished #iters [{}|{}]:\t'
        final_output_str += ' mAP: {:7.3f}'
        return final_output_str

    def get_iter_metrics_log_str(self, split):
        train_str = ' '.join((
            '| Train ETA: {} LR: {} Iters [{}/{}] Time {:0.3f} AvgTime {:0.3f}',
            ' Loss {:7.4f}'
        ))
        test_str = '| Test ETA: {} LR: {} [{}/{}] Time {:0.3f} AvgTime {:0.3f}'
        train_str += ' mAP {}'
        test_str += '  mAP {} ({})'
        if split == 'train':
            train_str += ' mb_size: {}'
            return train_str
        else:
            return test_str


def compute_mAP(preds, gts):
    preds = np.concatenate(preds, axis=0).T
    gts = np.concatenate(gts, axis=0).T
    aps = []
    for i in range(gts.shape[0]):
        # 0 = not present, 1 = present, -1 = ignore
        ap = metrics.average_precision_score(
            gts[i][gts[i] >= 0],
            preds[i][gts[i] >= 0] - (1e-5 * gts[i][gts[i] >= 0])
        )
        aps.append(ap)
    # it is possible that a class has seen no elements and can have nan AP.
    mAP = np.nanmean(aps)
    return mAP


def compute_multi_device_mAP(split):
    predictions, truth = [], []
    device_prefix, _ = helpers.get_prefix_and_device()
    for idx in range(0, cfg.NUM_DEVICES):
        prefix = '{}{}'.format(device_prefix, idx)
        preds = workspace.FetchBlob(prefix + '/pred')
        if cfg.TEST.TEN_CROP and split in ['test', 'val']:
            preds = np.reshape(
                preds, (int(preds.shape[0] / 10), 10, preds.shape[1]))
            preds = np.sum(preds, axis=1)
        labels = workspace.FetchBlob(prefix + '/labels')
        batch_size = preds.shape[0]
        assert labels.shape[0] == batch_size, \
            "Something went wrong with data loading"
        predictions.append(preds)
        truth.append(labels)
    mAP = compute_mAP(predictions, truth)
    # return the concatenated output (batch_size * num_gpus) x 20
    cat_preds = np.concatenate(predictions, axis=0)
    cat_gts = np.concatenate(truth, axis=0)
    return mAP, cat_preds, cat_gts
