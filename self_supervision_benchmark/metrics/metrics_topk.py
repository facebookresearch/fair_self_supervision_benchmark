# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This script is used to compute the TopK metrics reported during various transfer
tasks. Relevant transfer tasks: linear classifiers for image classification.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import datetime
import logging
import numpy as np

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.metrics import metrics_helper
from self_supervision_benchmark.utils import helpers

from caffe2.python import workspace

# create the logger
logger = logging.getLogger(__name__)


class TopkMetricsCalculator():

    def __init__(self, model, split, batch_size, prefix, generate_json=0):
        self.model = model
        self.split = split
        self.prefix = prefix
        self.batch_size = batch_size
        self.generate_json = generate_json
        self.best_top1 = self.init_best_err()
        self.best_top5 = self.init_best_err()
        self.json_predictions = {}
        self.reset()

    def init_best_err(self):
        best_err = {}
        for item in cfg.ACCURACY_BLOBS:
            best_err[item] = float('inf')
        return best_err

    def reset(self):
        # this should clear out all the metrics computed so far except the
        # best_topN metrics
        logger.info('Resetting {} metrics...'.format(self.split))
        self.split_N = 0
        self.split_loss, self.split_err, self.split_err5 = {}, {}, {}
        for blob in cfg.LOSS_BLOBS:
            self.split_loss[blob] = 0.0
        for blob in cfg.ACCURACY_BLOBS:
            self.split_err[blob] = 0.0
            self.split_err5[blob] = 0.0
            self.json_predictions[blob] = {}

    def finalize_metrics(self):
        for item in (self.split_loss.keys()):
            self.split_loss[item] /= self.split_N
        for item in self.split_err.keys():
            self.split_err[item] /= self.split_N
            self.split_err5[item] /= self.split_N

    def get_json_predictions(self):
        return self.json_predictions

    def make_round(self, to_round):
        rounded = {}
        for item in to_round.keys():
            rounded[item] = round(to_round[item], 3)
        return rounded

    def get_accuracy_from_err(self, err_dict):
        accuracy = {}
        for item in err_dict.keys():
            accuracy[item] = round(100 - err_dict[item], 3)
        return accuracy

    def compare_curr_and_best_err(self, curr_err, best_err):
        compared_err = {}
        for item in curr_err.keys():
            if curr_err[item] < best_err[item]:
                compared_err[item] = curr_err[item]
            else:
                compared_err[item] = best_err[item]
        return compared_err

    def get_computed_metrics(self):
        json_stats = {}
        if self.split == 'train':
            json_stats['train_loss'] = self.make_round(self.split_loss)
            json_stats['train_accuracy'] = self.get_accuracy_from_err(
                self.split_err
            )
            json_stats['train_err'] = self.make_round(self.split_err)
            json_stats['train_err5'] = self.make_round(self.split_err5)
        elif self.split in ['test', 'val']:
            json_stats['test_accuracy'] = self.get_accuracy_from_err(
                self.split_err
            )
            json_stats['test_err'] = self.make_round(self.split_err)
            json_stats['test_err5'] = self.make_round(self.split_err5)
            json_stats['best_accuracy'] = self.get_accuracy_from_err(
                self.best_top1
            )
            json_stats['best_err'] = self.make_round(self.best_top1)
            json_stats['best_err5'] = self.make_round(self.best_top5)
        return json_stats

    # this is valid only for the test data
    def log_best_model_metrics(self, model_iter, total_iters):
        best_model_metrics_str = self.get_best_model_metrics_str()
        print(best_model_metrics_str.format(
            model_iter + 1, total_iters, self.make_round(self.best_top1),
            self.make_round(self.best_top5)
        ))

    # This prints the metrics of an epoch
    def compute_and_log_epoch_best_metric(self, model_iter):
        self.best_top1 = self.compare_curr_and_best_err(
            self.split_err, self.best_top1
        )
        self.best_top5 = self.compare_curr_and_best_err(
            self.split_err5, self.best_top5
        )
        print('\n* Best model: top1_err: {} top5_err: {}\n'.format(
            self.make_round(self.best_top1), self.make_round(self.best_top5)
        ))
        # best_topN is calculated only when training happens
        final_metrics = self.get_epoch_metrics_log_str()
        print(final_metrics.format(
            model_iter + 1, cfg.SOLVER.NUM_ITERATIONS,
            self.make_round(self.split_err), self.make_round(self.split_err5)
        ))

    def get_split_err(self, input_db):
        blobs_split_err, blobs_split_err5 = {}, {}
        blobs_accuracy_metrics, blobs_accuracy5_metrics, out_json = {}, {}, {}
        for blob in cfg.ACCURACY_BLOBS:
            blobs_accuracy_metrics[blob], out_json[blob] = compute_multi_device_topk_accuracy(
                blob, top_k=1, split=self.split, input_db=input_db,
                generate_json=self.generate_json,
                out_json_pred=self.json_predictions[blob],
            )
            blobs_accuracy5_metrics[blob], _ = compute_multi_device_topk_accuracy(
                blob, top_k=5, split=self.split, input_db=input_db
            )
            split_err = (1.0 - blobs_accuracy_metrics[blob]) * 100
            split_err5 = (1.0 - blobs_accuracy5_metrics[blob]) * 100
            blobs_split_err[blob] = split_err
            blobs_split_err5[blob] = split_err5
            self.split_err[blob] += (split_err * self.batch_size)
            self.split_err5[blob] += (split_err5 * self.batch_size)
            if self.generate_json:
                self.json_predictions[blob] = out_json[blob]
        return blobs_split_err, blobs_split_err5

    def calculate_and_log_train_iter_metrics(
        self, curr_iter, timer, rem_iters, total_iters, mb_size=None,
        db_loader=None, input_db=None
    ):
        self.split_N += self.batch_size
        self.lr = round(float(workspace.FetchBlob(self.prefix + '/lr')), 5)
        loss_blob_names = cfg.LOSS_BLOBS
        blobs_loss = {}
        for blob in loss_blob_names:
            split_loss = metrics_helper.sum_multi_device_blob(blob)
            self.split_loss[blob] += (split_loss * self.batch_size)
            blobs_loss[blob] = split_loss
        blobs_split_err, blobs_split_err5 = self.get_split_err(input_db)
        if (curr_iter + 1) % cfg.LOGGER_FREQUENCY == 0:
            eta_seconds = timer.average_time * rem_iters
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            log_str = self.get_iter_metrics_log_str(self.split)
            print(log_str.format(
                eta, self.lr, curr_iter + 1, total_iters, timer.diff,
                timer.average_time, self.make_round(blobs_loss),
                self.make_round(blobs_split_err), mb_size
            ))

    def calculate_and_log_test_iter_metrics(
        self, curr_iter, timer, rem_iters, total_iters, mb_size=None,
        db_loader=None, input_db=None
    ):
        self.lr = cfg.SOLVER.BASE_LR
        self.split_N += self.batch_size
        blobs_split_err, blobs_split_err5 = self.get_split_err(input_db)
        if (curr_iter + 1) % cfg.LOGGER_FREQUENCY == 0:
            eta_seconds = timer.average_time * rem_iters
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            log_str = self.get_iter_metrics_log_str(self.split)
            print(log_str.format(
                eta, self.lr, curr_iter + 1, total_iters, timer.diff,
                timer.average_time, self.make_round(blobs_split_err),
            ))

    def get_best_model_metrics_str(self):
        best_model_metrics_str = '\n* Best val model [{}|{}]:\t'
        best_model_metrics_str += ' top1_err: {} top5_err: {}'
        return best_model_metrics_str

    def get_epoch_metrics_log_str(self):
        final_output_str = '\n* Finished #iters [{}|{}]:\t'
        final_output_str += ' top1_err: {} top5_err: {}'
        return final_output_str

    def get_iter_metrics_log_str(self, split):
        train_str = ' '.join((
            '| Train ETA: {} LR: {} Iters [{}/{}] Time {:0.3f} AvgTime {:0.3f}',
            ' Losses {}, curr_top1_err {}, mb_size: {}'
        ))
        test_str = '| Test ETA: {} LR: {} [{}/{}] Time {:0.3f} AvgTime {:0.3f}'
        test_str += ', curr_top1_err {}'
        if split == 'train':
            return train_str
        else:
            return test_str


def compute_topk_accuracy(
    top_k, preds, labels, input_db=None, paths=None, generate_json=False,
    out_json_pred=None, indices=None
):
    batch_size = preds.shape[0]
    for i in range(batch_size):
        preds[i, :] = np.argsort(-preds[i, :])
    top_k_preds = preds[:, :top_k]
    pred_labels = []
    for j in range(batch_size):
        pred_labels.append(int(
            labels[j] in top_k_preds[j, :].astype(np.int32).tolist()
        ))
    correct = sum(pred_labels)
    if generate_json:
        paths, _, _ = input_db.get_minibatch_path_indexes(indices)
        for idx in range(len(indices)):
            img_id = '/'.join(paths[idx].split('/')[-2:])
            pred_lbl = int(top_k_preds[idx, 0])
            out_json_pred[img_id] = pred_lbl
    return (float(correct) / batch_size), out_json_pred


def compute_multi_device_topk_accuracy(
    blob_name, top_k, split, input_db=None, generate_json=False,
    out_json_pred=None
):
    top_k_accuracy = 0.0
    device_prefix, _ = helpers.get_prefix_and_device()
    for idx in range(0, cfg.NUM_DEVICES):
        prefix = '{}{}'.format(device_prefix, idx)
        softmax = workspace.FetchBlob(prefix + '/' + blob_name)
        indices = workspace.FetchBlob(prefix + '/' + 'db_indices')
        if cfg.TEST.TEN_CROP and split in ['test', 'val']:
            softmax = np.reshape(
                softmax, (int(softmax.shape[0] / 10), 10, softmax.shape[1]))
            softmax = np.mean(softmax, axis=1)
        labels = workspace.FetchBlob(prefix + '/labels')
        paths = None
        batch_size = softmax.shape[0]
        assert labels.shape[0] == batch_size, \
            "Something went wrong with data loading"
        acc, out_json_pred = compute_topk_accuracy(
            top_k, softmax.copy(), labels, input_db, paths, generate_json,
            out_json_pred, indices
        )
        top_k_accuracy += acc
    accuracy = float(top_k_accuracy) / cfg.NUM_DEVICES
    return accuracy, out_json_pred
