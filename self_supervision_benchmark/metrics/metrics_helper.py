# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
A simple helper class containing common functions used in metrics calculation.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division


import json

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.utils import helpers

from caffe2.python import workspace


def sum_multi_device_blob(blob_name):
    """Average values of a blob on each device"""
    value = 0
    prefix, _ = helpers.get_prefix_and_device()
    for idx in range(0, cfg.NUM_DEVICES):
        value += workspace.FetchBlob('{}{}/{}'.format(prefix, idx, blob_name))
    return value


def print_json_stats(stats):
    print('\njson_stats: {:s}\n'.format(json.dumps(stats, sort_keys=False)))
    return None


def get_json_stats_dict(
    train_metrics_calculator, test_metrics_calculator, curr_iter, model_flops,
    model_params,
):
    used_gpu_memory = None
    if cfg.DEVICE == 'GPU':
        used_gpu_memory = helpers.get_gpu_stats()
    num_epochs = int(cfg.SOLVER.NUM_ITERATIONS / cfg.TRAIN.EVALUATION_FREQUENCY)
    json_stats = {
        'evaluation_frequency': cfg.TRAIN.EVALUATION_FREQUENCY,
        'batchSize': cfg.TRAIN.BATCH_SIZE,
        'model_flops': model_flops,
        'model_params': model_params,
        'dataset': cfg.DATASET,
        'num_classes': cfg.MODEL.NUM_CLASSES,
        'momentum': cfg.SOLVER.MOMENTUM,
        'weightDecay': cfg.SOLVER.WEIGHT_DECAY,
        'nGPU': cfg.NUM_DEVICES,
        'LR': cfg.SOLVER.BASE_LR,
        'model_name': cfg.MODEL.MODEL_NAME,
        'bn_init': cfg.MODEL.BN_INIT_GAMMA,
        'tenCrop': cfg.TEST.TEN_CROP,
        'bn_momentum': cfg.MODEL.BN_MOMENTUM,
        'bn_epsilon': cfg.MODEL.BN_EPSILON,
        'current_learning_rate': train_metrics_calculator.lr,
        'currentIter': (curr_iter + 1),
        'epoch': (1 + int(curr_iter / cfg.TRAIN.EVALUATION_FREQUENCY)),
        'used_gpu_memory': used_gpu_memory,
        'nEpochs': num_epochs,
    }
    computed_train_metrics = train_metrics_calculator.get_computed_metrics()
    json_stats.update(computed_train_metrics)
    if test_metrics_calculator is not None:
        computed_test_metrics = test_metrics_calculator.get_computed_metrics()
        json_stats.update(computed_test_metrics)
    return json_stats
