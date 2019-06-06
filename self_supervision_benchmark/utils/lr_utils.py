# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This script contains helper functions for Learning rate policy used.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import logging
import math
import numpy as np

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.utils import helpers

from caffe2.python import workspace, core

# create the logger
logger = logging.getLogger(__name__)


def create_multi_gpu_blob(blob_name):
    prefix, device = helpers.get_prefix_and_device()
    for idx in range(0, cfg.NUM_DEVICES):
        with core.DeviceScope(core.DeviceOption(device, idx)):
            workspace.CreateBlob('{}{}/{}'.format(prefix, idx, blob_name))


def create_learning_rate_blob():
    create_multi_gpu_blob('lr')


def get_lr_steps():
    # example input: [150150, 150150, 150150]
    split_lr_stepsizes = cfg.SOLVER.STEP_SIZES
    lr_iters = []
    lr_steps = [int(step) for step in split_lr_stepsizes]
    for idx in range(len(lr_steps)):
        if idx > 0:
            lr_iters.append(lr_steps[idx] + lr_iters[idx - 1])
        else:
            lr_iters.append(lr_steps[idx])
    # now we have [150150, 300300, 450450]
    return lr_iters


def scale_momentum(scale, model):
    """
    The MomentumSGDUpdate op implements the update V (a model parameter) as:

        V := mu * V + lr * grad,

    where mu is the momentum factor, lr is the learning rate, and grad is the
    stochastic gradient. Since V is not defined independently of the learning
    rate (as it should ideally be), when the learning rate is changed we should
    scale the update history V in order to make it compatible in scale with
    lr * grad.
    """
    # for the LR warm-up in distributed training, when we change the LR after
    # warm-up, then we need to update the momentum accordingly
    logger.info('Scaling momentum: {}'.format(scale))
    prefix, device = helpers.get_prefix_and_device()
    for idx in range(0, cfg.NUM_DEVICES):
        with core.DeviceScope(core.DeviceOption(device, idx)):
            with core.NameScope("{}{}".format(prefix, idx)):
                params = model.GetParams()
                for param in params:
                    op = core.CreateOperator(
                        'Scale', [param + '_momentum'], [param + '_momentum'],
                        scale=scale)
                    workspace.RunOperatorOnce(op)


def add_variable_stepsize_lr(
    curr_iter, num_devices, lr_iters, start_model_iter, epoch_iters=None,
    model=None, prev_checkpointed_lr=None,
):
    global CURRENT_LR
    # if the model is resumed from some checkpoint state, then we load the
    # checkpoint LR into the CURRENT_LR at the start of training only
    if prev_checkpointed_lr is not None and (curr_iter == start_model_iter):
        CURRENT_LR = prev_checkpointed_lr

    if curr_iter <= lr_iters[0]:
        gamma_pow = 0
    else:
        idx = 0
        while idx < len(lr_iters) and lr_iters[idx] < curr_iter:
            idx += 1
        gamma_pow = idx

    learning_rate = (cfg.SOLVER.BASE_LR * math.pow(cfg.SOLVER.GAMMA, gamma_pow))
    new_lr = learning_rate
    if curr_iter == 1:
        prev_lr = new_lr
    else:
        prev_lr = CURRENT_LR
    # import pdb; pdb.set_trace()
    if cfg.SOLVER.SCALE_MOMENTUM and (not new_lr == prev_lr):
        scale = new_lr / float(prev_lr)
        scale_momentum(scale, model)

    CURRENT_LR = new_lr
    prefix, device = helpers.get_prefix_and_device()
    for idx in range(0, num_devices):
        with core.DeviceScope(core.DeviceOption(device, idx)):
            workspace.FeedBlob(
                '{}{}/lr'.format(prefix, idx),
                np.array(learning_rate, dtype=np.float32)
            )
