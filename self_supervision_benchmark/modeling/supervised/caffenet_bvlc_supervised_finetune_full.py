# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
Caffenet BVLC model. Follows:
https://github.com/BVLC/caffe/tree/rc3/models/bvlc_alexnet

Relevant task: VOC07 full finetuning
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

from self_supervision_benchmark.core.config import config as cfg


def create_model(model, data, labels, split):
    num_classes = cfg.MODEL.NUM_CLASSES
    scale = 1. / cfg.NUM_DEVICES
    # for dropout
    test_mode = False
    if split in ['test', 'val']:
        test_mode = True

    ################################ conv1 ####################################
    conv1 = model.Conv(
        data, 'conv1', 3, 96, 11, stride=4,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    relu1 = model.Relu(conv1, conv1)

    ################################ pool1 #####################################
    lrn1 = model.LRN(relu1, 'norm1', size=5, alpha=0.0001, beta=0.75)
    pool1 = model.MaxPool(lrn1, 'pool1', kernel=3, stride=2)

    ################################ conv2 #####################################
    conv2 = model.Conv(
        pool1, 'conv2', 96, 256, 5,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}), pad=2, group=2,
    )
    relu2 = model.Relu(conv2, conv2)

    ################################ pool2 #####################################
    lrn2 = model.LRN(relu2, 'norm2', size=5, alpha=0.0001, beta=0.75)
    pool2 = model.MaxPool(lrn2, 'pool2', kernel=3, stride=2)

    ################################ conv3 #####################################
    conv3 = model.Conv(
        pool2, 'conv3', 256, 384, 3,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}), pad=1,
    )
    relu3 = model.Relu(conv3, conv3)

    ################################ conv4 #####################################
    conv4 = model.Conv(
        relu3, 'conv4', 384, 384, 3,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}), pad=1, group=2
    )
    relu4 = model.Relu(conv4, conv4)

    ################################ conv5 #####################################
    conv5 = model.Conv(
        relu4, 'conv5', 384, 256, 3,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}), pad=1, group=2
    )
    relu5 = model.Relu(conv5, conv5)

    ################################ pool5 #####################################
    blob_out = model.MaxPool(relu5, 'pool5', kernel=3, stride=2)

    ############################### fc6/fc7/fc8 ################################
    # fc6 / Dropout
    fc6 = model.FC(
        blob_out, 'fc6', 256 * 6 * 6, 4096,
        weight_init=('GaussianFill', {'std': 0.005}),
        bias_init=('ConstantFill', {'value': cfg.MODEL.FC_INIT_STD}),
    )
    blob_out = model.Relu(fc6, fc6)
    if split not in ['test', 'val']:
        blob_out = model.Dropout(
            blob_out, blob_out + '_dropout', ratio=0.5, is_test=test_mode
        )

    # fc7 / Dropout
    fc7 = model.FC(
        blob_out, 'fc7', 4096, 4096,
        weight_init=('GaussianFill', {'std': 0.005}),
        bias_init=('ConstantFill', {'value': cfg.MODEL.FC_INIT_STD}),
    )
    blob_out = model.Relu(fc7, fc7)
    if split not in ['test', 'val']:
        blob_out = model.Dropout(
            blob_out, blob_out + '_dropout', ratio=0.5, is_test=test_mode
        )

    # fc8
    fc8 = model.FC(
        blob_out, 'fc8_cls', 4096, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )

    ################################ Sigmoid ###################################
    model.net.Alias(fc8, 'pred')
    sigmoid = model.net.Sigmoid('pred', 'sigmoid')
    scale = 1. / cfg.NUM_DEVICES
    if split == 'train':
        loss = model.net.SigmoidCrossEntropyLoss(
            [fc8, labels], 'loss', scale=scale
        )
    elif split in ['test', 'val']:
        loss = None
    return model, sigmoid, loss
