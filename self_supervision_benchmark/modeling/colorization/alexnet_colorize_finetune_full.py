# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
AlexNet model for testing feature quality of Colorization approach pre-training.
Transfer Task - Full Finetuning on VOC07 dataset.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division


from self_supervision_benchmark.core.config import config as cfg


def create_model(model, data, labels, split):
    num_classes = cfg.MODEL.NUM_CLASSES
    test_mode = False
    if split in ['test', 'val']:
        test_mode = True

    if cfg.MODEL.FC_INIT_TYPE == 'gaussian':
        fc_weight_init = ('GaussianFill', {'std': 0.01})
        fc_bias_init = ('ConstantFill', {'value': 0.})
    elif cfg.MODEL.FC_INIT_TYPE == 'xavier':
        fc_weight_init = ('XavierFill', {})
        fc_bias_init = ('ConstantFill', {'value': 0.})

    ################################ Split #####################################
    # split the input LAB channels into L and AB. The input to the colorization
    # models is always L channel (even during transfer tasks).
    model.net.Split(data, ['data_l', 'data_ab'], axis=1, split=[1, 2])

    ################################ conv1 #####################################
    model.Conv(
        'data_l', 'conv1', 1, 96, 11, stride=4,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}),
    )
    bn1 = model.SpatialBN(
        'conv1', 'conv1norm', 96, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn1 + '_s'], bn1 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn1 + '_b'], bn1 + '_b', value=0.0)
    model.Relu('conv1norm', 'conv1norm')

    ################################ pool1 #####################################
    model.MaxPool('conv1norm', 'pool1', kernel=3, stride=2)

    ################################ conv2 #####################################
    model.Conv(
        'pool1', 'conv2', 96, 256, 5, stride=1, pad=2, group=2,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}),
    )
    bn2 = model.SpatialBN(
        'conv2', 'conv2norm', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn2 + '_s'], bn2 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn2 + '_b'], bn2 + '_b', value=0.0)
    model.Relu('conv2norm', 'conv2norm')
    model.MaxPool('conv2norm', 'pool2', kernel=3, stride=2)

    ################################ conv3 #####################################
    model.Conv(
        'pool2', 'conv3', 256, 384, 3, stride=1, pad=1,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}),
    )
    bn3 = model.SpatialBN(
        'conv3', 'conv3norm', 384, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn3 + '_s'], bn3 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn3 + '_b'], bn3 + '_b', value=0.0)
    model.Relu('conv3norm', 'conv3norm')

    ################################ conv4 #####################################
    model.Conv(
        'conv3norm', 'conv4', 384, 384, 3, stride=1, pad=1, group=2,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}),
    )
    bn4 = model.SpatialBN(
        'conv4', 'conv4norm', 384, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn4 + '_s'], bn4 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn4 + '_b'], bn4 + '_b', value=0.0)
    model.Relu('conv4norm', 'conv4norm')

    ################################ conv5 #####################################
    model.Conv(
        'conv4norm', 'conv5', 384, 256, 3, stride=1, pad=1, group=2,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}),
    )
    bn5 = model.SpatialBN(
        'conv5', 'conv5norm', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn5 + '_s'], bn5 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn5 + '_b'], bn5 + '_b', value=0.0)
    model.Relu('conv5norm', 'conv5norm')
    model.MaxPool('conv5norm', 'pool5', kernel=3, stride=2)

    ################################# fc6 ######################################
    fc6 = model.Conv(
        'pool5', 'fc6_1', 256, 4096, 6, stride=1,
        weight_init=fc_weight_init, bias_init=fc_bias_init,
    )
    bn6 = model.SpatialBN(
        fc6, fc6 + '_norm', 4096, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn6 + '_s'], bn6 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn6 + '_b'], bn6 + '_b', value=0.0)
    blob_out = model.Relu(bn6, bn6)

    ################################# drop6 ####################################
    if split not in ['test', 'val']:
        blob_out = model.Dropout(
            blob_out, blob_out + '_dropout', ratio=0.5, is_test=False
        )

    ################################# fc7 ######################################
    fc7 = model.Conv(
        blob_out, 'fc7_1', 4096, 4096, 1, stride=1,
        weight_init=fc_weight_init, bias_init=fc_bias_init,
    )
    bn7 = model.SpatialBN(
        fc7, fc7 + '_norm', 4096, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn7 + '_s'], bn7 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn7 + '_b'], bn7 + '_b', value=0.0)
    blob_out = model.Relu(bn7, bn7)

    ################################# drop7 ####################################
    if split not in ['test', 'val']:
        blob_out = model.Dropout(
            blob_out, blob_out + '_dropout', ratio=0.5, is_test=False
        )

    ################################## fc8 #####################################
    fc8 = model.FC(
        blob_out, 'fc8_cls', 4096, num_classes,
        weight_init=fc_weight_init, bias_init=fc_bias_init,
    )

    ################################ Sigmoid ###################################
    model.net.Alias(fc8, 'pred')
    sigmoid = model.net.Sigmoid('pred', 'sigmoid')
    scale = 1. / cfg.NUM_DEVICES
    if split == 'train':
        loss = model.net.SigmoidCrossEntropyLoss(
            ['fc8_cls', labels], 'loss', scale=scale
        )
    elif split in ['test', 'val']:
        loss = None
    return model, sigmoid, loss
