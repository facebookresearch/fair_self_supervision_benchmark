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

Relevant transfer task: Linear Classifier on IN1K/Places datasets
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

from self_supervision_benchmark.core.config import config as cfg


def create_model(model, data, labels, split):
    num_classes = cfg.MODEL.NUM_CLASSES
    scale = 1. / cfg.NUM_DEVICES
    losses, softmax = [], None
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

    model.StopGradient('conv1', 'conv1')
    resize_c1 = model.AveragePool(
        relu1, relu1 + '_s4k19_resize', kernel=19, stride=4, pad=0
    )
    bn_c1 = model.SpatialBN(
        resize_c1, resize_c1 + '_bn', 96, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [bn_c1 + '_s'], bn_c1 + '_s', value=1.0
        )
        model.param_init_net.ConstantFill(
            [bn_c1 + '_b'], bn_c1 + '_b', value=0.0
        )
    fc_conv1 = model.FC(
        bn_c1, 'fc_c1', 96 * 10 * 10, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_conv1, 'pred_c1')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_conv1, labels], 'accuracy_c1')
        if split == 'train':
            softmax, loss_c1 = model.SoftmaxWithLoss(
                ['pred_c1', labels], ['softmax_c1', 'loss_c1'], scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax('pred_c1', 'softmax_c1', engine='CUDNN')
            loss_c1 = None
        losses.append(loss_c1)

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
    model.StopGradient('conv2', 'conv2')
    resize_c2 = model.AveragePool(
        relu2, relu2 + '_s3k12_resize', kernel=12, stride=3, pad=0
    )
    bn_c2 = model.SpatialBN(
        resize_c2, resize_c2 + '_bn', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [bn_c2 + '_s'], bn_c2 + '_s', value=1.0
        )
        model.param_init_net.ConstantFill(
            [bn_c2 + '_b'], bn_c2 + '_b', value=0.0
        )
    fc_conv2 = model.FC(
        bn_c2, 'fc_c2', 256 * 6 * 6, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_conv2, 'pred_c2')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_conv2, labels], 'accuracy_c2')
        if split == 'train':
            softmax, loss_c2 = model.SoftmaxWithLoss(
                ['pred_c2', labels], ['softmax_c2', 'loss_c2'], scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax('pred_c2', 'softmax_c2', engine='CUDNN')
            loss_c2 = None
        losses.append(loss_c2)

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
    model.StopGradient('conv3', 'conv3')
    resize_c3 = model.AveragePool(
        relu3, relu3 + '_s1k9_resize', kernel=9, stride=1, pad=0
    )
    bn_c3 = model.SpatialBN(
        resize_c3, resize_c3 + '_bn', 384, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [bn_c3 + '_s'], bn_c3 + '_s', value=1.0
        )
        model.param_init_net.ConstantFill(
            [bn_c3 + '_b'], bn_c3 + '_b', value=0.0
        )
    fc_conv3 = model.FC(
        bn_c3, 'fc_c3', 384 * 5 * 5, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_conv3, 'pred_c3')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_conv3, labels], 'accuracy_c3')
        if split == 'train':
            softmax, loss_c3 = model.SoftmaxWithLoss(
                ['pred_c3', labels], ['softmax_c3', 'loss_c3'], scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax('pred_c3', 'softmax_c3', engine='CUDNN')
            loss_c3 = None
        losses.append(loss_c3)

    ################################ conv4 #####################################
    conv4 = model.Conv(
        relu3, 'conv4', 384, 384, 3,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}), pad=1, group=2
    )
    relu4 = model.Relu(conv4, conv4)
    model.StopGradient('conv4', 'conv4')
    resize_c4 = model.AveragePool(
        relu4, relu4 + '_s1k9_resize', kernel=9, stride=1, pad=0
    )
    bn_c4 = model.SpatialBN(
        resize_c4, resize_c4 + '_bn', 384, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [bn_c4 + '_s'], bn_c4 + '_s', value=1.0
        )
        model.param_init_net.ConstantFill(
            [bn_c4 + '_b'], bn_c4 + '_b', value=0.0
        )
    fc_conv4 = model.FC(
        bn_c4, 'fc_c4', 384 * 5 * 5, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_conv4, 'pred_c4')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_conv4, labels], 'accuracy_c4')
        if split == 'train':
            softmax, loss_c4 = model.SoftmaxWithLoss(
                ['pred_c4', labels], ['softmax_c4', 'loss_c4'], scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax('pred_c4', 'softmax_c4', engine='CUDNN')
            loss_c4 = None
        losses.append(loss_c4)

    ################################ conv5 #####################################
    conv5 = model.Conv(
        relu4, 'conv5', 384, 256, 3,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}), pad=1, group=2
    )
    relu5 = model.Relu(conv5, conv5)
    model.StopGradient('conv5', 'conv5')
    resize_c5 = model.AveragePool(
        relu5, relu5 + '_s1k8_resize', kernel=8, stride=1, pad=0
    )
    bn_c5 = model.SpatialBN(
        resize_c5, resize_c5 + '_bn', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [bn_c5 + '_s'], bn_c5 + '_s', value=1.0
        )
        model.param_init_net.ConstantFill(
            [bn_c5 + '_b'], bn_c5 + '_b', value=0.0
        )
    fc_conv5 = model.FC(
        bn_c5, 'fc_c5', 256 * 6 * 6, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_conv5, 'pred_c5')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_conv5, labels], 'accuracy_c5')
        if split == 'train':
            softmax, loss_c5 = model.SoftmaxWithLoss(
                ['pred_c5', labels], ['softmax_c5', 'loss_c5'], scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax('pred_c5', 'softmax_c5', engine='CUDNN')
            loss_c5 = None
        losses.append(loss_c5)

    ################################ pool5 #####################################
    blob_out = model.MaxPool(relu5, 'pool5', kernel=3, stride=2)

    ################################# fc6/fc7 ##################################
    fc6 = model.FC(
        blob_out, 'fc6', 256 * 6 * 6, 4096,
        weight_init=('GaussianFill', {'std': 0.005}),
        bias_init=('ConstantFill', {'value': cfg.MODEL.FC_INIT_STD}),
    )
    blob_out = model.Relu(fc6, fc6)

    # fc7
    fc7 = model.FC(
        blob_out, 'fc7', 4096, 4096,
        weight_init=('GaussianFill', {'std': 0.005}),
        bias_init=('ConstantFill', {'value': cfg.MODEL.FC_INIT_STD}),
    )
    blob_out = model.Relu(fc7, fc7)
    model.StopGradient(blob_out, blob_out)

    return model, softmax, losses
