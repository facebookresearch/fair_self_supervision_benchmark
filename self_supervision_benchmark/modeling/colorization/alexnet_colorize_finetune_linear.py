# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
AlexNet model for testing feature quality of Colorization approach pre-training.
Transfer Task - Linear classifiers on ImageNet-1K/Places205 datasets.
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
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=True,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn1 + '_s'], bn1 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn1 + '_b'], bn1 + '_b', value=0.0)
    relu1 = model.Relu('conv1norm', 'conv1norm')
    model.StopGradient('conv1norm', 'conv1norm')
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
    model.MaxPool('conv1norm', 'pool1', kernel=3, stride=2)

    ################################ conv2 #####################################
    model.Conv(
        'pool1', 'conv2', 96, 256, 5, stride=1, pad=2, group=2,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}),
    )
    bn2 = model.SpatialBN(
        'conv2', 'conv2norm', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=True,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn2 + '_s'], bn2 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn2 + '_b'], bn2 + '_b', value=0.0)
    relu2 = model.Relu('conv2norm', 'conv2norm')
    model.StopGradient('conv2norm', 'conv2norm')
    resize_c2 = model.AveragePool(
        relu2, relu2 + '_s3k12_resize', kernel=12, stride=3, pad=0
    )
    bn_c2 = model.SpatialBN(
        resize_c2, resize_c2 + '_bn', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn_c2 + '_s'], bn_c2 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn_c2 + '_b'], bn_c2 + '_b', value=0.0)
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
    model.MaxPool('conv2norm', 'pool2', kernel=3, stride=2)

    ################################ conv3 #####################################
    model.Conv(
        'pool2', 'conv3', 256, 384, 3, stride=1, pad=1, dilation=1,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}),
    )
    bn3 = model.SpatialBN(
        'conv3', 'conv3norm', 384, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=True,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn3 + '_s'], bn3 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn3 + '_b'], bn3 + '_b', value=0.0)
    relu3 = model.Relu('conv3norm', 'conv3norm')
    model.StopGradient('conv3norm', 'conv3norm')
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
    model.Conv(
        'conv3norm', 'conv4', 384, 384, 3, stride=1, pad=1, dilation=1, group=2,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}),
    )
    bn4 = model.SpatialBN(
        'conv4', 'conv4norm', 384, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=True,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn4 + '_s'], bn4 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn4 + '_b'], bn4 + '_b', value=0.0)
    relu4 = model.Relu('conv4norm', 'conv4norm')
    model.StopGradient('conv4norm', 'conv4norm')
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
    model.Conv(
        'conv4norm', 'conv5', 384, 256, 3, stride=1, pad=1, dilation=1, group=2,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}),
    )
    bn5 = model.SpatialBN(
        'conv5', 'conv5norm', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=True,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn5 + '_s'], bn5 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn5 + '_b'], bn5 + '_b', value=0.0)
    relu5 = model.Relu('conv5norm', 'conv5norm')
    model.StopGradient('conv5norm', 'conv5norm')
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
    pool5 = model.MaxPool('conv5norm', 'pool5', kernel=3, stride=2)
    bn_pool5 = model.SpatialBN(
        pool5, pool5 + '_bn', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [bn_pool5 + '_s'], bn_pool5 + '_s', value=1.0)
        model.param_init_net.ConstantFill(
            [bn_pool5 + '_b'], bn_pool5 + '_b', value=0.0
        )
    fc_pool5 = model.FC(
        bn_pool5, 'fc_pool5', 256 * 6 * 6, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_pool5, 'pred_pool5')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_pool5, labels], 'accuracy_pool5')
        if split == 'train':
            softmax, loss_pool5 = model.SoftmaxWithLoss(
                ['pred_pool5', labels], ['softmax_pool5', 'loss_pool5'],
                scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax(
                'pred_pool5', 'softmax_pool5', engine='CUDNN'
            )
            loss_pool5 = None
        losses.append(loss_pool5)

    return model, softmax, losses
