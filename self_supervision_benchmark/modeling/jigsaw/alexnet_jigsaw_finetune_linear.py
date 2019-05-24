# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
AlexNet model for testing feature quality of Jigsaw approach pre-training.
Transfer Task - Linear classifiers on ImageNet-1K/Places205 datasets. Closely
follow: https://github.com/MehdiNoroozi/JigsawPuzzleSolver/blob/master/train_val_cfn_jps.prototxt
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

from self_supervision_benchmark.core.config import config as cfg


def create_model(model, data, labels, split):
    # data and labels are the blobs
    num_classes = cfg.MODEL.NUM_CLASSES
    scale = 1. / cfg.NUM_DEVICES
    losses, softmax = [], None
    test_mode = False
    if split in ['test', 'val']:
        test_mode = True

    ################################ conv1 ####################################
    # suffixes s0, s1, s2,... are due to the siamese CNN network used in
    # pre-training. Hence, "s0" -> branch 1, "s1" -> branch 2 etc. of the
    # siamese network.
    conv1 = model.Conv(
        'data', 'conv1_s0', 3, 96, 11, stride=2,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    relu1 = model.Relu(conv1, conv1)
    model.StopGradient(relu1, relu1)
    resize_c1 = model.AveragePool(
        relu1, relu1 + '_s9k28_resize', kernel=28, stride=9, pad=0
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

    ################################ pool1 ####################################
    pool1 = model.MaxPool(relu1, 'pool1_s0', kernel=3, stride=2)
    resize_pool1 = model.AveragePool(
        pool1, pool1 + '_resize', kernel=6, stride=6, pad=3
    )
    bn_pool1 = model.SpatialBN(
        resize_pool1, resize_pool1 + '_bn', 96, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [bn_pool1 + '_s'], bn_pool1 + '_s', value=1.0
        )
        model.param_init_net.ConstantFill(
            [bn_pool1 + '_b'], bn_pool1 + '_b', value=0.0
        )
    fc_pool1 = model.FC(
        bn_pool1, 'fc_pool1', 96 * 10 * 10, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_pool1, 'pred_pool1')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_pool1, labels], 'accuracy_pool1')
        if split == 'train':
            softmax, loss_pool1 = model.SoftmaxWithLoss(
                ['pred_pool1', labels], ['softmax_pool1', 'loss_pool1'],
                scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax(
                'pred_pool1', 'softmax_pool1', engine='CUDNN'
            )
            loss_pool1 = None
        losses.append(loss_pool1)

    ################################ conv2 ####################################
    conv2 = model.Conv(
        pool1, 'conv2_s0', 96, 256, 5,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}), pad=2, group=2,
    )
    relu2 = model.Relu(conv2, conv2)
    model.StopGradient(relu2, relu2)
    resize_c2 = model.AveragePool(
        relu2, relu2 + '_s6k24_resize', kernel=24, stride=6, pad=0
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

    ################################ pool2 ####################################
    pool2 = model.MaxPool(relu2, 'pool2_s0', kernel=3, stride=2)
    resize_pool2 = model.AveragePool(
        pool2, pool2 + '_resize', kernel=4, stride=4, pad=0
    )
    bn_pool2 = model.SpatialBN(
        resize_pool2, resize_pool2 + '_bn', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [bn_pool2 + '_s'], bn_pool2 + '_s', value=1.0
        )
        model.param_init_net.ConstantFill(
            [bn_pool2 + '_b'], bn_pool2 + '_b', value=0.0
        )
    fc_pool2 = model.FC(
        bn_pool2, 'fc_pool2', 256 * 6 * 6, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_pool2, 'pred_pool2')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_pool2, labels], 'accuracy_pool2')
        if split == 'train':
            softmax, loss_pool2 = model.SoftmaxWithLoss(
                ['pred_pool2', labels], ['softmax_pool2', 'loss_pool2'],
                scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax(
                'pred_pool2', 'softmax_pool2', engine='CUDNN'
            )
            loss_pool2 = None
        losses.append(loss_pool2)

    ################################ conv3 ####################################
    conv3 = model.Conv(
        pool2, 'conv3_s0', 256, 384, 3,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}), pad=1,
    )
    bn3 = model.SpatialBN(
        conv3, 'conv3_s0_bn', 384, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=True,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn3 + '_s'], bn3 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn3 + '_b'], bn3 + '_b', value=0.0)
    relu3 = model.Relu(bn3, bn3 + '_relu')
    model.StopGradient(relu3, relu3)
    resize_c3 = model.AveragePool(
        relu3, relu3 + '_s3k14_resize', kernel=14, stride=3, pad=0
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

    ################################ conv4 ####################################
    conv4 = model.Conv(
        relu3, 'conv4_s0', 384, 384, 3,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
        pad=1, group=2,
    )
    bn4 = model.SpatialBN(
        conv4, 'conv4_s0_bn', 384, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=True,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn4 + '_s'], bn4 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn4 + '_b'], bn4 + '_b', value=0.0)
    relu4 = model.Relu(bn4, bn4 + '_relu')
    model.StopGradient(relu4, relu4)
    resize_c4 = model.AveragePool(
        relu4, relu4 + '_s3k14_resize', kernel=14, stride=3, pad=0
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

    ################################ conv5 ####################################
    conv5 = model.Conv(
        relu4, 'conv5_s0', 384, 256, 3,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {}),
        pad=1, group=2,
    )
    bn5 = model.SpatialBN(
        conv5, 'conv5_s0_bn', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=True,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn5 + '_s'], bn5 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn5 + '_b'], bn5 + '_b', value=0.0)
    relu5 = model.Relu(bn5, bn5 + '_relu')
    model.StopGradient(relu5, relu5)
    resize_c5 = model.AveragePool(
        relu5, relu5 + '_s2k16_resize', kernel=16, stride=2, pad=0
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

    ################################ pool5 ####################################
    pool5 = model.MaxPool(relu5, 'pool5_s0', kernel=3, stride=2)
    resize_pool5 = model.AveragePool(
        pool5, pool5 + '_resize', kernel=2, stride=2, pad=0
    )
    bn_pool5 = model.SpatialBN(
        resize_pool5, resize_pool5 + '_bn', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [bn_pool5 + '_s'], bn_pool5 + '_s', value=1.0
        )
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
            softmax = model.Softmax('pred_pool5', 'softmax_pool5', engine='CUDNN')
            loss_pool5 = None
        losses.append(loss_pool5)

    return model, softmax, losses
