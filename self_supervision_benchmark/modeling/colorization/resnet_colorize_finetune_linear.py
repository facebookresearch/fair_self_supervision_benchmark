# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
ResNet50 model for testing feature quality of Colorization approach pre-training.
Transfer Task - Linear classifiers on ImageNet-1K/Places205 datasets.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import logging

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.modeling.resnet_model_helper import \
    ModelHelper

# create the logger
logger = logging.getLogger(__name__)

# For more depths, add the depth config here: <depth>: (n1, n2, n3, n4) where
# n{} is the number of resnet blocks per stage. For the purpose of this
# benchmark, we experiment with ResNet-50 only.
BLOCK_CONFIG = {
    50: (3, 4, 6, 3),
}


def create_model(model, data, labels, split):
    model_helper = ModelHelper(model, split)
    logger.info(' | ResNet-{} {}'.format(cfg.MODEL.DEPTH, cfg.DATASET))
    assert cfg.MODEL.DEPTH in BLOCK_CONFIG.keys(), \
        'Block config is not defined for specified model depth. Please check.'
    (n1, n2, n3, n4) = BLOCK_CONFIG[cfg.MODEL.DEPTH]

    num_features = 2048
    residual_block = model_helper.bottleneck_block
    num_classes = cfg.MODEL.NUM_CLASSES
    test_mode = False
    if split in ['test', 'val']:
        test_mode = True

    scale = 1. / cfg.NUM_DEVICES
    losses, softmax = [], None

    ################################ Split #####################################
    # split the input LAB channels into L and AB. The input to the colorization
    # models is always L channel (even during transfer tasks).
    model.net.Split(data, ['data_l', 'data_ab'], axis=1, split=[1, 2])

    ################################## conv1 ###################################
    conv_blob = model.Conv(
        'data_l', 'conv1', 1, 64, 7, stride=2, pad=3,
        weight_init=('MSRAFill', {}), no_bias=1
    )
    bn_blob = model.SpatialBN(
        conv_blob, 'res_conv1_bn', 64, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=True,
    )
    relu_blob = model.Relu(bn_blob, bn_blob)
    model.StopGradient(relu_blob, relu_blob)
    resize_c1 = model.AveragePool(
        relu_blob, relu_blob + '_resize', kernel=10, stride=10, pad=4
    )
    bn_c1 = model.SpatialBN(
        resize_c1, resize_c1 + '_bn', 64, epsilon=cfg.MODEL.BN_EPSILON,
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
        bn_c1, 'conv1_cls', 64 * 12 * 12, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_conv1, 'pred_conv1')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_conv1, labels], 'accuracy_conv1')
        if split == 'train':
            softmax, loss_c1 = model.SoftmaxWithLoss(
                ['pred_conv1', labels],
                ['softmax_conv1', 'loss_conv1'], scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax(
                'pred_conv1', 'softmax_conv1', engine='CUDNN'
            )
            loss_c1 = None
        losses.append(loss_c1)

    ################################## pool1 ###################################
    max_pool = model.MaxPool(relu_blob, 'pool1', kernel=3, stride=2, pad=1)
    model.StopGradient(max_pool, max_pool)
    resize_pool1 = model.AveragePool(
        max_pool, max_pool + '_resize', kernel=5, stride=5, pad=2
    )
    bn_pool1 = model.SpatialBN(
        resize_pool1, resize_pool1 + '_bn', 64, epsilon=cfg.MODEL.BN_EPSILON,
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
        bn_pool1, 'pool1_cls', 64 * 12 * 12, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_pool1, 'pred_pool1')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_pool1, labels], 'accuracy_poo1')
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

    ################################## stage2 ##################################
    blob_in, dim_in = model_helper.residual_layer(
        residual_block, max_pool, 64, 256, stride=1, num_blocks=n1,
        prefix='res2', dim_inner=64,
    )
    model.StopGradient(blob_in, blob_in)
    resize_c2 = model.AveragePool(
        blob_in, blob_in + '_s8k16_resize', stride=8, kernel=16, pad=0
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
        bn_c2, 'conv2_cls', 256 * 6 * 6, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_conv2, 'pred_conv2')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_conv2, labels], 'accuracy_conv2')
        if split == 'train':
            softmax, loss_c2 = model.SoftmaxWithLoss(
                ['pred_conv2', labels], ['softmax_conv2', 'loss_conv2'],
                scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax(
                'pred_conv2', 'softmax_conv2', engine='CUDNN'
            )
            loss_c2 = None
        losses.append(loss_c2)

    ################################## stage3 ##################################
    blob_in, dim_in = model_helper.residual_layer(
        residual_block, blob_in, dim_in, 512, stride=2, num_blocks=n2,
        prefix='res3', dim_inner=128,
    )
    model.StopGradient(blob_in, blob_in)
    resize_c3 = model.AveragePool(
        blob_in, blob_in + '_s5k13_resize', stride=5, kernel=13, pad=0
    )
    bn_c3 = model.SpatialBN(
        resize_c3, resize_c3 + '_bn', 512, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn_c3 + '_s'], bn_c3 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn_c3 + '_b'], bn_c3 + '_b', value=0.0)
    fc_conv3 = model.FC(
        bn_c3, 'conv3_cls', 512 * 4 * 4, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_conv3, 'pred_conv3')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_conv3, labels], 'accuracy_conv3')
        if split == 'train':
            softmax, loss_c3 = model.SoftmaxWithLoss(
                ['pred_conv3', labels], ['softmax_conv3', 'loss_conv3'],
                scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax(
                'pred_conv3', 'softmax_conv3', engine='CUDNN'
            )
            loss_c3 = None
        losses.append(loss_c3)

    ################################## stage4 ##################################
    blob_in, dim_in = model_helper.residual_layer(
        residual_block, blob_in, dim_in, 1024, stride=2, num_blocks=n3,
        prefix='res4', dim_inner=256,
    )
    model.StopGradient(blob_in, blob_in)
    resize_c4 = model.AveragePool(
        blob_in, blob_in + '_s3k8_resize', stride=3, kernel=8, pad=0
    )
    bn_c4 = model.SpatialBN(
        resize_c4, resize_c4 + '_bn', 1024, epsilon=cfg.MODEL.BN_EPSILON,
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
        bn_c4, 'conv4_cls', 1024 * 3 * 3, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_conv4, 'pred_conv4')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_conv4, labels], 'accuracy_conv4')
        if split == 'train':
            softmax, loss_c4 = model.SoftmaxWithLoss(
                ['pred_conv4', labels], ['softmax_conv4', 'loss_conv4'],
                scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax(
                'pred_conv4', 'softmax_conv4', engine='CUDNN'
            )
            loss_c4 = None
        losses.append(loss_c4)

    ################################## stage5 ##################################
    blob_in, dim_in = model_helper.residual_layer(
        residual_block, blob_in, dim_in, num_features, stride=1,
        num_blocks=n4, prefix='res5', dim_inner=512,
    )
    model.StopGradient(blob_in, blob_in)
    resize_c5 = model.AveragePool(
        blob_in, blob_in + '_s2k12_resize', stride=2, kernel=12, pad=0
    )
    bn_c5 = model.SpatialBN(
        resize_c5, resize_c5 + '_bn', num_features, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn_c5 + '_s'], bn_c5 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn_c5 + '_b'], bn_c5 + '_b', value=0.0)
    fc_conv5 = model.FC(
        bn_c5, 'conv5_cls', 2048 * 2 * 2, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(fc_conv5, 'pred_conv5')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([fc_conv5, labels], 'accuracy_conv5')
        if split == 'train':
            softmax, loss_c5 = model.SoftmaxWithLoss(
                [fc_conv5, labels], ['softmax_conv5', 'loss_conv5'], scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax(
                'pred_conv5', 'softmax_conv5', engine='CUDNN'
            )
            loss_c5 = None
        losses.append(loss_c5)

    ################################## pool5 ###################################
    pool_blob = model.AveragePool(blob_in, 'pool5', kernel=14, stride=1)
    model.StopGradient(pool_blob, pool_blob)
    bn_pool5 = model.SpatialBN(
        pool_blob, pool_blob + '_bn', num_features,
        epsilon=cfg.MODEL.BN_EPSILON, momentum=cfg.MODEL.BN_MOMENTUM,
        is_test=test_mode
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [bn_pool5 + '_s'], bn_pool5 + '_s', value=1.0
        )
        model.param_init_net.ConstantFill(
            [bn_pool5 + '_b'], bn_pool5 + '_b', value=0.0
        )

    ################################## fc ######################################
    blob_out = model.FC(
        bn_pool5, 'pool5_cls', num_features, num_classes,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    model.net.Alias(blob_out, 'pred_pool5')
    if not cfg.MODEL.EXTRACT_FEATURES_ONLY:
        model.Accuracy([blob_out, labels], 'accuracy_pool5')
        if split == 'train':
            softmax, loss_pool5 = model.SoftmaxWithLoss(
                [blob_out, labels], ['softmax_pool5', 'loss_pool5'], scale=scale
            )
        elif split in ['test', 'val']:
            softmax = model.Softmax(
                'pred_pool5', 'softmax_pool5', engine='CUDNN'
            )
            loss_pool5 = None
        losses.append(loss_pool5)

    return model, softmax, losses
