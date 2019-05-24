# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
AlexNet model for testing feature quality of Jigsaw approach pre-training.
Transfer Task - Full Finetuning on VOC07 dataset. Closely
follow: https://github.com/MehdiNoroozi/JigsawPuzzleSolver/blob/master/train_val_cfn_jps.prototxt

SpatialBN layers names {}_s0_{}
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
    elif cfg.MODEL.FC_INIT_TYPE == 'msra':
        fc_weight_init = ('MSRAFill', {})
        fc_bias_init = ('ConstantFill', {'value': 0.})

    ################################ conv1 ####################################
    conv1 = model.Conv(
        'data', 'conv1_s0', 3, 96, 11, stride=4,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}),
    )
    relu1 = model.Relu(conv1, conv1)
    pool1 = model.MaxPool(relu1, 'pool1_s0', kernel=3, stride=2)
    lrn1 = model.LRN(pool1, 'norm1_s0', size=5, alpha=0.0001, beta=0.75)

    ################################ conv2 ####################################
    conv2 = model.Conv(
        lrn1, 'conv2_s0', 96, 256, 5,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 1.0}), pad=2, group=2,
    )
    relu2 = model.Relu(conv2, conv2)
    pool2 = model.MaxPool(relu2, 'pool2_s0', kernel=3, stride=2)
    lrn2 = model.LRN(pool2, 'norm2_s0', size=5, alpha=0.0001, beta=0.75)

    ################################ conv3 ####################################
    conv3 = model.Conv(
        lrn2, 'conv3_s0', 256, 384, 3,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.0}), pad=1,
    )
    bn3 = model.SpatialBN(
        conv3, 'conv3_s0_bn', 384, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn3 + '_s'], bn3 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn3 + '_b'], bn3 + '_b', value=0.0)
    relu3 = model.Relu(bn3, bn3 + '_relu')

    ################################ conv4 ####################################
    conv4 = model.Conv(
        relu3, 'conv4_s0', 384, 384, 3,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 1.0}), pad=1, group=2,
    )
    bn4 = model.SpatialBN(
        conv4, 'conv4_s0_bn', 384, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn4 + '_s'], bn4 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn4 + '_b'], bn4 + '_b', value=0.0)
    relu4 = model.Relu(bn4, bn4 + '_relu')

    ################################ conv5 ####################################
    conv5 = model.Conv(
        relu4, 'conv5_s0', 384, 256, 3,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {}), pad=1, group=2,
    )
    bn5 = model.SpatialBN(
        conv5, 'conv5_s0_bn', 256, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill([bn5 + '_s'], bn5 + '_s', value=1.0)
        model.param_init_net.ConstantFill([bn5 + '_b'], bn5 + '_b', value=0.0)
    relu5 = model.Relu(bn5, bn5 + '_relu')
    pool5 = model.MaxPool(relu5, 'pool5_s0', kernel=3, stride=2)

    ################################## fc6 #####################################
    # in Jigsaw pretext, the fc6-8 are different so can't initialize from those
    fc6 = model.FC(
        pool5, 'fc6', 256 * 6 * 6, 4096,
        weight_init=fc_weight_init, bias_init=fc_bias_init,
    )
    fc6_reshape, _ = model.net.Reshape(
        fc6, [fc6 + '_reshape', fc6 + '_old_shape'], shape=(-1, 4096, 1, 1)
    )
    fc6_bn = model.SpatialBN(
        fc6_reshape, fc6_reshape + '_bn', 4096, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [fc6_bn + '_s'], fc6_bn + '_s', value=1.0
        )
        model.param_init_net.ConstantFill(
            [fc6_bn + '_b'], fc6_bn + '_b', value=0.0
        )
    blob_out = model.Relu(fc6_bn, fc6_bn + '_relu')
    if split not in ['test', 'val']:
        blob_out = model.Dropout(
            blob_out, blob_out + '_dropout', ratio=0.5, is_test=test_mode
        )

    ################################## fc7 #####################################
    # for fc7 and fc8, we still name blobs with suffix 's0' so that we can
    # distinguish it with the jigsaw model otherwise the blob names will clash
    # when using checkpointed model from jigsaw and fc7/8 will have dimension
    # mismatch
    fc7 = model.FC(
        blob_out, 'fc7_s0', 4096, 4096,
        weight_init=fc_weight_init, bias_init=fc_bias_init,
    )

    fc7_reshape, _ = model.net.Reshape(
        fc7, [fc7 + '_reshape', fc7 + '_old_shape'], shape=(-1, 4096, 1, 1)
    )
    fc7_bn = model.SpatialBN(
        fc7_reshape, fc7_reshape + '_bn', 4096, epsilon=cfg.MODEL.BN_EPSILON,
        momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
    )
    if cfg.MODEL.BN_NO_SCALE_SHIFT:
        model.param_init_net.ConstantFill(
            [fc7_bn + '_s'], fc7_bn + '_s', value=1.0
        )
        model.param_init_net.ConstantFill(
            [fc7_bn + '_b'], fc7_bn + '_b', value=0.0
        )
    blob_out = model.Relu(fc7_bn, fc7_bn + '_relu')
    if split not in ['test', 'val']:
        blob_out = model.Dropout(
            blob_out, blob_out + '_dropout', ratio=0.5, is_test=test_mode
        )

    ################################## fc8 #####################################
    fc8 = model.FC(
        blob_out, 'fc8_s0', 4096, num_classes,
        weight_init=fc_weight_init,
        bias_init=fc_bias_init,
    )

    ################################## Sigmoid #################################
    # Sigmoid loss since VOC07 iss multi-label
    model.net.Alias(fc8, 'pred')
    sigmoid = model.net.Sigmoid('pred', 'sigmoid')
    scale = 1. / cfg.NUM_DEVICES
    if split == 'train':
        loss = model.net.SigmoidCrossEntropyLoss(
            ['pred', labels], 'loss', scale=scale
        )
    elif split in ['test', 'val']:
        loss = None
    return model, sigmoid, loss
