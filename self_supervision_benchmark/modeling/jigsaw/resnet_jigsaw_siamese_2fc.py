'''
This script builds a ResNet model on ImageNet data - concatenate at FC and have
1 FC layer only in end
'''

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import logging

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.modeling.jigsaw.siamese_model_helper import \
    SiameseModelHelper

logger = logging.getLogger(__name__)

# For more depths, add the block config here
BLOCK_CONFIG = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    200: (3, 32, 36, 3),
}


def create_model(model, data, labels, split):
    siamese_model_helper = SiameseModelHelper(model, split)
    dataset = cfg.DATASET
    logger.info(' | ResNet-{} {}'.format(cfg.MODEL.DEPTH, cfg.DATASET))
    assert cfg.MODEL.DEPTH in BLOCK_CONFIG.keys(), \
        'Block config is not defined for specified model depth. Please check.'
    (n1, n2, n3, n4) = BLOCK_CONFIG[cfg.MODEL.DEPTH]

    num_features = 2048
    fc_channels = 1024 #cfg.JIGSAW.FC6_CHANNELS
    Npatches = 1 #cfg.JIGSAW.NUM_PATCHES
    residual_block = siamese_model_helper.bottleneck_block
    num_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.DEPTH in [18, 34]:
        num_features = 512
        residual_block = siamese_model_helper.basic_block

    # fc_weight_init = ('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD})

    test_mode = False
    if split in ['test', 'val']:
        test_mode = True

    split_data_blobs = []
    for i in range(Npatches):
        split_data_blobs.append('data_{}'.format(i))
    # input data is of shape 32 x (9 * 3) x 64 x 64 - split this into 9 parts
    # of shape 32 x 3 x 64 x 64
    model.net.Split(data, split_data_blobs, axis=1)

    output_blobs = []
    # k_min = str(0)
    for i in range(Npatches):
        if i == 0:
            conv_blob = model.Conv(
                'data_{}'.format(i), 'conv1_s{}'.format(i),
                3, 64, 7, stride=2, pad=3,
                weight_init=('MSRAFill', {}),
                bias_init=('ConstantFill', {'value': 0.0}), no_bias=1
            )
        else:
            conv_blob = model.ConvShared(
                'data_{}'.format(i), 'conv1_s{}'.format(i),
                3, 64, 7, stride=2, pad=3,
                weight='conv1_s0_w',
                no_bias=1
            )

        bn_blob = model.SpatialBN(
            conv_blob, 'res_conv1_bn_s{}'.format(i),
            64, epsilon=cfg.MODEL.BN_EPSILON,
            momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
        )
        if cfg.MODEL.BN_NO_SCALE_SHIFT:
            model.param_init_net.ConstantFill([bn_blob + '_s'], bn_blob + '_s', value=1.0)
            model.param_init_net.ConstantFill([bn_blob + '_b'], bn_blob + '_b', value=0.0)
        relu_blob = model.Relu(bn_blob, bn_blob)
        max_pool = model.MaxPool(
            relu_blob, 'pool1_s{}'.format(i), kernel=3, stride=2, pad=1
        )

        if cfg.MODEL.DEPTH in [50, 101, 152, 200]:
            blob_in, dim_in = siamese_model_helper.residual_layer(
                residual_block, max_pool, 64, 256, stride=1, num_blocks=n1,
                prefix='res2', index=i, dim_inner=64,
                # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
            )
            blob_in, dim_in = siamese_model_helper.residual_layer(
                residual_block, blob_in, dim_in, 512, stride=2, num_blocks=n2,
                prefix='res3', index=i, dim_inner=128,
                # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
            )
            blob_in, dim_in = siamese_model_helper.residual_layer(
                residual_block, blob_in, dim_in, 1024, stride=2, num_blocks=n3,
                prefix='res4', index=i, dim_inner=256,
                # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
            )
            blob_in, dim_in = siamese_model_helper.residual_layer(
                residual_block, blob_in, dim_in, num_features, stride=2, num_blocks=n4,
                prefix='res5', index=i, dim_inner=512,
                # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
            )
        elif cfg.MODEL.DEPTH in [18, 34]:
            blob_in, dim_in = siamese_model_helper.residual_layer(
                residual_block, max_pool, 64, 64, stride=1, num_blocks=n1,
                prefix='res2', index=i,
                # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
            )
            blob_in, dim_in = siamese_model_helper.residual_layer(
                residual_block, blob_in, dim_in, 128, stride=2, num_blocks=n2,
                prefix='res3', index=i,
                # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
            )
            blob_in, dim_in = siamese_model_helper.residual_layer(
                residual_block, blob_in, dim_in, 256, stride=2, num_blocks=n3,
                prefix='res4', index=i,
                # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
            )
            blob_in, dim_in = siamese_model_helper.residual_layer(
                residual_block, blob_in, dim_in, 512, stride=2, num_blocks=n4,
                prefix='res5', index=i,
                # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
            )

        pool_blob = model.AveragePool(
            blob_in, 'pool5_s{}'.format(i), kernel=2, stride=1
        )
        if i == 0:
            blob_out = model.FC(
                pool_blob, 'fc1_s{}'.format(i), num_features, fc_channels,
                weight_init=('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD}),
                bias_init=('ConstantFill', {'value': 0.})
            )
        else:
            blob_out = model.FCShared(
                pool_blob, 'fc1_s{}'.format(i), num_features, fc_channels,
                weight='fc1_s0_w', bias='fc1_s0_b'
            )
        fc1_reshape, _ = model.net.Reshape(
            blob_out, [blob_out + '_reshape', blob_out + '_old_shape'],
            shape=(-1, fc_channels, 1, 1)
        )
        fc1_bn = model.SpatialBN(
            fc1_reshape, 'fc1_s{}_bn'.format(i), fc_channels,
            epsilon=cfg.MODEL.BN_EPSILON,
            momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
        )
        if cfg.MODEL.BN_NO_SCALE_SHIFT:
            model.param_init_net.ConstantFill([fc1_bn + '_s'], fc1_bn + '_s', value=1.0)
            model.param_init_net.ConstantFill([fc1_bn + '_b'], fc1_bn + '_b', value=0.0)
        blob_out = model.Relu(fc1_bn, fc1_bn + '_relu')
        output_blobs.append(blob_out)

    if cfg.EXTRACT.ENABLED:
        model.ReduceMean('res4_5_branch2c_bn_s0', 'res4_5_branch2c_bn_s0_avg', axes=[2,3])
        return model, None, None
    concat, _ = model.net.Concat(
        output_blobs, ['fc1_concat', '_concat_fc1_concat'], axis=1
    )
    # Here we can experiment with fc_channels=1024, 512 ablation
    blob_out = model.FC(
        'fc1_concat', 'fc2', fc_channels * Npatches, num_classes,
        weight_init=('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD}),
        bias_init=('ConstantFill', {'value': 0.})
    )

    model.net.Alias('fc2', 'pred')
    if split == 'train':
        if cfg.DISTRIBUTED.DISTR_ON:
            scale = (
                1. / cfg.NUM_DEVICES / cfg.DISTRIBUTED.NUM_NODES /
                cfg.MODEL.GRAD_ACCUM_FREQUENCY
            )
        else:
            scale = 1. / cfg.NUM_DEVICES / cfg.MODEL.GRAD_ACCUM_FREQUENCY
        if cfg.DATASET in ['instagram', 'imagenet22k', 'yfcc100m', 'yfcc100m_handles', 'places205'] or cfg.DROP_BATCHES.DROP_ON:
            softmax, loss, _ = model.SoftmaxWithLoss(
                [blob_out, labels, 'img_weight'],
                ['softmax', 'loss', 'elt_loss'],
                scale=scale
            )
        else:
            softmax, loss = model.SoftmaxWithLoss(
                [blob_out, labels],
                ['softmax', 'loss'],
                scale=scale
            )
    elif split in ['test', 'val']:
        softmax = model.Softmax(blob_out, 'softmax', engine='CUDNN')
        loss = None
    return model, softmax, loss
