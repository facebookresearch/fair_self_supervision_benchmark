'''
This script builds a ResNet model on ImageNet data - concatenate at FC and have
1 FC layer only in end
'''

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import logging

from caffe2.python import core

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
    center_patch = 4
    dataset = cfg.DATASET
    if cfg.SEPARATE_TEST_DATASET and split in ['test', 'val']:
        dataset = cfg.TEST_DATASET
    logger.info(' | ResNet-{} {}'.format(cfg.MODEL.DEPTH, dataset))
    assert cfg.MODEL.DEPTH in BLOCK_CONFIG.keys(), \
        'Block config is not defined for specified model depth. Please check.'
    (n1, n2, n3, n4) = BLOCK_CONFIG[cfg.MODEL.DEPTH]

    num_features = 2048
    fc_channels = cfg.JIGSAW.FC6_CHANNELS
    num_classes = cfg.MODEL.NUM_CLASSES

    siamese_model_helper = SiameseModelHelper(model, split,
        nonshared_index=[center_patch])
    residual_block = siamese_model_helper.bottleneck_block
    if cfg.MODEL.DEPTH in [18, 34]:
        num_features = 512
        residual_block = siamese_model_helper.basic_block

    frozen_siamese_model_helper = SiameseModelHelper(model, 'test',
        nonshared_index=[center_patch])
    frozen_residual_block = frozen_siamese_model_helper.bottleneck_block
    if cfg.MODEL.DEPTH in [18, 34]:
        num_features = 512
        frozen_residual_block = frozen_siamese_model_helper.basic_block

    # fc_weight_init = ('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD})

    test_mode = False
    if split in ['test', 'val']:
        test_mode = True

    split_data_blobs = []
    for i in range(cfg.JIGSAW.NUM_PATCHES):
        split_data_blobs.append('data_{}'.format(i))
    # input data is of shape 32 x (9 * 3) x 64 x 64 - split this into 9 parts
    # of shape 32 x 3 x 64 x 64
    model.net.Split(data, split_data_blobs, axis=1)

    for i in range(cfg.JIGSAW.NUM_PATCHES):
        model.net.Squeeze('data_{}'.format(i), 'data_{}'.format(i), dims=[1])
    #TODO for testing *******************************
    # h = model.net.ReduceSum(data, 'pred', axes=[2,3,4], keepdims=False)
    # model.FC(h, 'finalfc', 9, num_classes)
    # loss = model.net.ReduceSum('finalfc', 'loss', axes=[0,1])
    # return model, None, loss
    #TODO for testing *******************************

    output_blobs = []
    # k_min = str(0)
    for i in range(cfg.JIGSAW.NUM_PATCHES):
        if i == 0 or i == center_patch:
            conv_blob = model.Conv(
                'data_{}'.format(i), 'conv1_s{}'.format(i),
                3, 64, 7, stride=2, pad=3,
                weight_init=('MSRAFill', {}),
                bias_init=('ConstantFill', {'value': 0.0}), no_bias=1
            )
        else:
            # conv_blob = model.Conv(
            #     'data_{}'.format(i), 'conv1_s{}'.format(i),
            #     3, 64, 7, stride=2, pad=3,
            #     weight_init=('MSRAFill', {}),
            #     bias_init=('ConstantFill', {'value': 0.0}), no_bias=1
            # )
            conv_blob = model.ConvShared(
                'data_{}'.format(i), 'conv1_s{}'.format(i),
                3, 64, 7, stride=2, pad=3,
                weight='conv1_s0_w',
                no_bias=1
            )

        if i != center_patch:
            bn_blob = model.SpatialBN(
                conv_blob, 'res_conv1_bn_s{}'.format(i),
                64, epsilon=cfg.MODEL.BN_EPSILON,
                momentum=cfg.MODEL.BN_MOMENTUM, is_test=True,
            )
            if cfg.JIGSAW.BN_NO_SCALE_SHIFT:
                model.param_init_net.ConstantFill([bn_blob + '_s'], bn_blob + '_s', value=1.0)
                model.param_init_net.ConstantFill([bn_blob + '_b'], bn_blob + '_b', value=0.0)
            relu_blob = model.Relu(bn_blob, bn_blob)
            max_pool = model.MaxPool(
                relu_blob, 'pool1_s{}'.format(i), kernel=3, stride=2, pad=1
            )

            if cfg.MODEL.DEPTH in [50, 101, 152, 200]:
                blob_in, dim_in = frozen_siamese_model_helper.residual_layer(
                    frozen_residual_block, max_pool, 64, 256, stride=1, num_blocks=n1,
                    prefix='res2', index=i, dim_inner=64,
                    # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
                )
                blob_in, dim_in = frozen_siamese_model_helper.residual_layer(
                    frozen_residual_block, blob_in, dim_in, 512, stride=2, num_blocks=n2,
                    prefix='res3', index=i, dim_inner=128,
                    # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
                )
                blob_in, dim_in = frozen_siamese_model_helper.residual_layer(
                    frozen_residual_block, blob_in, dim_in, 1024, stride=2, num_blocks=n3,
                    prefix='res4', index=i, dim_inner=256,
                    # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
                )
                # blob_in, dim_in = frozen_siamese_model_helper.residual_layer(
                #     frozen_residual_block, blob_in, dim_in, num_features, stride=2, num_blocks=n4,
                #     prefix='res5', index=i, dim_inner=512,
                #     residual_scale=cfg.MODEL.RESIDUAL_SCALE,
                # )
            elif cfg.MODEL.DEPTH in [18, 34]:
                blob_in, dim_in = frozen_siamese_model_helper.residual_layer(
                    frozen_residual_block, max_pool, 64, 64, stride=1, num_blocks=n1,
                    prefix='res2', index=i,
                    # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
                )
                blob_in, dim_in = frozen_siamese_model_helper.residual_layer(
                    frozen_residual_block, blob_in, dim_in, 128, stride=2, num_blocks=n2,
                    prefix='res3', index=i,
                    # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
                )
                blob_in, dim_in = frozen_siamese_model_helper.residual_layer(
                    frozen_residual_block, blob_in, dim_in, 256, stride=2, num_blocks=n3,
                    prefix='res4', index=i,
                    # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
                )
                blob_in, dim_in = frozen_siamese_model_helper.residual_layer(
                    frozen_residual_block, blob_in, dim_in, 512, stride=2, num_blocks=n4,
                    prefix='res5', index=i,
                    # residual_scale=cfg.MODEL.RESIDUAL_SCALE,
                )
        else:
            bn_blob = model.SpatialBN(
                conv_blob, 'res_conv1_bn_s{}'.format(i),
                64, epsilon=cfg.MODEL.BN_EPSILON,
                momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
            )
            if cfg.JIGSAW.BN_NO_SCALE_SHIFT:
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
                if cfg.MIDLEVEL.SUPERVISION == 5:
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

        # pool_blob = model.AveragePool(
        #     blob_in, 'pool5_s{}'.format(i), kernel=2, stride=1
        # )

        # NxC
        res45 = 'res4_5_branch2c_bn_s{}'.format(i)
        if i != center_patch:
            pool_blob = model.net.ReduceMean(res45,
                res45 + '_avg'.format(i),
                axes=[2, 3], keepdims=False)
            model.StopGradient(pool_blob, pool_blob)
            output_blobs.append(pool_blob)
            continue
        else:
            if cfg.MIDLEVEL.SUPERVISION == 5:
                res45 = 'res5_2_branch2c_bn_s{}'.format(i)
            if cfg.MIDLEVEL.FC1024:
                pool_blob = model.net.ReduceMean(res45,
                    res45 + '_avg'.format(i),
                    axes=[2, 3], keepdims=False)
                blob_out = pool_blob
            else:
                blob_out = res45

    # 8N x C
    model.net.Concat(output_blobs, ['feats_concat', 'split_info'], axis=0)
    # N x C
    model.AssignCluster('feats_concat', [
        '_labels',
        # 'dists',
        labels,
        'individual_labels'])

    # model.net.Cast('labels_', labels, to=core.DataType.INT32)

    if cfg.MIDLEVEL.GAUSSIAN:
        model.FC(
            blob_out, 'pred',
            fc_channels if cfg.MIDLEVEL.FC1024 else fc_channels * 4 * 4,
            num_classes,
            weight_init=('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        blob_out, _ = model.net.Reshape(
            ['pred'], ['fc_reshaped', 'fc_old_shape'],
            shape=(-1, num_classes, 1, 1)
        )
        priors_shape = (cfg.TRAIN.BATCH_SIZE if model.train else cfg.TEST.BATCH_SIZE) // cfg.NUM_DEVICES
        model.net.ConstantFill([], 'priors', value=1., shape=(priors_shape,))

    else:
        raise
    # model.net.Alias('finalfc', 'pred')
    if split == 'train':
        if cfg.DISTRIBUTED.DISTR_ON:
            scale = (
                1. / cfg.NUM_DEVICES / cfg.DISTRIBUTED.NUM_NODES /
                cfg.MODEL.GRAD_ACCUM_FREQUENCY
            )
        else:
            scale = 1. / cfg.NUM_DEVICES / cfg.MODEL.GRAD_ACCUM_FREQUENCY
        if cfg.DATASET in ['instagram', 'imagenet22k', 'yfcc100m', 'yfcc100m_handles', 'places205'] or cfg.DROP_BATCHES.DROP_ON:
            raise
            # softmax, loss, _ = model.SoftmaxWithLoss(
            #     [blob_out, labels, 'img_weight'],
            #     ['softmax', 'loss', 'elt_loss'],
            #     scale=scale
            # )
        else:
            loss, softmax = model.net.SpatialSoftmaxCrossEntropyLoss(
                [blob_out, '_labels', 'priors'],
                ['loss', '_softmax'],
                scale=scale, num_classes=num_classes
            )
            model.net.Squeeze('_softmax', 'softmax', dims=[2, 3])
            # softmax, loss = model.SoftmaxWithLoss(
            #     [blob_out, labels],
            #     ['softmax', 'loss'],
            #     scale=scale
            # )
    elif split in ['test', 'val']:
        softmax = model.Softmax(blob_out, 'softmax', engine='CUDNN')
        loss = None
    return model, softmax, loss
