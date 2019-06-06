# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This script can be used to extract features from different layers of the model.
These extracted features can further be used to train SVMs on.

Relevant transfer task:
1. Image Classification (VOC07, COCO2014)
2. Low shot image classification (Places205, VOC07)
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import argparse
import logging
import numpy as np
import os
import sys

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.core.config import (
    cfg_from_file, cfg_from_list, assert_cfg, print_cfg
)
from self_supervision_benchmark.modeling import model_builder
from self_supervision_benchmark.utils import helpers, checkpoints

from caffe2.python import workspace

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def extract_features(opts):
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)

    # create the model in test mode only
    assert opts.data_type in ['train', 'val', 'test'], "Please specify valid type."
    model = model_builder.ModelBuilder(
        name='{}_test'.format(cfg.MODEL.MODEL_NAME), train=False,
        use_cudnn=True, cudnn_exhaustive_search=True, split=opts.data_type
    )
    model.build_model()
    model.create_net()
    model.start_data_loader()

    # initialize the model
    if cfg.TEST.PARAMS_FILE:
        checkpoints.load_model_from_params_file(
            model, params_file=cfg.TEST.PARAMS_FILE, checkpoint_dir=None)
    else:
        logger.info('=====WARN: No params files specified for testing model!')

    # initialize the dictionary to store features and targets
    img_features, img_targets = {}, {}
    for bl in cfg.MODEL.EXTRACT_BLOBS:
        img_features[bl], img_targets[bl] = {}, {}

    # we keep track of data indices seen so far. This ensures we extract feature
    # for each image only once.
    indices_list = []
    total_test_iters = helpers.get_num_test_iter(model.input_db)
    logger.info('Test epoch iters: {}'.format(total_test_iters))
    # when we extract features, we run 4 epochs to make sure we capture all the
    # data points. This is needed because we use the multi-processing dataloader
    # which shuffles the data. In very low-shot setting, making multiple passes
    # over the entire data becomes crucial.
    extraction_iters = int(total_test_iters * 4)
    for test_iter in range(0, extraction_iters):
        workspace.RunNet(model.net.Proto().name)
        if test_iter == 0:
            helpers.print_net(model)
        if test_iter % 100 == 0:
            logger.info('at: [{}/{}]'.format(test_iter, extraction_iters))
        for device in range(cfg.NUM_DEVICES):
            indices = workspace.FetchBlob('gpu_{}/db_indices'.format(device))
            labels = workspace.FetchBlob('gpu_{}/labels'.format(device))
            num_images = indices.shape[0]
            indices_list.extend(list(indices))
            for bl in cfg.MODEL.EXTRACT_BLOBS:
                features = workspace.FetchBlob('gpu_{}/{}'.format(device, bl))
                for idx in range(num_images):
                    index = indices[idx]
                    if not (index in img_features[bl]):
                        img_targets[bl][index] = labels[idx].reshape(-1)
                        img_features[bl][index] = features[idx]

    for bl in cfg.MODEL.EXTRACT_BLOBS:
        img_features[bl] = dict(sorted(img_features[bl].items()))
        img_targets[bl] = dict(sorted(img_targets[bl].items()))
        feats = np.array(list(img_features[bl].values()))
        N = feats.shape[0]
        logger.info('got image features: {} {}'.format(bl, feats.shape))
        output = {
            'img_features': feats.reshape(N, -1),
            'img_inds': np.array(list(img_features[bl].keys())),
            'img_targets': np.array(list(img_targets[bl].values())),
        }
        prefix = '{}_{}_'.format(opts.output_file_prefix, bl)
        out_feat_file = os.path.join(opts.output_dir, prefix + 'features.npy')
        out_target_file = os.path.join(opts.output_dir, prefix + 'targets.npy')
        out_inds_file = os.path.join(opts.output_dir, prefix + 'inds.npy')
        logger.info('Saving extracted features: {} {} to: {}'.format(
            bl, output['img_features'].shape, out_feat_file))
        np.save(out_feat_file, output['img_features'])
        logger.info('Saving extracted targets: {} to: {}'.format(
            output['img_targets'].shape, out_target_file))
        np.save(out_target_file, output['img_targets'])
        logger.info('Saving extracted indices: {} to: {}'.format(
            output['img_inds'].shape, out_inds_file))
        np.save(out_inds_file, output['img_inds'])

    logger.info('All Done!')
    # shut down the data loader
    model.data_loader.shutdown_dataloader()


def main():
    parser = argparse.ArgumentParser(description='Self-Supervision Benchmark')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output path where to extract the features/labels')
    parser.add_argument('--output_file_prefix', type=str, default='trainval',
                        help='trainval | test | val prefix for the file name')
    parser.add_argument('--data_type', type=str, default='train',
                        help='train | val | test')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional config file for params')
    parser.add_argument('--args_dict', type=str, default=None,
                        help='Args can also be passed as a dict.')
    parser.add_argument('opts', help='see configs.py for all options',
                        default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    assert os.path.exists(args.output_dir), "Output directory does not exist!"
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
    assert_cfg()
    print_cfg()
    extract_features(args)


if __name__ == '__main__':
    main()
