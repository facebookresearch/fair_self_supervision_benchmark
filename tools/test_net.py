# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

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
import self_supervision_benchmark.metrics.metrics_topk as metrics_topk
import self_supervision_benchmark.metrics.metrics_ap as metrics_ap
from self_supervision_benchmark.utils import helpers, checkpoints
from self_supervision_benchmark.utils.timer import Timer
from self_supervision_benchmark.modeling import model_builder

from caffe2.python import workspace

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# import the custom ops provided by detectron
helpers.import_detectron_ops()


def test_net(opts):
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)

    prefix, device = helpers.get_prefix_and_device()

    ############################################################################
    name = '{}_test'.format(cfg.MODEL.MODEL_NAME)
    logger.info('=================Creating model: {}============='.format(name))
    data_type, batch_size = cfg['TEST'].DATA_TYPE, cfg['TEST'].BATCH_SIZE
    test_model = model_builder.ModelBuilder(
        name=name, train=False, use_cudnn=True, cudnn_exhaustive_search=True,
        split=data_type, ws_nbytes_limit=(cfg.CUDNN_WORKSPACE_LIMIT * 1024 * 1024),
    )

    test_model.build_model()
    test_model.create_net()
    test_model.start_data_loader()

    assert cfg.METRICS.TYPE in ['topk', 'AP'], "Invalid metrics type"
    test_metrics_calculator = None
    if cfg.METRICS.TYPE == 'topk':
        test_metrics_calculator = metrics_topk.TopkMetricsCalculator(
            model=test_model, split=data_type, batch_size=batch_size, prefix=prefix
        )
    else:
        test_metrics_calculator = metrics_ap.APMetricsCalculator(
            model=test_model, split=data_type, batch_size=batch_size, prefix=prefix
        )

    test_timer = Timer()
    total_test_iters = helpers.get_num_test_iter(test_model.input_db)
    logger.info('Test epoch iters: {}'.format(total_test_iters))

    # save proto for debugging
    helpers.save_model_proto(test_model)

    ############################################################################
    # initialize the model from the checkpoint
    if cfg.TEST.PARAMS_FILE:
        checkpoints.load_model_from_params_file(
            test_model, params_file=cfg.TEST.PARAMS_FILE, checkpoint_dir=None)
    else:
        logger.info('No params files specified for testing model. Aborting!')
        os._exit(0)

    ############################################################################
    logger.info("Testing model...")
    test_metrics_calculator.reset()
    for test_iter in range(0, total_test_iters):
        test_timer.tic()
        workspace.RunNet(test_model.net.Proto().name)
        test_timer.toc()
        if test_iter == 0:
            helpers.print_net(test_model)
        rem_test_iters = total_test_iters - test_iter - 1
        test_metrics_calculator.calculate_and_log_test_iter_metrics(
            test_iter, test_timer, rem_test_iters, total_test_iters
        )
    test_metrics_calculator.finalize_metrics()
    test_metrics_calculator.compute_and_log_epoch_best_metric(model_iter=test_iter)
    test_metrics_calculator.log_best_model_metrics(test_iter, total_test_iters)
    logger.info('Total images tested: {}'.format(test_metrics_calculator.split_N))
    logger.info('Done!!!')
    test_model.data_loader.shutdown_dataloader()


def main():
    parser = argparse.ArgumentParser(description='Model testing')
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
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
    assert_cfg()
    print_cfg()
    test_net(args)


if __name__ == '__main__':
    main()
