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

import self_supervision_benchmark.metrics.metrics_helper as metrics_helper
from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.core.config import (
    cfg_from_file, cfg_from_list, assert_cfg, print_cfg
)
from self_supervision_benchmark.utils import helpers, checkpoints, lr_utils
from self_supervision_benchmark.utils.timer import Timer
from self_supervision_benchmark.modeling import model_builder

from caffe2.python import workspace

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# import the custom ops provided by detectron
helpers.import_detectron_ops()


def build_wrapper(is_train, prefix):
    if is_train:
        name = '{}_train'.format(cfg.MODEL.MODEL_NAME)
        data_type = cfg['TRAIN'].DATA_TYPE
        batch_size = cfg['TRAIN'].BATCH_SIZE
    else:
        name = '{}_test'.format(cfg.MODEL.MODEL_NAME)
        data_type = cfg['TEST'].DATA_TYPE
        batch_size = cfg['TEST'].BATCH_SIZE

    logger.info('=================Creating model: {}============='.format(name))
    model = model_builder.ModelBuilder(
        name=name, train=is_train, use_cudnn=True, cudnn_exhaustive_search=True,
        ws_nbytes_limit=(cfg.CUDNN_WORKSPACE_LIMIT * 1024 * 1024),
        split=data_type
    )

    model.build_model()
    model.create_net()
    model.start_data_loader()

    metrics_calculator = model.get_metrics_calculator(data_type, batch_size, prefix)

    timer = Timer()
    return model, metrics_calculator, timer


def train_net(opts):
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logging.getLogger(__name__)
    np.random.seed(cfg.RNG_SEED)

    prefix, device = helpers.get_prefix_and_device()
    device_prefix = '{}{}'.format(prefix, 0)

    ############################################################################
    total_test_iters, test_metrics_calculator = 0, None
    if cfg.MODEL.TEST_MODEL:
        # Build test_model: we do this first so we don't overwrite init (if any)
        test_model, test_metrics_calculator, test_timer = build_wrapper(
            is_train=False, prefix=device_prefix
        )
        total_test_iters = helpers.get_num_test_iter(test_model.input_db)
        logger.info('Test epoch iters: {}'.format(total_test_iters))

    ############################################################################
    # create training model and metrics
    train_model, train_metrics_calculator, train_timer = build_wrapper(
        is_train=True, prefix=device_prefix
    )
    # save proto for debugging
    helpers.save_model_proto(train_model)

    ############################################################################
    # setup the checkpoint directory and load model from checkpoint
    if cfg.CHECKPOINT.CHECKPOINT_ON:
        checkpoint_dir = checkpoints.get_checkpoint_directory()
        logger.info('Checkpoint directory: {}'.format(checkpoint_dir))

    # checkpoint_exists variable is used to track whether this is a model
    # resume from a failed training or not. It will look up in the checkpoint_dir
    # whether there are any model checkpoints available. If yes, then it is set
    # to True.
    start_model_iter, prev_checkpointed_lr, checkpoint_exists = 0, None, False
    if cfg.CHECKPOINT.RESUME or cfg.TRAIN.PARAMS_FILE:
        start_model_iter, prev_checkpointed_lr, checkpoint_exists = (
            checkpoints.load_model_from_params_file(
                train_model, params_file=cfg.TRAIN.PARAMS_FILE,
                checkpoint_dir=checkpoint_dir
            )
        )
        # if we are fine-tuning, it's possible that the training might get
        # stopped/killed
        if cfg.MODEL.FINE_TUNE and not checkpoint_exists:
            start_model_iter = 0

    ############################################################################
    logger.info("=> Training model...")
    model_flops, model_params = 0, 0
    lr_iters = lr_utils.get_lr_steps()
    for curr_iter in range(start_model_iter, cfg.SOLVER.NUM_ITERATIONS):
        # set LR
        lr_utils.add_variable_stepsize_lr(
            curr_iter + 1, cfg.NUM_DEVICES, lr_iters, start_model_iter + 1,
            cfg.TRAIN.EVALUATION_FREQUENCY, train_model, prev_checkpointed_lr
        )

        # run the model training iteration
        train_timer.tic()
        workspace.RunNet(train_model.net.Proto().name)
        train_timer.toc(average=False)

        # logging after 1st iteration
        if curr_iter == start_model_iter:
            helpers.print_net(train_model)
            os.system('nvidia-smi')
            model_flops, model_params = helpers.get_flops_params(train_model)

        # check nan loses
        helpers.check_nan_losses(cfg.NUM_DEVICES)

        # log metrics at the cfg.LOGGER_FREQUENCY
        rem_train_iters = cfg.SOLVER.NUM_ITERATIONS - curr_iter - 1
        train_metrics_calculator.calculate_and_log_train_iter_metrics(
            curr_iter, train_timer, rem_train_iters, cfg.SOLVER.NUM_ITERATIONS,
            train_model.data_loader.minibatch_queue_size()
        )

        # checkpoint model at CHECKPOINT_PERIOD
        if (
            cfg.CHECKPOINT.CHECKPOINT_ON
            and (curr_iter + 1) % cfg.CHECKPOINT.CHECKPOINT_PERIOD == 0
        ):
            params_file = os.path.join(
                checkpoint_dir, 'c2_model_iter{}.pkl'.format(curr_iter + 1)
            )
            checkpoints.save_model_params(
                model=train_model, params_file=params_file,
                model_iter=curr_iter, checkpoint_dir=checkpoint_dir
            )

        if (curr_iter + 1) % cfg.TRAIN.EVALUATION_FREQUENCY == 0:
            train_metrics_calculator.finalize_metrics()
            # test model if the testing is ON
            if cfg.MODEL.TEST_MODEL:
                test_metrics_calculator.reset()
                logger.info("=> Testing model...")
                for test_iter in range(0, total_test_iters):
                    # run a test iteration
                    test_timer.tic()
                    workspace.RunNet(test_model.net.Proto().name)
                    test_timer.toc()
                    rem_test_iters = (total_test_iters - test_iter - 1)
                    num_rem_iter = (cfg.SOLVER.NUM_ITERATIONS - curr_iter - 1)
                    num_rem_ep = num_rem_iter / cfg.TRAIN.EVALUATION_FREQUENCY
                    if (test_iter + 1) % cfg.LOGGER_FREQUENCY == 0:
                        rem_test_iters += int(total_test_iters * num_rem_ep)
                    test_metrics_calculator.calculate_and_log_test_iter_metrics(
                        test_iter, test_timer, rem_test_iters, total_test_iters
                    )
                test_metrics_calculator.finalize_metrics()
                test_metrics_calculator.compute_and_log_epoch_best_metric(
                    model_iter=curr_iter
                )
            json_stats = metrics_helper.get_json_stats_dict(
                train_metrics_calculator, test_metrics_calculator, curr_iter,
                model_flops, model_params,
            )

            json_stats['average_time'] = round(
                train_timer.average_time + test_timer.average_time, 3
            )
            metrics_helper.print_json_stats(json_stats)
            train_metrics_calculator.reset()

    if test_metrics_calculator is not None:
        test_metrics_calculator.log_best_model_metrics(
            model_iter=curr_iter, total_iters=cfg.SOLVER.NUM_ITERATIONS,
        )

    train_model.data_loader.shutdown_dataloader()
    if cfg.MODEL.TEST_MODEL:
        test_model.data_loader.shutdown_dataloader()

    logger.info('Training has successfully finished...exiting!')
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description='Self-Supervision Benchmark')
    parser.add_argument('--config_file', type=str, default=None, required=True,
                        help='Optional config file for params')
    parser.add_argument('--args_dict', type=str, default=None,
                        help='Args can also be passed as a dict.')
    parser.add_argument('opts', help='see config.py for all options',
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
    train_net(args)


if __name__ == '__main__':
    main()
