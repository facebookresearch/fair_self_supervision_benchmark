# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This script contains helpful functions to build the multi-device training/testing
model
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import logging

import self_supervision_benchmark.metrics.metrics_ap as metrics_ap
import self_supervision_benchmark.metrics.metrics_topk as metrics_topk
from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.data.dataloader import DataLoader, get_input_db
from self_supervision_benchmark.modeling.jigsaw import (
    alexnet_jigsaw_finetune_full, alexnet_jigsaw_finetune_linear,
    resnet_jigsaw_finetune_full, resnet_jigsaw_finetune_linear,
)
from self_supervision_benchmark.modeling.colorization import (
    alexnet_colorize_finetune_full, alexnet_colorize_finetune_linear,
    resnet_colorize_finetune_full, resnet_colorize_finetune_linear,
)
from self_supervision_benchmark.modeling.supervised import (
    caffenet_bvlc_supervised_finetune_full,
    caffenet_bvlc_supervised_finetune_linear, resnet_supervised_finetune_full,
    resnet_supervised_finetune_linear,
)
from self_supervision_benchmark.utils import helpers, lr_utils

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, scope, core, cnn, data_parallel_model

# create the logger
logger = logging.getLogger(__name__)


# To add new models, import them, add them to this map and models/TARGETS.
# The model file should define a create_model() function that creates the model.
model_creator_map = {
    'resnet_colorize_finetune_full': resnet_colorize_finetune_full,
    'resnet_colorize_finetune_linear': resnet_colorize_finetune_linear,
    'resnet_jigsaw_finetune_full': resnet_jigsaw_finetune_full,
    'resnet_jigsaw_finetune_linear': resnet_jigsaw_finetune_linear,
    'resnet_supervised_finetune_full': resnet_supervised_finetune_full,
    'resnet_supervised_finetune_linear': resnet_supervised_finetune_linear,
    'alexnet_colorize_finetune_full': alexnet_colorize_finetune_full,
    'alexnet_colorize_finetune_linear': alexnet_colorize_finetune_linear,
    'alexnet_jigsaw_finetune_full': alexnet_jigsaw_finetune_full,
    'alexnet_jigsaw_finetune_linear': alexnet_jigsaw_finetune_linear,
    'caffenet_bvlc_supervised_finetune_full': caffenet_bvlc_supervised_finetune_full,
    'caffenet_bvlc_supervised_finetune_linear':
        caffenet_bvlc_supervised_finetune_linear,
}


class ModelBuilder(cnn.CNNModelHelper):

    def __init__(self, **kwargs):
        kwargs['order'] = 'NCHW'
        self.train = kwargs.get('train', False)
        self.split = kwargs.get('split', 'train')
        if 'train' in kwargs:
            del kwargs['train']
        if 'split' in kwargs:
            del kwargs['split']
        super(ModelBuilder, self).__init__(**kwargs)
        self.do_not_update_params = []
        self.data_loader = None
        self.input_db = None

    def ConvShared(
        self, blob_in, blob_out, dim_in, dim_out, kernel, weight=None,
        bias=None, **kwargs
    ):
        """
        Add conv op that shares weights and/or biases with another Conv op.
        """
        use_bias = (
            False if ('no_bias' in kwargs and kwargs['no_bias']) else True
        )

        if use_bias:
            blobs_in = [blob_in, weight, bias]
        else:
            blobs_in = [blob_in, weight]

        if 'no_bias' in kwargs:
            del kwargs['no_bias']

        return self.net.Conv(
            blobs_in, blob_out, kernel=kernel, order='NCHW', **kwargs
        )

    def TrainableParams(self, scope=''):
        return [
            param for param in self.params
            if (
                param in self.param_to_grad   # param has a gradient
                and param not in self.do_not_update_params  # not on blacklist
                and (scope == '' or str(param).find(scope) == 0)
                # filter for gpu assignment, if device_id set
            )
        ]

    def build_model(self):
        self.input_db = get_input_db(
            dataset=cfg.DATASET, data_type=self.split
        )
        batch_size = helpers.get_batch_size(self.split)
        self.data_loader = DataLoader(
            split=self.split, input_db=self.input_db, batch_size=batch_size,
            preprocess=True
        )
        self.create_data_parallel_model(
            model=self, train=self.train, db_loader=self.data_loader,
            split=self.split
        )

    def create_data_parallel_model(self, model, db_loader, split, train=True):
        forward_pass_builder_fun = create_model(model=self, split=split)
        input_builder_fun = add_inputs(model=model, data_loader=db_loader)
        if train:
            param_update_builder_fun = add_parameter_update_ops(model=model)
        else:
            param_update_builder_fun = None
            lr_utils.create_learning_rate_blob()
        devices = range(0, cfg.NUM_DEVICES)
        parallelize_fn = (
            data_parallel_model.Parallelize_GPU if cfg.DEVICE == 'GPU'
            else data_parallel_model.Parallelize_CPU
        )
        parallelize_fn(
            model,
            input_builder_fun=input_builder_fun,
            forward_pass_builder_fun=forward_pass_builder_fun,
            param_update_builder_fun=param_update_builder_fun,
            devices=devices,
            rendezvous=None,
            broadcast_computed_params=False,
            optimize_gradient_memory=cfg.MODEL.MEMONGER,
            use_nccl=True,
        )

    def create_net(self):
        workspace.RunNetOnce(self.param_init_net)
        workspace.CreateNet(self.net)

    def start_data_loader(self):
        self.data_loader.register_sigint_handler()
        self.data_loader.start()
        self.data_loader.prefill_minibatch_queue()

    def get_metrics_calculator(self, data_type, batch_size, prefix, generate_json=0):
        assert cfg.METRICS.TYPE in ['topk', 'AP'], "Invalid metrics type"
        if cfg.METRICS.TYPE == 'topk':
            metrics_calculator = metrics_topk.TopkMetricsCalculator(
                model=self, split=data_type, batch_size=batch_size, prefix=prefix,
                generate_json=generate_json
            )
        else:
            metrics_calculator = metrics_ap.APMetricsCalculator(
                model=self, split=data_type, batch_size=batch_size, prefix=prefix
            )
        return metrics_calculator


def create_model(model, split):
    """
    Call the create_model() method of the desired model that defines the model.
    """
    model_name = cfg.MODEL.MODEL_NAME
    assert model_name in model_creator_map.keys(), \
        'Unknown model_type {}'.format(model_name)

    def model_creator(model, loss_scale):
        model, pred, loss = model_creator_map[model_name].create_model(
            model=model, data="data", labels="labels", split=split,
        )
        if not isinstance(loss, list):
            return [loss]
        else:
            return loss

    return model_creator


def add_inputs(model, data_loader):
    """
    The blobs queue per gpu has the input data blobs. We want to get the blobs
    from the queue and train model on it. Create the blobs for the gpu which
    will be populated with the data.
    """
    blob_names = data_loader.get_blob_names()
    queue_name = data_loader._blobs_queue_name

    def input_fn(model):
        for blob_name in blob_names:
            workspace.CreateBlob(scope.CurrentNameScope() + blob_name)
        model.net.DequeueBlobs(queue_name, blob_names)
        model.StopGradient('data', 'data')

    return input_fn


def add_parameter_update_ops(model):
    lr_utils.create_learning_rate_blob()
    with core.NameScope("cpu"):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            model.Iter("ITER")

    def param_update_ops(model):
        weight_decay = model.param_init_net.ConstantFill(
            [], 'weight_decay', shape=[1], value=cfg.SOLVER.WEIGHT_DECAY
        )
        weight_decay_bn = model.param_init_net.ConstantFill(
            [], 'weight_decay_bn', shape=[1], value=cfg.SOLVER.WEIGHT_DECAY_BN
        )
        # for jigsaw model, all the bias params have weight decay set to
        weight_decay_zero_bias = model.param_init_net.ConstantFill(
            [], 'weight_decay_zero_bias', shape=[1], value=0.0
        )
        zero = model.param_init_net.ConstantFill(
            [], "ZERO", shape=[1], value=0.0
        )
        one = model.param_init_net.ConstantFill(
            [], "ONE", shape=[1], value=1.0
        )
        two = model.param_init_net.ConstantFill(
            [], "TWO", shape=[1], value=2.0
        )
        params = model.GetParams()
        curr_scope = scope.CurrentNameScope()
        # scope is of format 'gpu_{}/'.format(device_id), so remove the separator
        trainable_params = model.TrainableParams(curr_scope[:-1])
        assert len(params) > 0, 'No trainable params found in model'
        for param in params:
            # only update trainable params
            if param in trainable_params:
                # the param grad is the summed gradient for the parameter across
                # all devices/hosts
                param_momentum = model.param_init_net.ConstantFill(
                    [param], param + '_momentum', value=0.0
                )
                param_grad = model.param_to_grad[param]
                # add weight decay
                if '_bn' in str(param):
                    # make LR 0 and weight decay 0 to keep scale and bias same.
                    # Scale/bias are the learnable parameters in BN. See
                    # Algorithm1 https://arxiv.org/pdf/1502.03167.pdf
                    if cfg.MODEL.BN_NO_SCALE_SHIFT:
                        model.WeightedSum(
                            [param_grad, zero, param, weight_decay_bn],
                            param_grad
                        )
                    else:
                        model.WeightedSum(
                            [param_grad, one, param, weight_decay_bn],
                            param_grad
                        )
                elif cfg.MODEL.NO_BIAS_DECAY:
                    # In jigsaw model, all the bias params have decay=0 and
                    # lr_multiplier=2
                    if '_b' in str(param):
                        model.WeightedSum([
                            param_grad, two, param, weight_decay_zero_bias
                        ], param_grad)
                else:
                    model.WeightedSum(
                        [param_grad, one, param, weight_decay], param_grad
                    )
                model.net.MomentumSGDUpdate(
                    [param_grad, param_momentum, 'lr', param],
                    [param_grad, param_momentum, param],
                    momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV,
                )
    return param_update_ops
