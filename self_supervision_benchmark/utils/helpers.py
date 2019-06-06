# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import logging
import math
import numpy as np
import subprocess

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.utils import env

from caffe2.python import workspace, scope, dyndep
from caffe2.proto import caffe2_pb2


# create the logger
logger = logging.getLogger(__name__)


def get_prefix_and_device():
    prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    device = caffe2_pb2.CUDA if cfg.DEVICE == 'GPU' else caffe2_pb2.CPU
    return prefix, device


def get_master_device_prefix(prefix):
    master_device = '{}{}'.format(prefix, 0)
    return master_device


def get_batch_size(split):
    batch_size = None
    if split in ['test', 'val']:
        if cfg.TEST.TEN_CROP:
            batch_size = int(cfg.TEST.BATCH_SIZE / (cfg.NUM_DEVICES * 10))
        else:
            batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_DEVICES)
    elif split == 'train':
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_DEVICES)
    return batch_size


def get_num_test_iter(db):
    if cfg.TEST.TEN_CROP:
        test_epoch_iter = int(math.ceil(
            db.get_db_size() / (float(cfg.TEST.BATCH_SIZE) / 10)
        ))
    else:
        test_epoch_iter = int(math.ceil(
            db.get_db_size() / float(cfg.TEST.BATCH_SIZE)
        ))
    return test_epoch_iter


def get_model_proto_directory():
    odir = os.path.abspath(os.path.join(
        cfg.CHECKPOINT.DIR, cfg.DATASET, cfg.MODEL.MODEL_NAME
    ))
    if not os.path.exists(odir):
        os.makedirs(odir)
    return odir


# save model protobuf for inspection later
def save_model_proto(model):
    net_proto = str(model.net.Proto())
    net_name = model.net.Proto().name
    init_net_proto = str(model.param_init_net.Proto())
    proto_path = get_model_proto_directory()
    net_proto_path = os.path.join(proto_path, net_name + "_net.pbtxt")
    init_net_proto_path = os.path.join(proto_path, net_name + "_init_net.pbtxt")
    with open(net_proto_path, 'w') as wfile:
        wfile.write(net_proto)
    with open(init_net_proto_path, 'w') as wfile:
        wfile.write(init_net_proto)
    logger.info("{}: Net proto saved to: {}".format(net_name, net_proto_path))
    logger.info("{}: Init net proto saved to: {}".format(
        net_name, init_net_proto_path))
    return None


def is_loss_nan(blob):
    loss = workspace.FetchBlob(blob)
    if math.isnan(loss):
        logger.error("ERROR: NaN losses on {}{}".format(blob))
        os._exit(0)


def check_nan_losses(num_devices):
    # if any of the losses is NaN, raise exception
    prefix, _ = get_prefix_and_device()
    for idx in range(num_devices):
        if isinstance(cfg.LOSS_BLOBS, list):
            for blob in cfg.LOSS_BLOBS:
                blob_name = prefix + str(idx) + '/' + blob
                is_loss_nan(blob_name)
        else:
            blob_name = prefix + str(idx) + '/loss'
            is_loss_nan(blob_name)
    return None


def get_flops_params(model, device_id=0):
    model_ops = model.net.Proto().op
    prefix, _ = get_prefix_and_device()
    master_device = prefix + str(device_id)
    param_ops = []
    for idx in range(len(model_ops)):
        op_type = model.net.Proto().op[idx].type
        op_input = model.net.Proto().op[idx].input[0]
        if op_type in ['Conv', 'FC'] and op_input.find(master_device) >= 0:
            param_ops.append(model.net.Proto().op[idx])

    num_flops = 0
    num_params = 0
    for idx in range(len(param_ops)):
        op = param_ops[idx]
        op_type = op.type
        op_inputs = param_ops[idx].input
        op_output = param_ops[idx].output[0]
        layer_flops = 0
        layer_params = 0
        if op_type == 'Conv':
            for op_input in op_inputs:
                if '_w' in op_input:
                    param_blob = op_input
                    param_sh = np.array(
                        workspace.FetchBlob(str(param_blob))).shape
                    layer_params = (
                        param_sh[0] * param_sh[1] * param_sh[2] * param_sh[3]
                    )
                    output_shape = np.array(
                        workspace.FetchBlob(str(op_output))).shape
                    layer_flops = layer_params * output_shape[2] * output_shape[3]
        elif op_type == 'FC':
            for op_input in op_inputs:
                if '_w' in op_input:
                    param_blob = op_input
                    param_shape = np.array(
                        workspace.FetchBlob(str(param_blob))).shape
                    layer_params = param_shape[0] * param_shape[1]
                    layer_flops = layer_params
        layer_params /= 1000000
        layer_flops /= 1000000000
        num_flops += layer_flops
        num_params += layer_params
    logger.info('Total network FLOPs (10^9): {}'.format(num_flops))
    logger.info('Total network params (10^6): {}'.format(num_params))
    return num_flops, num_params


def unscope_name(blob_name):
    return blob_name[blob_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]


def scoped_name(blob_name):
    return scope.CurrentNameScope() + blob_name


def HWC2CHW(image):
    if len(image.shape) == 3:
        return image.swapaxes(1, 2).swapaxes(0, 1)
    elif len(image.shape) == 4:
        # NHWC -> NCHW
        return image.swapaxes(2, 3).swapaxes(1, 2)
    else:
        logger.error('The image does not have correct dimensions')
        return None


def CHW2HWC(image):
    if len(image.shape) >= 3:
        return image.swapaxes(0, 1).swapaxes(1, 2)
    else:
        logger.error('Your image does not have correct dimensions')
        return None


def print_net(model):
    logger.info("Printing Model: {}".format(model.net.Name()))
    prefix, _ = get_prefix_and_device()
    master_device = get_master_device_prefix(prefix)
    op_output = model.net.Proto().op
    model_params = model.GetAllParams(master_device)
    for idx in range(len(op_output)):
        input_b = model.net.Proto().op[idx].input
        # for simplicity: only print the first output;
        # not recommended if there are split layers.
        output_b = str(model.net.Proto().op[idx].output[0])
        type_b = model.net.Proto().op[idx].type
        if output_b.find(master_device) >= 0:
            # Only print the forward pass network
            if output_b.find('grad') >= 0:
                break
            output_shape = np.array(workspace.FetchBlob(str(output_b))).shape
            first_blob = True
            suffix = ' ------- (op: {:s})'.format(type_b)
            for j in range(len(input_b)):
                if input_b[j] in model_params:
                        continue
                input_shape = np.array(
                    workspace.FetchBlob(str(input_b[j]))).shape
                if input_shape != ():
                    logger.info(
                        '{:28s}: {:20s} => {:36s}: {:20s}{}'.format(
                            unscope_name(str(input_b[j])),
                            '{}'.format(input_shape),  # suppress warning
                            unscope_name(str(output_b)),
                            '{}'.format(output_shape),
                            suffix
                        ))
                    if first_blob:
                        first_blob = False
                        suffix = ' ------|'
    logger.info("End of model: {}".format(model.net.Name()))


def get_gpu_stats():
    sp = subprocess.Popen(
        ['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode('utf-8').split('\n')
    out_dict = {}
    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except Exception:
            pass
    used_gpu_memory = out_dict['Used GPU Memory']
    return used_gpu_memory


# used in dataloader_benchmark only
def display_net_proto(net):
    print("Net Protobuf:\n{}".format(net.Proto()))
    return None


def import_detectron_ops():
    """
    We use the SigmoidCrossEntropyLoss provided by detectron ops. The ops are
    shipped as the binary libcaffe2_detectron_ops_gpu.so that we load using
    dyndep in Caffe2.
    """
    detectron_ops_lib = env.get_detectron_ops_lib()
    dyndep.InitOpsLibrary(detectron_ops_lib)
