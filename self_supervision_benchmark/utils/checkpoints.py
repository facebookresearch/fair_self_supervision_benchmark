# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
Helper functions to load and save model checkpoints.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import logging
import numpy as np
import os
import six
from collections import OrderedDict

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.utils import helpers
from self_supervision_benchmark.utils.io import load_object, save_object

from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2

# create the logger
logger = logging.getLogger(__name__)


def get_checkpoint_directory():
    if cfg.CHECKPOINT.DIR:
        odir = os.path.abspath(os.path.join(
            cfg.CHECKPOINT.DIR, cfg.DATASET, cfg.MODEL.MODEL_NAME
        ))
    else:
        raise Exception('No cfg.CHECKPOINT.DIR specified.')

    if not os.path.exists(odir):
        os.makedirs(odir)
    return odir


def save_model_params(model, params_file, checkpoint_dir, model_iter):
    logger.info("Saving model params to weights file {}".format(params_file))
    prefix, _ = helpers.get_prefix_and_device()
    master_device = helpers.get_master_device_prefix(prefix)
    save_params = [str(param) for param in model.GetParams(master_device)]
    save_computed_params = [
        str(param) for param in model.GetComputedParams(master_device)]
    save_blobs = {}
    # also save total model iterations so far
    save_blobs['model_iter'] = model_iter + 1
    save_blobs['lr'] = workspace.FetchBlob(prefix + '{}/lr'.format(0))
    # save param momentum as well
    for param in save_params:
        for scoped_blob_name in get_momentum_blobs(param):
            unscoped_blob_name = helpers.unscope_name(scoped_blob_name)
            if unscoped_blob_name not in save_blobs:
                if workspace.HasBlob(scoped_blob_name):
                    data = workspace.FetchBlob(scoped_blob_name)
                    save_blobs[unscoped_blob_name] = data

    for param in save_params + save_computed_params:
        scoped_blob_name = str(param)
        unscoped_blob_name = helpers.unscope_name(scoped_blob_name)
        if unscoped_blob_name not in save_blobs:
            data = workspace.FetchBlob(scoped_blob_name)
            save_blobs[unscoped_blob_name] = data
    save_object(dict(blobs=save_blobs), params_file)


# This function looks at all the iters checkpointed and returns latest iter file
def get_checkpoint_resume_file(checkpoint_dir):
    all_files = os.listdir(checkpoint_dir)
    all_iters = []
    for f in all_files:
        if 'c2_model_iter' in f:
            iter_num = int(f.replace('.pkl', '').replace('c2_model_iter', ''))
            all_iters.append(iter_num)
    if len(all_iters) > 0:
        all_iters.sort(reverse=True)
        last_iter = int(all_iters[0])
        filepath = os.path.join(
            checkpoint_dir, 'c2_model_iter{}.pkl'.format(last_iter)
        )
        return filepath
    else:
        return None


def load_model_from_params_file(model, params_file, checkpoint_dir=None):
    # in case of cluster failure, we should resume from the last checkpoint
    # than the params_file if specified
    checkpoint_exists = False
    if checkpoint_dir is not None:
        checkpointed_files = os.listdir(checkpoint_dir)
        for f in checkpointed_files:
            if f.endswith('.pkl'):
                checkpoint_exists = True
                break
    prev_lr = None
    if params_file and os.path.exists(params_file) and not checkpoint_exists:
        logger.info('Model init from params file: {}'.format(params_file))
        start_model_iter, prev_lr = initialize_params_from_file(
            model=model, weights_file=params_file, num_devices=cfg.NUM_DEVICES,
        )
    elif cfg.CHECKPOINT.RESUME and cfg.CHECKPOINT.CHECKPOINT_ON:
        start_model_iter = 0
        params_file = get_checkpoint_resume_file(checkpoint_dir)
        if params_file is not None and os.path.exists(params_file):
            logger.info('Model init from checkpoint: {}'.format(params_file))
            start_model_iter, prev_lr = initialize_params_from_file(
                model=model, weights_file=params_file,
                num_devices=cfg.NUM_DEVICES,
            )
        else:
            logger.info('Params file does not exist: {}'.format(params_file))
    return start_model_iter, prev_lr, checkpoint_exists


def get_momentum_blobs(param):
    momentum_blobs = []
    momentum_blobs.append(str(param) + '_momentum')
    return momentum_blobs


def initialize_master_device_model_params(model, weights_file):
    ws_blobs = workspace.Blobs()
    if weights_file.endswith('npy'):
        # The default encoding in Python 2 is ascii; in Python 3 it is utf-8.
        # latin1 (a.k.a., ISO-8859-1) is a superset of ascii. That's why loading
        # ascii-encoded strings with latin1 works and gives the same result as
        # loading it with ascii. https://stackoverflow.com/a/38316603
        if six.PY2:
            blobs = np.load(weights_file)[()]
        else:
            blobs = np.load(weights_file, allow_pickle=True, encoding='latin1')[()]
    else:
        blobs = load_object(weights_file)

    if 'blobs' in blobs:
        blobs = blobs['blobs']

    # Return the model iter from which training should start
    model_iter = 0
    if 'model_iter' in blobs:
        model_iter = blobs['model_iter']
    prev_lr = None
    if 'lr' in blobs:
        prev_lr = blobs['lr']

    # initialize params, params momentum, computed params
    unscoped_blob_names = OrderedDict()
    if 'test' not in model.net.Name():
        for param in model.params:
            for mom_blob in get_momentum_blobs(param):
                unscoped_blob_names[helpers.unscope_name(mom_blob)] = True
    # NOTE: Currently GetAllParams() and GetAllParams('') both work. Use any.
    for blob in model.GetAllParams():
        unscoped_blob_names[helpers.unscope_name(str(blob))] = True

    device = caffe2_pb2.CUDA if cfg.DEVICE == 'GPU' else caffe2_pb2.CPU
    with core.NameScope('gpu_{}'.format(0)):
        with core.DeviceScope(core.DeviceOption(device, 0)):
            for unscoped_blob_name in unscoped_blob_names.keys():
                scoped_blob_name = helpers.scoped_name(unscoped_blob_name)
                if unscoped_blob_name not in blobs:
                    logger.info('{:s} not found'.format(unscoped_blob_name))
                    continue
                logger.info('{:s} loaded from weights file into: {:s}'.format(
                    unscoped_blob_name, scoped_blob_name
                ))
                # match the shape of the blob loaded and the one existing in
                # workspace
                if scoped_blob_name in ws_blobs:
                    ws_blob = workspace.FetchBlob(scoped_blob_name)
                    assert ws_blob.shape == blobs[unscoped_blob_name].shape, \
                        ('Workspace blob {} with shape {} does not match '
                         'weights file shape {}').format(
                            unscoped_blob_name, ws_blob.shape,
                            blobs[unscoped_blob_name].shape)
                data = blobs[unscoped_blob_name].astype(np.float32, copy=False)
                workspace.FeedBlob(scoped_blob_name, data)
    return model_iter, prev_lr


def broadcast_parameters(model, num_devices):
    if num_devices == 1:
        return
    prefix, device = helpers.get_prefix_and_device()
    master_device = helpers.get_master_device_prefix(prefix)
    # get all params to initialize: model params + param momentum
    all_model_params = model.GetAllParams(master_device)
    all_params_momentum = []
    if 'test' not in model.net.Name():
        for param in model.GetParams(master_device):
            for mom_blob in get_momentum_blobs(param):
                all_params_momentum.append(mom_blob)
    all_params = all_model_params + all_params_momentum

    # load the value of params
    for param in all_params:
        if workspace.HasBlob(str(param)):
            data = workspace.FetchBlob(str(param))
            unscoped_name = helpers.unscope_name(str(param))
            logger.info('Broadcasting {} to'.format(str(param)))
            for idx in range(1, num_devices):
                with core.NameScope(prefix + str(idx)):
                    with core.DeviceScope(core.DeviceOption(device, idx)):
                        device_scoped_name = helpers.scoped_name(unscoped_name)
                        if workspace.HasBlob(device_scoped_name):
                            logger.info(' |-> {}'.format(device_scoped_name))
                            workspace.FeedBlob(device_scoped_name, data)
                        else:
                            logger.info('Blob non-existent. Not broadcasting')


# initialize the model from a file and broadcast the parameters to all_gpus
# if num_devices > 1
def initialize_params_from_file(model, weights_file, num_devices):
    model_iter, prev_lr = initialize_master_device_model_params(
        model, weights_file
    )
    broadcast_parameters(model, num_devices)
    return model_iter, prev_lr
