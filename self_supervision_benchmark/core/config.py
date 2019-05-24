# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This file specifies the config options for benchmarking. You should write a yaml
file which specifies the values for parameters in this file. Do NOT change the
default values in this file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import logging
import numpy as np
import pprint
import six
import yaml
from ast import literal_eval

from self_supervision_benchmark.utils.collections import AttrDict
from self_supervision_benchmark.utils import env
from self_supervision_benchmark.utils.io import cache_url

# create the logger
logger = logging.getLogger(__name__)

__C = AttrDict()
config = __C

# Model options
__C.MODEL = AttrDict()
# during the model training, whether to test the model on test/val data or not.
# the model is tested at EVALUATION_FREQUENCY.
__C.MODEL.TEST_MODEL = True
# Depth for ResNet model. Only supported 50 in this benchmark. Can be very
# very easily extended to larger depths.
__C.MODEL.DEPTH = 50
# number of classes in the model (used in softmax)
__C.MODEL.NUM_CLASSES = -1
# one can create many model definitions and given the definition a key name that
# maps to the model definition. See modeling/model_builder.py
__C.MODEL.MODEL_NAME = ''
# if we are fine-tuning by initializing weights, we normally capture the
# number of iterations from the weights file (saved by checkpoints.py). In case
# of fine-tuning, we want to avoid the number of iterations and start from 0.
__C.MODEL.FINE_TUNE = True
# Useful for saving memory for ResNet models. Disable the inplace relu for stats
__C.MODEL.ALLOW_INPLACE_RELU = True
# Memory optimizer for caffe2 graphs. The memory is optimized by recycling the
# blobs during the backwards pass. This can help fit bigger models in GPU
# memory if that's a constraint.
__C.MODEL.MEMONGER = True
# options to optimize memory usage. "Sum" is the operation in the ResNet model
# when the shortcut connection and output of bottleneck block is summed. We can
# perform the sum inplace to save memory.
__C.MODEL.ALLOW_INPLACE_SUM = True
# whether to apply weight decay on the bias parameters in the model or not.
# By default, we want to apply weight decay and hence set to False.
__C.MODEL.NO_BIAS_DECAY = False
# Batch Normalization has two learnable parameters during model training (see
# Algorithm1 https://arxiv.org/abs/1502.03167). If we want to make these params
# non-learnable, we can initialize scale=1.0, shift=0 and set the LR=0, weight
# decay=0 for these parameters. By default, we want to learn these parameters
# and hence the value to False.
__C.MODEL.BN_NO_SCALE_SHIFT = False
# when training linear classifiers on a transfer task (in benchmark suite), we
# want to use Batch norm in test mode in the model. This parameters ensures
# the BN is set to test mode. Set to False to tune the BN as well.
__C.MODEL.FORCE_BN_TEST_MODE = True
# standard setting for Batch norm momentum and epsilon values.
__C.MODEL.BN_MOMENTUM = 0.9
__C.MODEL.BN_EPSILON = 1.0000001e-5
# Inspired from the ImageNet-1hr work https://arxiv.org/abs/1706.02677. We can
# custom set the gamma for BatchNorm operation after 3rd conv layer in ResNet
# bottleneck blob. Normally, the gamma is initialized to 1.0 but the work above
# reasons to initialize it to 0.0.
__C.MODEL.CUSTOM_BN_INIT = False
__C.MODEL.BN_INIT_GAMMA = 1.0
# type of initialization from the FC layer. Only applicable in case of full
# finetuning tasks.
# possible values: gaussian | xavier | msra
__C.MODEL.FC_INIT_TYPE = 'gaussian'
# for the gaussian FC init type, what std we want to set. Mean is 0 by default.
__C.MODEL.FC_INIT_STD = 0.01
# for training svms, we want to extract the features from the model at various
# different levels.
__C.MODEL.EXTRACT_FEATURES_ONLY = False
# during the feature extraction, specify the blob names which we want to extract
__C.MODEL.EXTRACT_BLOBS = []


# Training options
__C.TRAIN = AttrDict()
__C.TRAIN.DATA_TYPE = 'train'
# the model weights file to initialize the model from (if we want to init).
__C.TRAIN.PARAMS_FILE = ''
# Number of iterations after which model should be tested on test/val data.
# preferably set it to #iterations = 1 epoch of train data. For ImageNet-1K,
# 1 epoch = 5005 iterations (minibatch size = 256).
__C.TRAIN.EVALUATION_FREQUENCY = 5005
# minibatch size for training. Default is 32 img/gpu and 8 gpus
__C.TRAIN.BATCH_SIZE = 256
# crop the image of specified dimension from the full image.
__C.TRAIN.CROP_SIZE = 224
# the .npy file containing information about the images in the dataset. More
# specifically where to read the images from.
__C.TRAIN.DATA_FILE = ''
# the .npy containing the labels for each image in the DATA_FILE. The labels
# should map to the images.
__C.TRAIN.LABELS_FILE = ''
# scale the shortest image side to SCALE value
__C.TRAIN.SCALE = 256
# randomly select the scale from [1/max_size, 1/min_size] - ResNet style jitter
__C.TRAIN.RANDOM_SCALE_JITTER = [256, 480]
# resize the full image to the square image of the specified size.
__C.TRAIN.GLOBAL_RESIZE_VALUE = 224
# Random sized crop involves cropping out 8% - 100% of image area with a certain
# aspect ratio (GoogleNet paper). IMG_CROP_MIN_AREA specifies the minimum area
# and IMG_CROP_MAX_AREA specifies the maximum area from the given image.
__C.TRAIN.IMG_CROP_MIN_AREA = 0.08
__C.TRAIN.IMG_CROP_MAX_AREA = 1.0
# Transforms just modifies the image like cropping.
# possible values: [scale | global_resize | random_scale_jitter | center_crop |
#                   random_crop | random_sized_crop | horizontal_flip]
# Defaults set to ResNet on IN training for compatibility
__C.TRAIN.DATA_TRANSFORMS = ['random_sized_crop', 'horizontal_flip']
# Data Processing augments the data
# values: [color_jitter | lighting | color_normalization | mean_normalization]
# Defaults set to ResNet-50 on IN-1K training for reproducibility.
__C.TRAIN.DATA_PROCESSING = ['color_jitter', 'lighting', 'color_normalization']


# Test options
__C.TEST = AttrDict()
# val | test
__C.TEST.DATA_TYPE = 'val'
# the file containing information about the images. More precisely, image paths
__C.TEST.DATA_FILE = ''
# the file containing labels for images in DATA_FILE
__C.TEST.LABELS_FILE = ''
# whether to perform 10-crop evaluation or not.
__C.TEST.TEN_CROP = False
# whether to generate ten-crops randomly or standard way
# possible values: 'random' | 'standard'
__C.TEST.TEN_CROP_TYPE = 'random'
# scale the smallest image side to the value
__C.TEST.SCALE = 256
# resize the full test image to the square image of the specified size
__C.TEST.CROP_SIZE = 224
# batch size for testing - over 8 gpus [32 images per gpu]
__C.TEST.BATCH_SIZE = 256
# the model weights file to test the model on (if we want to test) or extract
# features from.
__C.TEST.PARAMS_FILE = ''


# Checkpoint options
__C.CHECKPOINT = AttrDict()
# whether to checkpoint the model or not. Recommended to set to True to not lose
# the training progress in case training is killed for some reason.
__C.CHECKPOINT.CHECKPOINT_ON = True
# how ofter should the model be checkpointed. Specify the number of iterations.
# Normally, choose iterations = 1 epoch.
__C.CHECKPOINT.CHECKPOINT_PERIOD = -1
# during training, if the training fails for some reason and restarts, should
# the model resume training from the last saved checkpoint.
__C.CHECKPOINT.RESUME = True
# the disk path where model checkpoints will be saved. The path should exist.
__C.CHECKPOINT.DIR = '.'


# SOLVER
__C.SOLVER = AttrDict()
__C.SOLVER.NESTEROV = True
__C.SOLVER.BASE_LR = 0.1
__C.SOLVER.SCALE_MOMENTUM = False
__C.SOLVER.GAMMA = 0.1
# For imagenet1k, 150150 = 30 epochs for batch size of 256 images over 8 gpus
__C.SOLVER.STEP_SIZES = [150150, 150150, 150150]
# Total number of iterations for which model should train.
# For above batch size, running 90 epochs = 450450 iterations
__C.SOLVER.NUM_ITERATIONS = 450450
__C.SOLVER.WEIGHT_DECAY = 0.0001
# inspired from the ImageNet-1hour work https://arxiv.org/abs/1706.02677
__C.SOLVER.WEIGHT_DECAY_BN = 0.0001
__C.SOLVER.MOMENTUM = 0.9


# Metrics calculation
__C.METRICS = AttrDict()
# possible values: topk | AP
# topk corresponds to top1, top5 metrics commonly used in classification
# problems. AP metrics is used for VOC classification like tasks which are
# multi-label.
__C.METRICS.TYPE = 'topk'


# Colorization model.
__C.COLORIZATION = AttrDict()
# The input is LAB image and data processing requires conversion to LAB space.
# Use this flag to enable the data processing for the colorization task.
__C.COLORIZATION.COLOR_ON = False


# GPU only
__C.DEVICE = 'GPU'
__C.NUM_DEVICES = 8
# dataset = imagenet1k | places205 | coco | voc2007
__C.DATASET = ''
# specify the name of the loss blobs for further calculation, metrics report etc
__C.LOSS_BLOBS = []
# specify the name of the accuracy blobs for further calculation, metrics
# reporting etc.
__C.ACCURACY_BLOBS = []
# Caffe2 specific
__C.CUDNN_WORKSPACE_LIMIT = 256
# random seed to use. This is set at the beginning of training in train_net.py
__C.RNG_SEED = 2
# number of iterations after which we log stats during training/testing. This
# can be helpful to keep track of training and check the training progress.
__C.LOGGER_FREQUENCY = 10

# Download the models specified as urls to the DOWNLOAD_CACHE.
__C.DOWNLOAD_CACHE = '/tmp/fair_self_supervision_benchmark_cache/'


def merge_dicts(dict_a, dict_b, stack=None):
    for key, value_ in dict_a.items():
        full_key = '.'.join(stack) + '.' + key if stack is not None else key
        if key not in dict_b:
            raise KeyError('Invalid key in config file: {}'.format(key))
        value = copy.deepcopy(value_)
        value = _decode_cfg_value(value)
        value = _check_and_coerce_cfg_value_type(value, dict_b[key], key, full_key)

        # recursively merge dicts
        if isinstance(value, AttrDict):
            try:
                stack_push = [key] if stack is None else stack + [key]
                merge_dicts(dict_a[key], dict_b[key], stack_push)
            except BaseException:
                logger.critical('Error under config key: {}'.format(key))
                raise
        else:
            dict_b[key] = value


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    with open(filename, 'r') as fopen:
        yaml_config = AttrDict(env.yaml_load(fopen))
    merge_dicts(yaml_config, __C)


def print_cfg():
    logger.info('Training with config:')
    logger.info(pprint.pformat(__C))


def cache_cfg_urls():
    """
    If we have urls in the config file, cache them locally and point the cfg to
    use those cache file instead now.
    """
    __C.TRAIN.PARAMS_FILE = cache_url(__C.TRAIN.PARAMS_FILE, __C.DOWNLOAD_CACHE)
    __C.TEST.PARAMS_FILE = cache_url(__C.TEST.PARAMS_FILE, __C.DOWNLOAD_CACHE)


def assert_cfg():
    # this function should check for the validity of arguments
    assert __C.TRAIN.BATCH_SIZE % __C.NUM_DEVICES == 0, \
        "Batch size should be multiple of num_devices"
    if __C.TEST.TEN_CROP:
        assert __C.TEST.BATCH_SIZE % (__C.NUM_DEVICES * 10) == 0, \
            'test minibatch size is not multiple of (num_devices * 10)'
    else:
        assert __C.TEST.BATCH_SIZE % __C.NUM_DEVICES == 0, \
            'test minibatch size is not multiple of num_devices'
    cache_cfg_urls()


def cfg_from_list(args_list):
    """
    Set config keys via list (e.g., from command line).
    Example: `args_list = ['NUM_DEVICES', 8]`.
    """
    assert len(args_list) % 2 == 0, \
        'Looks like you forgot to specify values or keys for some args'
    for key, value in zip(args_list[0::2], args_list[1::2]):
        key_list = key.split('.')
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, 'Config key {} not found'.format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        assert subkey in cfg, 'Config key {} not found'.format(subkey)
        val = _decode_cfg_value(value)
        val = _check_and_coerce_cfg_value_type(val, cfg[subkey], subkey, key)
        cfg[subkey] = val


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as: string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """
    Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b, type_a = type(value_b), type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
