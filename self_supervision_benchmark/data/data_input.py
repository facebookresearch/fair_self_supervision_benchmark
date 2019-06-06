# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This module uses shared memory buffer to load the data using multiprocessing.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import atexit
import collections
import ctypes
import cv2
import logging
import numpy as np
from multiprocessing import Pool
from multiprocessing.sharedctypes import RawArray

from self_supervision_benchmark.core.config import config as cfg
import self_supervision_benchmark.data.data_transformation as data_transformation
from self_supervision_benchmark.utils import helpers

logger = logging.getLogger(__name__)

# Mean and Std are BGR based
DATA_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
DATA_STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)
# PCA is RGB based
PCA = {
    'eigval': np.array([0.2175, 0.0188, 0.0045]).astype(np.float32),
    'eigvec': np.array([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203]
    ]).astype(np.float32)
}

global execution_context
execution_context = None


def create_data_input(
    input_db, expected_data_size, num_processes, num_workers, split, batch_size,
):
    # create a global execution context for the dataloader which contains the
    # pool for each thread and each pool has num_processes and a shared
    # data list
    global execution_context

    def init(worker_ids):
        # To point the global variable at a different object
        # (i.e. in every function where you want to update), it's required
        # to use the global keyword again. So when the init happens for train
        # and test object, the global variable will be pointed to a different
        # object for train/test.
        global execution_context
        logging.info('Creating the execution context for worker_ids: {}'.format(
            worker_ids
        ))
        execution_context = _create_execution_context(
            worker_ids, expected_data_size, num_processes, batch_size)
        atexit.register(_shutdown_pools)

    # in order to get the minibatch, we need some information from the db class
    def get_minibatch_out(input_db, worker_id, batch_size, db_indices):
        pools = execution_context.pools
        shared_data_lists = execution_context.shared_data_lists
        curr_pool = pools[worker_id]
        shared_data_list = shared_data_lists[worker_id]

        # for the given input_db, get the image paths we want to load and their
        # labels
        image_paths, labels, split_list = input_db.get_minibatch_path_indexes(
            db_indices
        )
        results = _load_and_process_images(
            worker_id, curr_pool, image_paths, split_list, shared_data_list,
            labels, db_indices, batch_size
        )
        images, labels, weights = results[0], results[1], results[2]
        return images, labels, weights

    return (init, get_minibatch_out)


def convert_to_batch(data):
    data = np.concatenate(
        [arr[np.newaxis] for arr in data]).astype(np.float32)
    return np.ascontiguousarray(data)


def _load_and_process_images(
    worker_id, curr_pool, image_paths, split_list, shared_data_list, labels,
    db_indices, batch_size
):
    map_results = curr_pool.map_async(
        get_image_from_source,
        zip(
            [i for i in range(0, len(image_paths))],
            image_paths,
            split_list,
            db_indices,
        )
    )
    results, lbls, weights = [], [], []
    for index, invalid in map_results.get():
        if index is not None:
            np_arr = shared_data_list[index]
            if not invalid:
                weights.append(1)
            else:
                weights.append(0)
            if split_list[0] == 0 and cfg.TEST.TEN_CROP:
                np_arr = np.reshape(
                    np_arr, (10, 3, cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE))
            else:
                np_arr = np.reshape(
                    np_arr, (3, cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE))
            lbls.append(labels[index])
            results.append(np_arr)

    if len(results) > 0 and len(lbls) > 0:
        lbls = np.array(lbls).astype(np.int32)
        weights = np.array(weights).astype(np.float32)
        if split_list[0] == 0 and cfg.TEST.TEN_CROP:
            data = np.ascontiguousarray(np.concatenate(results, axis=0))
        else:
            data = convert_to_batch(results)
    return data, lbls, weights


def get_image_from_source(args):
    (index, image_path, split, db_index) = args
    img, invalid = None, False
    try:
        img = cv2.imread(image_path)
        if not cfg.COLORIZATION.COLOR_ON:
            # in colorization, the image read from disk is uint8, we keep it as is
            img = np.array(img).astype(np.float32)
        elif cfg.COLORIZATION.COLOR_ON:
            # transform_image() step converts this image to float32 which messes
            # with the LAB conversion
            img = data_transformation.convert2lab(img, db_index)
        img = fix_bad_image(img)
    except Exception as e:
        logger.info('Exception in reading image from source: {}'.format(e))
        return None, True
    else:
        img = transform_image(img, split)
        if cfg.COLORIZATION.COLOR_ON:
            img = helpers.HWC2CHW(img)

        crop_size, num_crops = cfg.TRAIN.CROP_SIZE, 1
        if split == 0 and cfg.TEST.TEN_CROP:
            num_crops = 10
        img = np.ascontiguousarray(img.reshape((
            num_crops, 3, crop_size, crop_size))).astype(np.float32)

        if len(cfg.TRAIN.DATA_PROCESSING) > 0:
            img = preprocess_image(img, split)

        np_arr = shared_data_list[index]
        np_arr = np.reshape(np_arr, img.shape)
        np_arr[:] = img
        return index, invalid


def preprocess_image(image, split):
    # image is of shape (1 x 3 x H x W) or (10 x 3 x H x W) for 10-crop testing.
    if not (split == 0 and cfg.TEST.TEN_CROP):
        C, H, W = image.shape[1], image.shape[2], image.shape[3]
        image = image.reshape(C, H, W)
    if split == 1 and 'color_jitter' in cfg.TRAIN.DATA_PROCESSING:
        image = data_transformation.color_jitter(
            image, img_brightness=0.4, img_contrast=0.4, img_saturation=0.4
        )
    if split == 1 and 'lighting' in cfg.TRAIN.DATA_PROCESSING:
        image = data_transformation.lighting(
            image, alphastd=0.1, eigval=PCA['eigval'], eigvec=PCA['eigvec']
        )
    # split = 1 (train), 0 (test/val)
    ten_crop = False
    if split == 0 and cfg.TEST.TEN_CROP:
        ten_crop = True
    if 'color_normalization' in cfg.TRAIN.DATA_PROCESSING:
        if 'caffenet_bvlc' in cfg.MODEL.MODEL_NAME:
            logger.info("========>WARN: using mean/std normalization for caffenet")
        image = data_transformation.color_normalization(
            image, DATA_MEAN, DATA_STD, ten_crop=ten_crop,
        )
    if 'mean_normalization' in cfg.TRAIN.DATA_PROCESSING:
        if not ('caffenet' in cfg.MODEL.MODEL_NAME):
            image = data_transformation.mean_normalization(
                image, DATA_MEAN, ten_crop=ten_crop,
            )
        else:
            image = data_transformation.mean_normalization(
                image, 255.0 * DATA_MEAN, ten_crop=ten_crop,
            )
    return image


def fix_bad_image(img):
    if len(img.shape) == 2:
        img = np.repeat(img[np.newaxis, :, :], 3, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, 0:3]
    return img


def transform_train_image(img, preprocess=True):
    if 'scale' in cfg.TRAIN.DATA_TRANSFORMS:
        # scale the shorter side to 256 and the other using aspect ratio
        img = data_transformation.scale(cfg.TRAIN.SCALE, img)
    if 'global_resize' in cfg.TRAIN.DATA_TRANSFORMS:
        img = data_transformation.global_resize(
            cfg.TRAIN.GLOBAL_RESIZE_VALUE, img, order='HWC')
    if 'random_scale_jitter' in cfg.TRAIN.DATA_TRANSFORMS:
        img = data_transformation.random_scale_jitter(
            img, min_size=cfg.TRAIN.RANDOM_SCALE_JITTER[0],
            max_size=cfg.TRAIN.RANDOM_SCALE_JITTER[1])
    if 'center_crop' in cfg.TRAIN.DATA_TRANSFORMS:
        img = data_transformation.center_crop(cfg.TRAIN.CROP_SIZE, img)
    if 'random_crop' in cfg.TRAIN.DATA_TRANSFORMS:
        img = data_transformation.random_crop(
            img, cfg.TRAIN.CROP_SIZE, pad_size=0, order='HWC')
    if 'random_sized_crop' in cfg.TRAIN.DATA_TRANSFORMS:
        img = data_transformation.random_sized_crop(
            img, cfg.TRAIN.CROP_SIZE,
            min_scale=cfg.TRAIN.IMG_CROP_MIN_AREA,
            max_scale=cfg.TRAIN.IMG_CROP_MAX_AREA
        )
    if 'horizontal_flip' in cfg.TRAIN.DATA_TRANSFORMS:
        img = data_transformation.horizontal_flip(0.5, img, order='HWC')
    return img


def transform_image(img, split, preprocess=True):
    # TRAIN = split 1 and TEST = split 0
    if split == 1:
        img = transform_train_image(img, preprocess)
    else:
        img = data_transformation.scale(cfg.TEST.SCALE, img)
        if 'global_resize' in cfg.TRAIN.DATA_TRANSFORMS:
            img = data_transformation.global_resize(
                cfg.TRAIN.GLOBAL_RESIZE_VALUE, img, order='HWC')
        if cfg.TEST.TEN_CROP:
            # image will be 10 x H x W x C format for ten crop
            if cfg.TEST.TEN_CROP_TYPE == 'random':
                img = data_transformation.ten_random_crop(cfg.TEST.CROP_SIZE, img)
            else:
                img = data_transformation.ten_crop(cfg.TEST.CROP_SIZE, img)
        else:
            img = data_transformation.center_crop(cfg.TEST.CROP_SIZE, img)
    if not cfg.COLORIZATION.COLOR_ON:
        # Convert image to CHW keeping BGR order
        img = helpers.HWC2CHW(img)
        # image [0, 255] -> [0, 1] if the model is not caffenet
        if not ('caffenet' in cfg.MODEL.MODEL_NAME):
            img = img / 255.0
    return img


def _create_execution_context(
    worker_ids, expected_data_size, num_processes, batch_size
):
    print('execution context - batch_size: {} expected_data_size: {}'.format(
        batch_size, expected_data_size))
    logger.info('Creating Execution Context...')
    if execution_context is None:
        pools = {}
        shared_data_lists = {}
    else:
        pools = execution_context.pools
        shared_data_lists = execution_context.shared_data_lists
    logger.info('POOLS: {}'.format(pools))
    logger.info('DATA LISTS: {}'.format(len(shared_data_lists)))
    for worker_id in worker_ids:
        # for each worker_id, create a shared pool
        shared_data_list = []
        shared_data_lists[worker_id] = shared_data_list
        logger.info('worker_id: {} list: {}'.format(
            worker_id, len(shared_data_lists)))
        logger.info('worker_id: {} list keys: {}'.format(
            worker_id, shared_data_lists.keys()))
        # for each worker_id, we fetch a batch size of 32 and this is being
        # done by various parallel processes
        for _ in range(batch_size):
            shared_arr = RawArray(ctypes.c_float, expected_data_size)
            shared_data_list.append(shared_arr)
        # in multi-processing, various parallel instances of global variable will
        # be created if we want. Here we are doing init only
        pools[worker_id] = Pool(
            processes=num_processes, initializer=_init_pool,
            initargs=(shared_data_list,)
        )
    context = collections.namedtuple(
        'ExecutionContext', ['pools', 'shared_data_lists']
    )
    context.pools = pools
    context.shared_data_lists = shared_data_lists
    logger.info('CREATED POOL: {}'.format(pools))
    logger.info('CREATED SHARED LISTS: {}'.format(len(shared_data_lists)))
    logger.info('POOL keys: {}'.format(pools.keys()))
    logger.info('LIST keys: {}'.format(shared_data_lists.keys()))
    return context


def _init_pool(data_list):
    """
    Each pool process calls this initializer.
    Load the array to be populated into that process's global namespace.
    """
    global shared_data_list
    shared_data_list = data_list


def _shutdown_pools():
    pools = execution_context.pools
    logger.info("Shutting down multiprocessing pools..")
    for i, p in enumerate(pools.values()):
        logger.info("Shutting down pool {}".format(i))
        try:
            p.close()
            p.join()
        except Exception as e:
            logger.info(e)
            continue
    logger.info("Pools closed")
