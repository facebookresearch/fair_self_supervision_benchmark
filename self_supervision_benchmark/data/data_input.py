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
    global train_laser_loader
    global val_laser_loader

    def init(worker_ids):
        # To point the global variable at a different object
        # (i.e. in every function where you want to update), it's required
        # to use the global keyword again. So when the init happens for train
        # and test object, the global variable will be pointed to a different
        # object for train/test.
        global execution_context
        global train_laser_loader
        global val_laser_loader
        logging.info('Creating the execution context for worker_ids: {}'.format(
            worker_ids
        ))
        if split == 'train' and (cfg.DATA_SOURCE == 'laser'):
            train_laser_loader = input_db._laser_loader
        if split == 'val' and (cfg.DATA_SOURCE == 'laser'):
            val_laser_loader = input_db._laser_loader
        execution_context = _create_execution_context(
            worker_ids, expected_data_size, num_processes, batch_size)
        atexit.register(_shutdown_pools)

    # in order to get the minibatch, we need some information from the db class
    def get_minibatch_out(
        input_db, worker_id, batch_size, db_indices, jigsaw_perms
    ):
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
            labels, db_indices, batch_size, jigsaw_perms
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
    db_indices, batch_size, jigsaw_perms=None
):
    # this is for the case of simulating the batch drop experiment where we see
    # random tanking in the train loss curves during training
    # validity_vector = [1] * batch_size
    validity_vector = np.ones(batch_size, dtype=np.int32)
    # print('batch_size: {}'.format(batch_size))
    if cfg.DROP_BATCHES.DROP_ON:
        # print('dropping batches')
        max_drop_num = cfg.DROP_BATCHES.DROP_NUM
        random_drop = np.random.randint(0, max_drop_num)
        assert random_drop < batch_size, "Can not drop larger than batch size"
        for idx in range(random_drop):
            validity_vector[idx] = 0
    # print(np.sum(validity_vector))
    map_results = curr_pool.map_async(
        get_image_from_source,
        zip(
            [i for i in range(0, len(image_paths))],
            image_paths,
            split_list,
            db_indices,
            validity_vector
        )
    )
    results, lbls, weights = [], [], []
    # counter = 0
    # print('111 HERE HERE HERE')
    for index, invalid in map_results.get():
        if index is not None:
            # print('HERE HERE HERE HERE HERE')
            np_arr = shared_data_list[index]
            if not invalid:
                weights.append(1)
            else:
                weights.append(0)
            # print('is_jigsaw: {}'.format(cfg.JIGSAW.JIGSAW_ON))
            # print('patch: {}'.format(cfg.JIGSAW.PATCH_SIZE))
            if not cfg.JIGSAW.JIGSAW_ON:
                # print('CROP_SIZE: {}'.format(cfg.TRAIN.CROP_SIZE))
                # print(split_list)
                if split_list[0] == 0 and cfg.TEST.TEN_CROP:
                    np_arr = np.reshape(
                        np_arr, (cfg.TEST.TEN_CROP_N, 3, cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE))
                else:
                    np_arr = np.reshape(
                        np_arr, (3, cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE))
                lbls.append(labels[index])
                results.append(np_arr)
            elif cfg.JIGSAW.JIGSAW_ON:
                crop_size = cfg.JIGSAW.PATCH_SIZE
                patches = cfg.JIGSAW.NUM_PATCHES
                np_arr = np.reshape(np_arr, (patches, 3, crop_size, crop_size))
                if split_list[index] == 1 and (not cfg.JIGSAW.TRAIN_FIXED_PERM_ASSIGNMENT):
                    # print('loading expected random perm')
                    # print('random perm train')
                    n = np.random.randint(jigsaw_perms.shape[0])
                elif split_list[index] == 1 and cfg.DATASET in ['instagram', 'imagenet1k', 'imagenet22k', 'yfcc100m', 'yfcc100m_handles', 'places205', 'nyuv2', 'voc2007', 'coco']:
                    # we want to use the fixed permutation set for the evaluation
                    n = labels[index]
                elif split_list[index] == 0 and (not cfg.JIGSAW.TEST_FIXED_PERM_ASSIGNMENT):
                    # print('random perm val')
                    n = np.random.randint(jigsaw_perms.shape[0])
                elif split_list[index] == 0 and cfg.DATASET in ['instagram', 'imagenet1k', 'imagenet22k', 'yfcc100m', 'yfcc100m_handles', 'places205', 'nyuv2', 'voc2007', 'coco']:
                    n = labels[index]
                    # print('using fixed permutation assignment: {}'.format(n))
                lbls.append(int(n))
                if cfg.JIGSAW.NO_SHUFFLE:
                    # print('no shuffling')
                    data = [np_arr[t] for t in range(patches)]
                else:
                    data = [np_arr[jigsaw_perms[n][t]] for t in range(patches)]
                if 'jigsaw_siamese' in cfg.MODEL.MODEL_NAME:
                    # (9 * 3) x 64 x 64
                    data = np.ascontiguousarray(np.concatenate(data, axis=0))
                else:
                    # 9 x 3 x 64 x 64
                    data = convert_to_batch(data)
                # for i in range(data.shape[0]):
                #     visualize.visualize_image(
                #         data[i], '/home/prigoyal/local/jigsaw_loader_in1k_final',
                #         'split1_idx{}_patch{}.png'.format(db_indices[counter], i), order='CHW'
                #     )
                # counter += 1
                results.append(data)

    if len(results) > 0 and len(lbls) > 0:
        lbls = np.array(lbls).astype(np.int32)
        weights = np.array(weights).astype(np.float32)
        if cfg.JIGSAW.JIGSAW_ON and cfg.MODEL.MODEL_NAME == 'alexnet_jigsaw':
            # Length of labels is 32. Finally, we want the data to be arranged as:
            # [img0_patch0, img1_patch0,.....,img31_patch0, img0_patch1,......]
            # Currently, data is arranged as below:
            # [img0_patch0, img0_patch1, img0_patch2,....., img1_patch0,.....]
            n = np.random.randint(0, 1000)
            num_patches = cfg.JIGSAW.NUM_PATCHES
            # (9*32) x 3 x 64 x 64
            # in case of siamese network, we have data of shape: 32 x (9*3) x 64 x 64
            # this data is already
            results = np.ascontiguousarray(np.concatenate(results, axis=0))
            # [0, 9, 18, 27,......, 279] --> length 32
            indices = np.arange(batch_size) * num_patches
            data = []
            for idx in range(num_patches):
                minibatch_patch = results[indices + idx]
                data.append(minibatch_patch)
            data = np.concatenate(data, axis=0)
            # np.save(
            #     '/home/prigoyal/local/jigsaw_loader_in1k_final/batch{}.npy'.format(n),
            #     data)
            # np.save(
            #     '/home/prigoyal/local/jigsaw_loader_in1k_final/labels{}.npy'.format(n),
            #     lbls)
        else:
            if split_list[0] == 0 and cfg.TEST.TEN_CROP:
                data = np.ascontiguousarray(np.concatenate(results, axis=0))
            else:
                data = convert_to_batch(results)
            # n = np.random.randint(0, 1000)
            # np.save('/home/prigoyal/local/color/batch{}.npy'.format(n), data)
    # print('data: {} labels: {} weights_min: {} max: {}'.format(
    #     data.shape, lbls.shape, np.min(weights), np.max(weights))
    # )
    # print('data: {} labels: {}'.format(data.shape, lbls.shape))
    return data, lbls, weights

def create_patches(image):
    # input image: 3 X H x W    &    output: 9 x 3 x 64 x 64
    output = []
    grid_size = math.sqrt(cfg.JIGSAW.NUM_PATCHES)           # 3
    size = int(image.shape[1] / grid_size)                  # 255 / 3 = 85
    # print('create_patches input: {}'.format(image.shape))
    for i in range(int(grid_size)):
        for j in range(int(grid_size)):
            patch_num = (3 * i) + j
            x_offset = i * size
            y_offset = j * size
            patch = image[:, y_offset:y_offset + size, x_offset:x_offset + size]
            # now we want to do random crop 64 x 64 from these patches
            if cfg.JIGSAW.CHANNEL_JITTER > 0:
                cropped = data_transformation.random_crop_with_channel_jitter(
                    patch, cfg.JIGSAW.PATCH_SIZE,
                    channel_jitter=cfg.JIGSAW.CHANNEL_JITTER, pad_size=0)
            else:
                # patch = helpers.CHW2HWC(patch)
                # cropped = data_transformation.center_crop(cfg.JIGSAW.PATCH_SIZE, patch)
                # cropped = helpers.HWC2CHW(cropped)
                cropped = data_transformation.random_crop(
                    patch, cfg.JIGSAW.PATCH_SIZE, pad_size=0)
            output.append(cropped)
    patches = convert_to_batch(output)
    if len(cfg.JIGSAW.PATCH_PROCESSING) > 0:
        if 'gray_center' in cfg.JIGSAW.PATCH_PROCESSING:
            # print('processing gray_center: {}'.format(patches.shape))
            patches = data_transformation.batch_gray_center(
                patches, cfg.JIGSAW.GRAY_CENTER_SIZE
            )
        if 'gray_border' in cfg.JIGSAW.PATCH_PROCESSING:
            patches = data_transformation.batch_gray_border(
                patches, cfg.JIGSAW.GRAY_BORDER_SIZE
            )
        if 'gray_patches' in cfg.JIGSAW.PATCH_PROCESSING:
            patches = data_transformation.batch_gray_patches(
                patches, cfg.JIGSAW.GRAY_PATCH_NUM
            )
        if 'interpolate_patches' in cfg.JIGSAW.PATCH_PROCESSING:
            patches = data_transformation.batch_patch_interpolate(
                patches, cfg.JIGSAW.GRAY_BORDER_SIZE
            )
    # print('create_patches output: {}'.format(patches.shape))
    return patches

# def _load_and_process_images(
#     worker_id, curr_pool, image_paths, split_list, shared_data_list, labels,
#     db_indices, batch_size
# ):
#     map_results = curr_pool.map_async(
#         get_image_from_source,
#         zip(
#             [i for i in range(0, len(image_paths))],
#             image_paths,
#             split_list,
#             db_indices,
#         )
#     )
#     results, lbls, weights = [], [], []
#     for index, invalid in map_results.get():
#         if index is not None:
#             np_arr = shared_data_list[index]
#             if not invalid:
#                 weights.append(1)
#             else:
#                 weights.append(0)
#             if split_list[0] == 0 and cfg.TEST.TEN_CROP:
#                 np_arr = np.reshape(
#                     np_arr, (cfg.TEST.TEN_CROP_N, 3, cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE))
#             else:
#                 np_arr = np.reshape(
#                     np_arr, (3, cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE))
#             lbls.append(labels[index])
#             results.append(np_arr)
#
#     if len(results) > 0 and len(lbls) > 0:
#         lbls = np.array(lbls).astype(np.int32)
#         weights = np.array(weights).astype(np.float32)
#         if split_list[0] == 0 and cfg.TEST.TEN_CROP:
#             data = np.ascontiguousarray(np.concatenate(results, axis=0))
#         else:
#             data = convert_to_batch(results)
#     return data, lbls, weights


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
        crop_size = cfg.TRAIN.CROP_SIZE
        if cfg.JIGSAW.JIGSAW_ON and cfg.JIGSAW.NUM_PATCHES > 1:
            # if we want to create patches from the image, we do that now.
            # currently, the patches are created as per jigsaw code.
            # 1 x 3 x H x W -> 9 x 3 x 64 x 64
            img = create_patches(img)
            crop_size = cfg.JIGSAW.PATCH_SIZE
            img = np.ascontiguousarray(img.reshape((
                cfg.JIGSAW.NUM_PATCHES, 3, crop_size, crop_size))).astype(np.float32)
        elif cfg.COLORIZATION.COLOR_ON:
            crop_size = cfg.TRAIN.CROP_SIZE
            img = helpers.HWC2CHW(img)
            num_crops = 1
            if split == 0 and cfg.TEST.TEN_CROP:
                num_crops = cfg.TEST.TEN_CROP_N
            img = np.ascontiguousarray(img.reshape((
                num_crops, 3, crop_size, crop_size))).astype(np.float32)
        else:
            num_crops = 1
            if split == 0 and cfg.TEST.TEN_CROP:
                num_crops = cfg.TEST.TEN_CROP_N
            img = np.ascontiguousarray(img.reshape((
                num_crops, 3, cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE))).astype(np.float32)
        if len(cfg.TRAIN.DATA_PROCESSING) > 0:
            img = preprocess_image(img, split)
        # in case of Jigsaw, we actually have 9 patches
        if not cfg.JIGSAW.JIGSAW_ON and not (split == 0 and cfg.TEST.TEN_CROP):
            img = img[0]
        np_arr = shared_data_list[index]
        np_arr = np.reshape(np_arr, img.shape)
        np_arr[:] = img
        return index, invalid


def preprocess_image(image, split):
    # print('entry preprocess_image: {}'.format(image.shape))
    # print(cfg.TRAIN.DATA_PROCESSING)
    if split == 1 and 'batch_color_jitter' in cfg.TRAIN.DATA_PROCESSING:
        image = data_transformation.batch_color_jitter(
            image, img_brightness=0.4, img_contrast=0.4, img_saturation=0.4
        )
    if split == 1 and 'batch_lighting' in cfg.TRAIN.DATA_PROCESSING:
        image = data_transformation.batch_lighting(
            image, alphastd=0.1, eigval=PCA['eigval'], eigvec=PCA['eigvec']
        )
    if 'batch_color_normalization' in cfg.TRAIN.DATA_PROCESSING:
        image = data_transformation.batch_color_normalization(
            image, DATA_MEAN, DATA_STD
        )
    if 'mean_normalization' in cfg.TRAIN.DATA_PROCESSING:
        # if not (cfg.MODEL.MODEL_NAME in ['caffenet', 'caffenet_bvlc']):
        if not ('caffenet' in cfg.MODEL.MODEL_NAME):
            image = data_transformation.batch_mean_normalization(image, DATA_MEAN)
        else:
            # logger.info('caffenet mean: {}'.format(255.0*DATA_MEAN))
            image = data_transformation.batch_mean_normalization(image, 255.0 * DATA_MEAN)
    if 'batch_grayscale' in cfg.TRAIN.DATA_PROCESSING:
        if 'all_gray' in cfg.TRAIN.DATA_PROCESSING:
            make_gray = np.random.randint(0,10) >= 0
        elif 'all_color' in cfg.TRAIN.DATA_PROCESSING:
            make_gray = False
        else:
            make_gray = np.random.randint(0, 10) >= 5

        if 'all_color_proj' in cfg.TRAIN.DATA_PROCESSING:
            color_proj = True
        else:
            color_proj = np.random.randint(0, 2) == 0
        if make_gray:
            image = data_transformation.batch_grayscale(image)
        elif color_proj > 0 and 'color_proj' in cfg.TRAIN.DATA_PROCESSING:
            image = data_transformation.batch_color_proj(image)
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
                img = data_transformation.ten_random_crop(cfg.TEST.CROP_SIZE, img, cfg.TEST.TEN_CROP_N)
            else:
                if cfg.TEST.TEN_CROP_N == 9:
                    img = data_transformation.nine_crop(cfg.TEST.CROP_SIZE, img)
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
