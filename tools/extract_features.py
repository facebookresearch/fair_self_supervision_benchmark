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
import cv2
from tqdm import tqdm #("python-tqdm", "any"),
import pickle
import faiss # "//deeplearning/projects/faiss:pyfaiss_gpu"
import matplotlib.pyplot as plt
import scipy
from collections import defaultdict
import random

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.core.config import (
    cfg_from_file, cfg_from_list, print_cfg
)
from self_supervision_benchmark.modeling import model_builder
from self_supervision_benchmark.utils import helpers, checkpoints
from self_supervision_benchmark.modeling.clustering import Kmeans

from caffe2.python import workspace

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def extractonce(opts):
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)
    assert 'npy' not in cfg.DATASET

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

def extract(opts):
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)
    assert 'npy' not in cfg.DATASET
    # create the model in test mode only
    assert opts.data_type in ['train', 'val', 'test'], "Please specify valid type."
    print(cfg.TEST.LABELS_FILE)
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

    def init():
        global img_features, img_imgs, img_targets
        # initialize the dictionary to store features and targets
        img_features, img_targets, img_imgs = {}, {}, {}
        for bl in cfg.MODEL.EXTRACT_BLOBS:
            img_features[bl], img_targets[bl], img_imgs[bl] = {}, {}, {}
    init()

    def save(idx):
        global img_features, img_imgs, img_targets
        for bl in cfg.MODEL.EXTRACT_BLOBS:
            img_features[bl] = dict(sorted(img_features[bl].items()))
            img_imgs[bl] = dict(sorted(img_imgs[bl].items()))
            img_targets[bl] = dict(sorted(img_targets[bl].items()))
            feats = np.array(list(img_features[bl].values()))
            N = feats.shape[0]
            logger.info('got image features: {} {}'.format(bl, feats.shape))
            output = {
                'img_features': feats.reshape(N, -1),
                'img_imgs': np.array(list(img_imgs[bl].values()), dtype=np.uint8),
                'img_inds': np.array(list(img_features[bl].keys())),
                'img_targets': np.array(list(img_targets[bl].values())),
            }
            prefix = '{}_{}_part{}_'.format(opts.output_file_prefix, bl, idx)

            out_feat_file = os.path.join(opts.output_dir, prefix + 'features.npy')
            logger.info('Saving extracted features: {} {} to: {}'.format(
                bl, output['img_features'].shape, out_feat_file))
            np.save(out_feat_file, output['img_features'])

            out_feat_file = os.path.join(opts.output_dir, prefix + 'imgs.npy')
            logger.info('Saving extracted features: {} {} to: {}'.format(
                bl, output['img_imgs'].shape, out_feat_file))
            np.save(out_feat_file, output['img_imgs'])

            out_target_file = os.path.join(opts.output_dir, prefix + 'targets.npy')
            logger.info('Saving extracted targets: {} to: {}'.format(
                output['img_targets'].shape, out_target_file))
            np.save(out_target_file, output['img_targets'])

            out_inds_file = os.path.join(opts.output_dir, prefix + 'inds.npy')
            logger.info('Saving extracted indices: {} to: {}'.format(
                output['img_inds'].shape, out_inds_file))
            np.save(out_inds_file, output['img_inds'])

    # we keep track of data indices seen so far. This ensures we extract feature
    # for each image only once.
    indices_list = []
    total_test_iters = helpers.get_num_test_iter(model.input_db)
    logger.info('Test epoch iters: {}'.format(total_test_iters))
    # when we extract features, we run 4 epochs to make sure we capture all the
    # data points. This is needed because we use the multi-processing dataloader
    # which shuffles the data. In very low-shot setting, making multiple passes
    # over the entire data becomes crucial.
    # extraction_iters = int(total_test_iters * 4)
    extraction_iters = int(total_test_iters)
    save_idx = 0
    for test_iter in tqdm(range(0, extraction_iters)):
        workspace.RunNet(model.net.Proto().name)
        if test_iter == 0:
            helpers.print_net(model)
        if test_iter % 100 == 0:
            logger.info('at: [{}/{}]'.format(test_iter, extraction_iters))
        for device in range(cfg.NUM_DEVICES):
            # 20 * ncrop
            imgs = workspace.FetchBlob('gpu_{}/data'.format(device))
            imgs = deprocess_imgs(imgs).astype(np.uint8)
            # 20
            indices = workspace.FetchBlob('gpu_{}/db_indices'.format(device))
            indices_list.extend(list(indices))
            num_images = indices.shape[0]
            # 20
            labels = workspace.FetchBlob('gpu_{}/labels'.format(device))
            for bl in cfg.MODEL.EXTRACT_BLOBS:
                if cfg.EXTRACT.SPATIAL:
                    features = workspace.FetchBlob('gpu_{}/{}'.format(device, bl))
                else:
                    features = workspace.FetchBlob('gpu_{}/{}_avg'.format(device, bl))
                for idx in range(num_images):
                    index = indices[idx]
                    if not (index in img_targets[bl]):
                        img_targets[bl][index] = labels[idx].reshape(-1)
                        for i in range(cfg.TEST.TEN_CROP_N):
                            img_features[bl][index * cfg.TEST.TEN_CROP_N + i] = features[idx * cfg.TEST.TEN_CROP_N + i]
                            img_imgs[bl][index * cfg.TEST.TEN_CROP_N + i] = imgs[idx * cfg.TEST.TEN_CROP_N + i]
        if (test_iter + 1) % 500 == 0 or (test_iter + 1) == extraction_iters:
            save(save_idx)
            save_idx += 1
            init()

    logger.info('All Done!')
    # shut down the data loader
    model.data_loader.shutdown_dataloader()

def npyextract(opts):
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)
    assert 'npy' in cfg.DATASET

    # create the model in test mode only
    assert opts.data_type in ['train', 'val', 'test'], "Please specify valid type."
    print(cfg.TEST.LABELS_FILE)
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

    def init():
        global img_features, img_targets
        # initialize the dictionary to store features and targets
        img_features, img_targets = {}, {}
        for bl in cfg.MODEL.EXTRACT_BLOBS:
            img_features[bl], img_targets[bl] = {}, {}
    init()

    def save(idx):
        global img_features, img_targets
        for bl in cfg.MODEL.EXTRACT_BLOBS:
            img_features[bl] = dict(sorted(img_features[bl].items()))
            img_targets[bl] = dict(sorted(img_targets[bl].items()))
            feats = np.array(list(img_features[bl].values()))
            N = feats.shape[0]
            logger.info('got image features: {} {}'.format(bl, feats.shape))
            output = {
                'img_features': feats.reshape(N, -1),
                'img_inds': np.array(list(img_features[bl].keys())),
            }
            prefix = '{}_{}_part{}_'.format(opts.output_file_prefix, bl, idx)

            out_feat_file = os.path.join(opts.output_dir, prefix + 'features.npy')
            logger.info('Saving extracted features: {} {} to: {}'.format(
                bl, output['img_features'].shape, out_feat_file))
            np.save(out_feat_file, output['img_features'])

            out_inds_file = os.path.join(opts.output_dir, prefix + 'inds.npy')
            logger.info('Saving extracted indices: {} to: {}'.format(
                output['img_inds'].shape, out_inds_file))
            np.save(out_inds_file, output['img_inds'])

    # we keep track of data indices seen so far. This ensures we extract feature
    # for each image only once.
    indices_list = []
    total_test_iters = helpers.get_num_test_iter(model.input_db)
    logger.info('Test epoch iters: {}'.format(total_test_iters))
    # when we extract features, we run 4 epochs to make sure we capture all the
    # data points. This is needed because we use the multi-processing dataloader
    # which shuffles the data. In very low-shot setting, making multiple passes
    # over the entire data becomes crucial.
    # extraction_iters = int(total_test_iters * 4)
    extraction_iters = int(total_test_iters)
    save_idx = 0
    for test_iter in tqdm(range(0, extraction_iters)):
        workspace.RunNet(model.net.Proto().name)
        if test_iter == 0:
            helpers.print_net(model)
        if test_iter % 100 == 0:
            logger.info('at: [{}/{}]'.format(test_iter, extraction_iters))
        for device in range(cfg.NUM_DEVICES):
            indices = workspace.FetchBlob('gpu_{}/db_indices'.format(device))
            indices_list.extend(list(indices))
            num_images = indices.shape[0]
            # 20
            labels = workspace.FetchBlob('gpu_{}/labels'.format(device))
            for bl in cfg.MODEL.EXTRACT_BLOBS:
                if cfg.EXTRACT.SPATIAL:
                    features = workspace.FetchBlob('gpu_{}/{}'.format(device, bl))
                else:
                    features = workspace.FetchBlob('gpu_{}/{}_avg'.format(device, bl))
                for idx in range(num_images):
                    index = indices[idx]
                    if not (index in img_targets[bl]):
                        img_targets[bl][index] = labels[idx].reshape(-1)
                        for i in range(cfg.TEST.TEN_CROP_N):
                            img_features[bl][index * cfg.TEST.TEN_CROP_N + i] = features[idx * cfg.TEST.TEN_CROP_N + i]
        if (test_iter + 1) % 500 == 0 or (test_iter + 1) == extraction_iters:
            save(save_idx)
            save_idx += 1
            init()

    logger.info('All Done!')
    # shut down the data loader
    model.data_loader.shutdown_dataloader()

def pca(opts):
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)
    assert 'npy' not in cfg.DATASET
    assert cfg.EXTRACT.SPATIAL
    # create the model in test mode only
    assert opts.data_type in ['train', 'val', 'test'], "Please specify valid type."
    print(cfg.TEST.LABELS_FILE)
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

    def init():
        global img_features, img_imgs, img_targets
        # initialize the dictionary to store features and targets
        img_features, img_targets, img_imgs = {}, {}, {}
        for bl in cfg.MODEL.EXTRACT_BLOBS:
            img_features[bl], img_targets[bl], img_imgs[bl] = {}, {}, {}
    init()

    def save(idx):
        global img_features, img_imgs, img_targets
        for bl in cfg.MODEL.EXTRACT_BLOBS:
            img_features[bl] = dict(sorted(img_features[bl].items()))
            img_imgs[bl] = dict(sorted(img_imgs[bl].items()))
            img_targets[bl] = dict(sorted(img_targets[bl].items()))
            feats = np.array(list(img_features[bl].values()))
            N = feats.shape[0]
            logger.info('got image features: {} {}'.format(bl, feats.shape))
            output = {
                'img_features': feats.reshape(N, -1),
                'img_imgs': np.array(list(img_imgs[bl].values()), dtype=np.uint8),
                'img_inds': np.array(list(img_features[bl].keys())),
                'img_targets': np.array(list(img_targets[bl].values())),
            }
            prefix = '{}_{}_part{}_'.format(opts.output_file_prefix, bl, idx)

            out_feat_file = os.path.join(opts.output_dir, prefix + 'features.npy')
            logger.info('Saving extracted features: {} {} to: {}'.format(
                bl, output['img_features'].shape, out_feat_file))
            np.save(out_feat_file, output['img_features'])

            out_feat_file = os.path.join(opts.output_dir, prefix + 'imgs.npy')
            logger.info('Saving extracted features: {} {} to: {}'.format(
                bl, output['img_imgs'].shape, out_feat_file))
            np.save(out_feat_file, output['img_imgs'])

            out_target_file = os.path.join(opts.output_dir, prefix + 'targets.npy')
            logger.info('Saving extracted targets: {} to: {}'.format(
                output['img_targets'].shape, out_target_file))
            np.save(out_target_file, output['img_targets'])

            out_inds_file = os.path.join(opts.output_dir, prefix + 'inds.npy')
            logger.info('Saving extracted indices: {} to: {}'.format(
                output['img_inds'].shape, out_inds_file))
            np.save(out_inds_file, output['img_inds'])

    # we keep track of data indices seen so far. This ensures we extract feature
    # for each image only once.
    indices_list = []
    total_test_iters = helpers.get_num_test_iter(model.input_db)
    logger.info('Test epoch iters: {}'.format(total_test_iters))
    # when we extract features, we run 4 epochs to make sure we capture all the
    # data points. This is needed because we use the multi-processing dataloader
    # which shuffles the data. In very low-shot setting, making multiple passes
    # over the entire data becomes crucial.
    # extraction_iters = int(total_test_iters * 4)
    extraction_iters = int(total_test_iters)
    save_idx = 0
    for test_iter in tqdm(range(0, extraction_iters)):
        workspace.RunNet(model.net.Proto().name)
        if test_iter == 0:
            helpers.print_net(model)
        if test_iter % 100 == 0:
            logger.info('at: [{}/{}]'.format(test_iter, extraction_iters))
        for device in range(cfg.NUM_DEVICES):
            # 20 * ncrop
            imgs = workspace.FetchBlob('gpu_{}/data'.format(device))
            imgs = deprocess_imgs(imgs).astype(np.uint8)
            # 20
            indices = workspace.FetchBlob('gpu_{}/db_indices'.format(device))
            indices_list.extend(list(indices))
            num_images = indices.shape[0]
            # 20
            labels = workspace.FetchBlob('gpu_{}/labels'.format(device))
            for bl in cfg.MODEL.EXTRACT_BLOBS:
                features = workspace.FetchBlob('gpu_{}/{}'.format(device, bl))
                for idx in range(num_images):
                    index = indices[idx]
                    if not (index in img_targets[bl]):
                        img_targets[bl][index] = labels[idx].reshape(-1)
                        for i in range(cfg.TEST.TEN_CROP_N):
                            img_features[bl][index * cfg.TEST.TEN_CROP_N + i] = features[idx * cfg.TEST.TEN_CROP_N + i]
                            img_imgs[bl][index * cfg.TEST.TEN_CROP_N + i] = imgs[idx * cfg.TEST.TEN_CROP_N + i]
        if (test_iter + 1) % 500 == 0 or (test_iter + 1) == extraction_iters:
            save(save_idx)
            save_idx += 1
            init()

    logger.info('All Done!')
    # shut down the data loader
    model.data_loader.shutdown_dataloader()

def parts_read(opts, name='features', maxparts=10000000, sel=None, dir=None):
    feats = []
    start = 0
    end = 0
    newdict = defaultdict(list)
    for idx in range(maxparts):
        prefix = '{}_{}_part{}_{}.npy'.format(opts.output_file_prefix, cfg.MODEL.EXTRACT_BLOBS[0], idx, name)
        if dir is None:
            dir = opts.output_dir
        prefix = os.path.join(dir, prefix)
        logger.info(prefix)
        if os.path.exists(prefix):
            feat = np.load(prefix)
            end += len(feat)
            if sel is None:
                feats.append(feat)
            else:
                for k, v in sel.items():
                    for i in v:
                        if i < end and i >= start:
                            newdict[k].append(feat[i - start].copy())
                del feat
            start = end
        else:
            break
    if sel is None:
        return np.vstack(feats)
    return newdict


def cluster(opts):
    feats = parts_read(opts, 'features')
    model = Kmeans(cfg.EXTRACT.KMEANS_K, 256, cfg.EXTRACT.PREPROCESS)
    # model = faiss.Kmeans(feats.shape[1], k=1024)
    model.train(feats.astype('float32'))
    out_inds_file = os.path.join(opts.output_dir, 'centroids'+str(cfg.EXTRACT.KMEANS_K))
    if cfg.EXTRACT.PREPROCESS:
        out_inds_file += 'pre'
    logger.info('Saving centroids: {} to: {}'.format(
        model.centroids.shape, out_inds_file))
    np.savez(out_inds_file, model.centroids, model.pca_trans[0], model.pca_trans[1])

    name = os.path.join(opts.output_dir, "assign" + str(cfg.EXTRACT.KMEANS_K))
    if cfg.EXTRACT.PREPROCESS:
        name += 'pre'
    with open(name, 'wb') as f:
        pickle.dump(model.images_lists, f)

def deprocess_imgs(imgs):
    DATA_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
    DATA_STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    imgs = imgs * DATA_STD[:, None, None] + DATA_MEAN[:, None, None]
    return imgs.transpose((0, 2, 3, 1)) * 255


def vispatches(opts):
    os.makedirs(os.path.join(opts.output_dir, 'rndvis'), exist_ok=True)
    imgs = parts_read(opts, 'imgs', maxparts=1)
    # idxs = np.array(range(len(imgs)))
    # np.random.shuffle(idxs)
    Nsamples = 100
    for i in tqdm(range(0, len(imgs), Nsamples)):
        # show_images(imgs[idxs[i:i + Nsamples]], int(Nsamples**.5), name=os.path.join(opts.output_dir, 'rndvis', str(i)))
        show_images(imgs[i:i + Nsamples], int(Nsamples**.5), name=os.path.join(opts.output_dir, 'rndvis', str(i)))


def assignsanitycheck(opts):
    feats = parts_read(opts, 'features')
    name = os.path.join(opts.output_dir, "centroids" + str(cfg.EXTRACT.KMEANS_K))
    if cfg.EXTRACT.PREPROCESS:
        name += 'pre'
    npzfile = np.load(name + ".npz")
    centroids, A, b = npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2']
    feats = np.dot(feats, A.T) + b
    row_sums = np.linalg.norm(feats , axis=1)
    feats = feats / row_sums[:, np.newaxis]
    name = os.path.join(opts.output_dir, "assign" + str(cfg.EXTRACT.KMEANS_K))
    Nsamples = cfg.EXTRACT.VIS_K
    if cfg.EXTRACT.PREPROCESS:
        name += 'pre'
    with open(name, 'rb') as f:
        d = pickle.load(f)
    dist = scipy.spatial.distance.cdist(centroids[0, None], feats)
    a = dist.squeeze().argsort()
    print(a[:1151] == np.array(d[0]))
    from IPython import embed ; embed()
    exit()

def assign(opts):
    # read original features as reference
    feats = parts_read(opts, 'features')
    # feats = parts_read(opts, 'features', dir='/home/yihuihe/outs/')
    name = os.path.join(opts.reference_dir if opts.reference_dir is not None else opts.output_dir,
        "centroids" + str(cfg.EXTRACT.KMEANS_K))
    if cfg.EXTRACT.PREPROCESS:
        name += 'pre'
    npzfile = np.load(name + ".npz")
    centroids, A, b = npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2']
    if cfg.EXTRACT.PREPROCESS:
        feats = np.dot(feats, A.T) + b
        row_sums = np.linalg.norm(feats , axis=1)
        feats = feats / row_sums[:, np.newaxis]

    # assign feats to centroids
    # N x K
    import faiss                   # make faiss available
    #res = faiss.StandardGpuResources()
    #flat_config = faiss.GpuIndexFlatConfig()
    #flat_config.useFloat16 = False
    #flat_config.device = 0
    #index = faiss.GpuIndexFlatL2(res, centroids.shape[1], flat_config)   # build the index
    index = faiss.IndexFlatL2(centroids.shape[1])   # build the index
    index.add(feats)                  # add vectors to the index
    print(index.ntotal)
    D, I = index.search(centroids, cfg.EXTRACT.VIS_K) # sanity check
    name = os.path.join(opts.output_dir, "assign" + str(cfg.EXTRACT.KMEANS_K))
    if cfg.EXTRACT.PREPROCESS:
        name += 'pre'
    d = {}
    for i, a in enumerate(tqdm(I)):
        d[i] = a
    with open(name, 'wb') as f:
        pickle.dump(d, f)


#     # N
#     assignment = dist.argmin(1)
#     d = defaultdict(list)
#     Nsamples = 100
#     for i, a in enumerate(tqdm(assignment)):
#         d[a].append(i)
#     # for i, feat in enumerate(tqdm(feats)):
#     #     if cfg.EXTRACT.PREPROCESS:
#     #         feat = np.dot(feat[None, ...], A.T) + b
#     #     else:
#     #         feat = feat[None, ...]
#     #     dist = scipy.spatial.distance.cdist(feat, centroids).squeeze()
#     #     cls = dist.argmin()
#     #     if len(d[cls]) > Nsamples:
#     #         continue
#     #     d[cls].append(i)
#     name = os.path.join(opts.output_dir, "assign" + str(cfg.EXTRACT.KMEANS_K))
#     if cfg.EXTRACT.PREPROCESS:
#         name += 'pre'
#     with open(name, 'wb') as f:
#         pickle.dump(d, f)

def viscluster(opts):
    name = os.path.join(opts.output_dir, "assign" + str(cfg.EXTRACT.KMEANS_K))
    Nsamples = cfg.EXTRACT.VIS_K
    if cfg.EXTRACT.PREPROCESS:
        name += 'pre'
    with open(name, 'rb') as f:
        d = pickle.load(f)
    for k, v in d.items():
        d[k] = v[:Nsamples]
    imgs = parts_read(opts, 'imgs', sel=d, dir=opts.reference_dir)
    # for k, v in d.items():
    #     show_images(imgs[v][:Nsamples], int(Nsamples**.5), name=os.path.join(opts.output_dir, 'vis', str(k)))
    path = os.path.join(opts.output_dir, 'vis'+str(cfg.EXTRACT.KMEANS_K))
    if cfg.EXTRACT.PREPROCESS:
        path += 'pre'
    os.makedirs(path, exist_ok=True)
    for k, v in tqdm(imgs.items()):
    # keys = random.sample(list(imgs), 100)
    # for k in tqdm(keys):
        # v = imgs[k]
        show_images(v, name=os.path.join(path, str(k)))


def comparecluster(opts):
    reference_dir = opts.reference_dir
    new_dir = opts.output_dir
    def getcentroids(dir):
        name = os.path.join(dir, 'centroids'+str(cfg.EXTRACT.KMEANS_K))
        if cfg.EXTRACT.PREPROCESS:
            name += 'pre'
        npzfile = np.load(name + ".npz")
        centroids, A, b = npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2']
        return centroids
    logger.info('loading')
    reference_centroids = getcentroids(reference_dir)
    new_centroids = getcentroids(new_dir)

    logger.info('compute distance')
    dist = scipy.spatial.distance.cdist(reference_centroids, new_centroids)

    comparedir = os.path.join(new_dir, 'compare'+str(cfg.EXTRACT.KMEANS_K))
    try:
        os.mkdir(comparedir)
    except:
        pass
    for i, j in enumerate(tqdm(dist.argmin(1)[:300])):
        print(i, j)
        name = os.path.join(reference_dir, 'vis'+str(cfg.EXTRACT.KMEANS_K))
        if cfg.EXTRACT.PREPROCESS:
            name += 'pre'
        reference_cluster = plt.imread(os.path.join(name, str(i) + '.png'))
        name = os.path.join(new_dir, 'vis'+str(cfg.EXTRACT.KMEANS_K))
        if cfg.EXTRACT.PREPROCESS:
            name += 'pre'
        new_cluster = plt.imread(os.path.join(name, str(j) + '.png'))
        show_images([reference_cluster, new_cluster], cols=1, BGR=False,
            titles=['old cluster', 'closest new cluster'],
            name=os.path.join(comparedir, str(i)), dpi=600)

# def show_images(images, cols=None, titles=None, name='tmp'):
#     """https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
#     Display a list of images in a single figure with matplotlib.
#
#     Parameters
#     ---------
#     images: List of np.arrays compatible with plt.imshow.
#
#     cols (Default = 1): Number of columns in figure (number of rows is
#                         set to np.ceil(n_images/float(cols))).
#
#     titles: List of titles corresponding to each image. Must have
#             the same length as titles.
#     """
#     assert((titles is None)or (len(images) == len(titles)))
#     n_images = len(images)
#     if cols is None:
#         cols = int(n_images**.5)
#     if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
#     fig = plt.figure()
#     for n, (image, title) in enumerate(zip(images, titles)):
#         a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
#         a.set_axis_off()
#         if image.ndim == 2:
#             plt.gray()
#         if cfg.EXTRACT.BGR:
#             image = image[..., ::-1]
#         plt.imshow(image, aspect="auto")
#         #a.set_title(title)
#     #fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
#     plt.subplots_adjust(wspace=.05, hspace=.05)
#     plt.savefig(name + '.png')
#     plt.close('all')

def show_images(images, cols=None, titles=None, name='tmp', BGR=True, pos=None, size=None, dpi=300):
    """https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    import matplotlib.pyplot as plt
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if cols is None:
        cols = int(n_images**.5)

    if pos is None:
        fig = plt.figure(figsize=(np.ceil(n_images/float(cols)), cols))
    else:
        assert size is not None
        fig = plt.figure(figsize=size)

    for n, image in enumerate(images):
        if pos is None:
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        else:
            a = fig.add_subplot(size[1], size[0], pos[n])

        a.set_axis_off()
        if image.ndim == 2:
            plt.gray()
        if BGR:
            image = image[..., ::-1]
        plt.imshow(image, aspect="auto")
        if titles is not None:
            a.set_title(titles[n], fontsize=5, y=0.9)
    #fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.subplots_adjust(wspace=.05, hspace=.05, )#left=0, bottom=0, right=1, top=1)
    if isinstance(name, str):
        name += '.png'
    plt.savefig(name, dpi=dpi)
    plt.close('all')

# def extract_cluster_vis(opts):
#     workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
#     np.random.seed(cfg.RNG_SEED)
#     assert 'npy' not in cfg.DATASET
#     # create the model in test mode only
#     assert opts.data_type in ['train', 'val', 'test'], "Please specify valid type."
#     print(cfg.TEST.LABELS_FILE)
#     model = model_builder.ModelBuilder(
#         name='{}_test'.format(cfg.MODEL.MODEL_NAME), train=False,
#         use_cudnn=True, cudnn_exhaustive_search=True, split=opts.data_type
#     )
#     model.build_model()
#     model.create_net()
#     model.start_data_loader()
#
#     # initialize the model
#     if cfg.TEST.PARAMS_FILE:
#         checkpoints.load_model_from_params_file(
#             model, params_file=cfg.TEST.PARAMS_FILE, checkpoint_dir=None)
#     else:
#         logger.info('=====WARN: No params files specified for testing model!')
#
#     def init():
#         global img_features, img_imgs, img_targets
#         # initialize the dictionary to store features and targets
#         img_features, img_targets, img_imgs = {}, {}, {}
#         for bl in cfg.MODEL.EXTRACT_BLOBS:
#             img_features[bl], img_targets[bl], img_imgs[bl] = {}, {}, {}
#     init()
#     # we keep track of data indices seen so far. This ensures we extract feature
#     # for each image only once.
#     indices_list = []
#     total_test_iters = helpers.get_num_test_iter(model.input_db)
#     logger.info('Test epoch iters: {}'.format(total_test_iters))
#     # when we extract features, we run 4 epochs to make sure we capture all the
#     # data points. This is needed because we use the multi-processing dataloader
#     # which shuffles the data. In very low-shot setting, making multiple passes
#     # over the entire data becomes crucial.
#     # extraction_iters = int(total_test_iters * 4)
#     extraction_iters = int(total_test_iters)
#     for test_iter in tqdm(range(0, extraction_iters)):
#         workspace.RunNet(model.net.Proto().name)
#         if test_iter == 0:
#             helpers.print_net(model)
#         if test_iter % 100 == 0:
#             logger.info('at: [{}/{}]'.format(test_iter, extraction_iters))
#         for device in range(cfg.NUM_DEVICES):
#             # 20 * ncrop
#             imgs = workspace.FetchBlob('gpu_{}/data'.format(device))
#             imgs = deprocess_imgs(imgs).astype(np.uint8)
#             # 20
#             indices = workspace.FetchBlob('gpu_{}/db_indices'.format(device))
#             indices_list.extend(list(indices))
#             num_images = indices.shape[0]
#             # 20
#             labels = workspace.FetchBlob('gpu_{}/labels'.format(device))
#             for bl in cfg.MODEL.EXTRACT_BLOBS:
#                 if cfg.EXTRACT.SPATIAL:
#                     features = workspace.FetchBlob('gpu_{}/{}'.format(device, bl))
#                 else:
#                     features = workspace.FetchBlob('gpu_{}/{}_avg'.format(device, bl))
#                 for idx in range(num_images):
#                     index = indices[idx]
#                     if not (index in img_targets[bl]):
#                         img_targets[bl][index] = labels[idx].reshape(-1)
#                         for i in range(cfg.TEST.TEN_CROP_N):
#                             img_features[bl][index * cfg.TEST.TEN_CROP_N + i] = features[idx * cfg.TEST.TEN_CROP_N + i]
#                             img_imgs[bl][index * cfg.TEST.TEN_CROP_N + i] = imgs[idx * cfg.TEST.TEN_CROP_N + i]
#                             'img_imgs': np.array(list(img_imgs[bl].values()), dtype=np.uint8),
#
#     img_features[bl] = dict(sorted(img_features[bl].items()))
#     img_imgs[bl] = dict(sorted(img_imgs[bl].items()))
#     img_features[bl] = np.array(list(img_features[bl].values()))
#     N = feats.shape[0]
#     img_features.reshape(N, -1)
#     logger.info('got image features: {} {}'.format(bl, feats.shape))
#     logger.info('All Done!')
#     # shut down the data loader
#     model.data_loader.shutdown_dataloader()

def faisstest(opts):
    d = 64                           # dimension
    nb = 100000                      # database size
    nq = 10000                       # nb of queries
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    import faiss                   # make faiss available
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)   # build the index
    print(index.is_trained)
    index.add(xb)                  # add vectors to the index
    print(index.ntotal)
    k = 4                          # we want to see 4 nearest neighbors
    D, I = index.search(xb[:5], k) # sanity check
    print(I)
    print(D)
    D, I = index.search(xq, k)     # actual search
    print(I[:5])                   # neighbors of the 5 first queries
    print(I[-5:])                  # neighbors of the 5 last queries
    from IPython import embed ; embed()

def visap(opts):
    aps = {}
    suffix = {}
    for folder in os.listdir(opts.output_dir):
        f = os.path.join(opts.output_dir, folder, 'test_ap.npy')
        if os.path.exists(f):
            logger.info('loading ' + f)
            ap = np.load(f).mean()
            aps[folder] = ap

    sortedaps  = defaultdict(list)
    prefix = ['svm5005_', 'svm10010_', 'svmlatest', 'svmbestModel_']
    for k in aps.keys():
        stripname = k
        for i in prefix:
            stripname = stripname .lstrip(i)
        if stripname in sortedaps:
            continue
        for i in prefix:
            name = prefix + stripname
            if name in aps:
                sortedaps[stripname].append(aps[name])
            else:
                logger.info(stripname + ' not found')
                sortedaps[stripname].append(0.)

    for k, v in sortedaps.items():
        plt.plot(v, label=k)
    plt.xlabel([prefix.rstrip('_') for i in prefix])
    plt.savefig('visap.png')

def main():
    parser = argparse.ArgumentParser(description='Self-Supervision Benchmark')
    parser.add_argument('-a', '--action', type=str, default='extractonce',
                            help='extractonce | extract | cluster | viscluster | visap')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output path where to extract the features/labels')
    parser.add_argument('--reference_dir', type=str, default=None,
                        help='path where take imgs for reference')
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
    # assert os.path.exists(args.output_dir), "Output directory does not exist!"
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
    print_cfg()
    globals()[args.action](args)

if __name__ == '__main__':
    main()
