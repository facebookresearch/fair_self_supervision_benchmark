# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Kaiming: adapted from DeepCluster

import time
from collections import defaultdict
import logging

import faiss
import numpy as np
from scipy.spatial.distance import cdist
from numpy import linalg as LA
import os

from caffe2.python import workspace

from self_supervision_benchmark.core.config import config as cfg

__all__ = ['Kmeans']
logger = logging.getLogger(__name__)


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # get the transform: x_out = np.dot(x, A.T) + b
    A = faiss.vector_to_array(mat.A).reshape(mat.d_out, mat.d_in)
    b = faiss.vector_to_array(mat.b)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata, [A, b]


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    Dist, Ind = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    centers = faiss.vector_to_array(clus.centroids).reshape(nmb_clusters, -1)
    kmeans_trans_weight = -2 * centers
    kmeans_trans_bias = (centers**2).sum(axis=1)  # (K,)

    return [int(n[0]) for n in Ind], [float(n[0]) for n in Dist], losses[-1], clus, \
        [kmeans_trans_weight, kmeans_trans_bias], centers


class Kmeans:
    def __init__(self, k, pca_dim=256, preprocess=True):
        self.k = k
        self.pca_dim = pca_dim

        self.pca_trans = [None, None]
        self.kmeans_trans = None
        self.centroids = None
        self.preprocess = preprocess

    def train(self, data, verbose=True):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        if self.preprocess:
            # PCA-reducing, whitening and L2-normalization
            xb, self.pca_trans = preprocess_features(data, self.pca_dim)
            if verbose:
                print('preprocess time: {0:.0f} s'.format(time.time() - end))
        else:
            xb = data
        # cluster the data
        I, D, loss, clus, self.kmeans_trans, centers = run_kmeans(xb, self.k, verbose)
        self.centroids = centers
        self.images_lists = defaultdict(list)
        self.dist_lists = defaultdict(list)
        for i, (cls, dist) in enumerate(zip(I, D)):
            self.images_lists[cls].append(i)
            self.dist_lists[cls].append(dist)
        for k, v in self.images_lists.items():
            idx = np.argsort(self.dist_lists[k])
            self.images_lists[k] = [v[i] for i in idx]

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))
        return loss

# This function looks at all the iters checkpointed and returns latest iter file
def get_centroids_file(checkpoint_dir):
    all_files = os.listdir(checkpoint_dir)
    all_iters = []
    for f in all_files:
        if 'centroids_iter' in f:
            iter_num = int(f.replace('.npz', '').replace('centroids_iter', ''))
            all_iters.append(iter_num)
    if len(all_iters) > 0:
        all_iters.sort(reverse=True)
        last_iter = int(all_iters[0])
        filepath = os.path.join(
            checkpoint_dir, 'centroids_iter{}.npz'.format(last_iter)
        )
        return filepath
    else:
        return None


class AssignClusterOp(object):
    """
    inputs:
        features: 8N x C
    outputs:
        assignment: N x K or 8N
        dists: 8N x K
    """
    def _load_centroids(self):
        from self_supervision_benchmark.utils import checkpoints
        import os
        checkpoint_dir = checkpoints.get_checkpoint_directory()
        logger.info('checking if clusters exist in directory: {}'.format(checkpoint_dir))
        latest = get_centroids_file(checkpoint_dir)
        if latest is not None:
            name = latest
        else:
            if cfg.MIDLEVEL.PREPROCESS:
                name = cfg.MIDLEVEL.CENTROIDS_FILE + str(cfg.MODEL.NUM_CLASSES) + 'pre.npz'
            else:
                name = cfg.MIDLEVEL.CENTROIDS_FILE + str(cfg.MODEL.NUM_CLASSES) + '.npz'
        logger.info('loading centroids from ' + name)
        npzfile = np.load(name)
        if cfg.MIDLEVEL.PREPROCESS:
            self.A, self.b = npzfile['arr_1'], npzfile['arr_2']
        centroids = npzfile['arr_0']
        assert len(centroids.shape) == 2
        return centroids

    def __init__(self):
        self.usefaiss = True if (cfg.MIDLEVEL.FAISS_ASSIGN and cfg.MIDLEVEL.GAUSSIAN_TOPK > 0 and cfg.MIDLEVEL.GAUSSIAN_TOPK < 1024) else False
        if self.usefaiss:
            if not hasattr(self, 'index'):
                res = faiss.StandardGpuResources()
                flat_config = faiss.GpuIndexFlatConfig()
                flat_config.useFloat16 = False
                device = workspace.GetNameScope()
                logger.info('init faiss on ' + device)
                device = int(device.lstrip('gpu_').rstrip('/'))
                flat_config.device = device
                centroids = self._load_centroids()
                self.index = faiss.GpuIndexFlatL2(res, centroids.shape[1], flat_config)
                self.index.add(centroids.astype('float32'))
                self.yy = None
        else:
            self.centroids = self._load_centroids()

        self.sigma = cfg.MIDLEVEL.GAUSSIAN_SIGMA

        # from IPython import embed ; embed()


    def forward(self, inputs, outputs):
        inp_shape = inputs[0].data.shape
        assert len(inp_shape) == 2, inp_shape

        if cfg.MIDLEVEL.PREPROCESS:
            features = np.dot(inputs[0].data, self.A.T) + self.b
            # L2 normalization
            row_sums = np.linalg.norm(features, axis=1)
            features = features / row_sums[:, np.newaxis]
        else:
            features = inputs[0].data
        if self.usefaiss:
            assert cfg.MIDLEVEL.GAUSSIAN_TOPK > 0, 'gpu search limit'
            assert cfg.MIDLEVEL.GAUSSIAN_TOPK < 1024, 'gpu search limit'
            # N x topK, sorted
            dists, Ind = self.index.search(features, cfg.MIDLEVEL.GAUSSIAN_TOPK)
        else:
            # N x K
            dists = cdist(features, self.centroids)

        if cfg.MIDLEVEL.GAUSSIAN:
            # soft assignment
            if self.usefaiss:
                if self.yy is None or self.yy.shape[0] != inp_shape[0]:
                    self.yy = np.tile(np.arange(inp_shape[0]), (cfg.MIDLEVEL.GAUSSIAN_TOPK, 1)).T
                assignment = np.exp(-dists / (2 * self.sigma ** 2))
                tmp = np.zeros((assignment.shape[0], cfg.MODEL.NUM_CLASSES),
                    dtype=assignment.dtype)
                tmp[self.yy, Ind] = assignment
                assignment = tmp
            else:
                assignment = np.exp(-dists ** 2 / (2 * self.sigma ** 2))
                if cfg.MIDLEVEL.GAUSSIAN_TOPK != -1:
                    idxs = assignment.argsort()[:, :-cfg.MIDLEVEL.GAUSSIAN_TOPK]
                    yy = np.tile(np.arange(inp_shape[0]), (cfg.MODEL.NUM_CLASSES - cfg.MIDLEVEL.GAUSSIAN_TOPK, 1)).T
                    assignment[yy, idxs] = 0
            if False: # sanity check
                dists = cdist(features, self._load_centroids())
                newassignment = np.exp(-dists ** 2 / (2 * self.sigma ** 2))
                if cfg.MIDLEVEL.GAUSSIAN_TOPK != -1:
                    idxs = newassignment.argsort()[:, :-cfg.MIDLEVEL.GAUSSIAN_TOPK]
                    yy = np.tile(np.arange(inp_shape[0]), (cfg.MODEL.NUM_CLASSES - cfg.MIDLEVEL.GAUSSIAN_TOPK, 1)).T
                    newassignment[yy, idxs] = 0
                # from IPython import embed ; embed()
                print((assignment - newassignment)[~np.isclose(assignment, newassignment)])
                # assert np.isclose(newassignment, assignment).all()
            assignment = assignment / np.sum(assignment, axis=1)[:, np.newaxis]
            assignment = assignment.reshape(8, -1, cfg.MODEL.NUM_CLASSES, 1, 1)
            individual_labels = assignment.squeeze()
            assignment = assignment.mean(0)
            outputs[0].reshape(assignment.shape)

            labels = assignment.squeeze().argmax(1)
            outputs[1].reshape(labels.shape)
            outputs[1].data[...] = labels

            outputs[2].reshape(individual_labels.shape)
            outputs[2].data[...] = individual_labels

        else:
            assignment = dists.argmin(1)
            outputs[0].reshape((inp_shape[0],))

        outputs[0].data[...] = assignment
        # outputs[1].reshape(dists.shape)
        # outputs[1].data[...] = dists

class SelectPatchOp(object):
    """select a patch for predicting surrounding patch clusters
    inputs:
        features: 9 x N x C
    outputs:
        surrounding patches: 8 x N x C
        selected patches: N x C
    """
    def __init__(self):
        assert cfg.MIDLEVEL.SELECTED_PATCH >= -1 and cfg.MIDLEVEL.SELECTED_PATCH < cfg.JIGSAW.NUM_PATCHES, cfg.MIDLEVEL.SELECTED_PATCH

    def forward(self, inputs, outputs):
        inp_shape = inputs[0].data.shape
        assert len(inp_shape) == 3, inp_shape
        assert inp_shape[0] == cfg.JIGSAW.NUM_PATCHES, inp_shape
        features = inputs[0].data

        if cfg.MIDLEVEL.SELECTED_PATCH == -1:
            idx = np.random.randint(cfg.JIGSAW.NUM_PATCHES)
        else:
            idx = cfg.MIDLEVEL.SELECTED_PATCH
        selected = features[idx]
        surrounding = np.vstack([features[:idx], features[idx + 1:]])

        outputs[0].reshape(selected.shape)
        outputs[0].data[...] = selected
        outputs[1].reshape(surrounding.shape)
        outputs[1].data[...] = surrounding
