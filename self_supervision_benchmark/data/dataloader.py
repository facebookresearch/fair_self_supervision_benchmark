# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
This script is responsible for loading the data per device (GPU).
The data is loaded per type: train/val/test. The loader used multiprocessing and
multi-threading for enabling high data throughput.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import logging
import numpy as np
import random
import signal
import threading
import time
import uuid
from collections import OrderedDict
from six.moves import queue as Queue

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.data.image_dataset import ImageDataset
from self_supervision_benchmark.data.data_input import create_data_input
from self_supervision_benchmark.utils import helpers
from self_supervision_benchmark.utils.coordinator import (
    Coordinator, coordinated_put, coordinated_get,
)

from caffe2.python import core, workspace, scope
from caffe2.proto import caffe2_pb2

logger = logging.getLogger(__name__)

# ImageDataset is the common class to load data for the datasets below. If your
# data needs to be loaded in a different way, you can add a new class which
# provides 2 functions: get_db_size() and get_minibatch_path_indexes().
db_loader_map = {
    'coco': ImageDataset,
    'imagenet1k': ImageDataset,
    'voc2007': ImageDataset,
    'voc2012': ImageDataset,
    'places205': ImageDataset,
}


def get_input_db(dataset, data_type):
    assert dataset in db_loader_map.keys(), \
        "Unknown dataset: {}".format(dataset)
    # this db class should have a get_db_size() and get_minibatch_path_indexes()
    # function to get the data.
    input_db = db_loader_map[dataset](split=data_type)
    return input_db


class DataLoader(object):

    def __init__(
        self,
        split,
        input_db,
        batch_size,
        preprocess,
        num_workers=4,
        # num_workers=8,
        # num_workers=12,
        num_processes=4,
        num_enqueuers=1,
        # minibatch_queue_size=128,
        minibatch_queue_size=64,
        blobs_queue_capacity=1,
    ):
        self._split = split
        self._input_db = input_db
        self._db_size = input_db.get_db_size()
        self._lock = threading.Lock()
        self._batch_size = batch_size
        self._current = 0
        self._preprocess = preprocess
        self._perm = np.arange(input_db.get_db_size())

        self.coordinator = Coordinator()
        self._num_devices = cfg.NUM_DEVICES
        self._num_workers = num_workers
        self._num_processes = num_processes
        self._num_enqueuers = num_enqueuers
        self._minibatch_queue_capacity = minibatch_queue_size
        self._minibatch_queue = Queue.Queue(maxsize=minibatch_queue_size)
        self._device_blobs_queue_capacity = blobs_queue_capacity
        self._blobs_queue_name = '{}_blobs_queue_{}'.format(
            cfg.DATASET, str(uuid.uuid4())
        )

        # we assign indexes to the blobs so they can be queued in the same order
        self._blobs_idx_map = OrderedDict()
        self._blobs_idx_map['data'] = 0
        self._blobs_idx_map['labels'] = 1
        self._blobs_idx_map['db_indices'] = 2
        self._blobs_idx_map['img_weight'] = 3

        # get the expected data size - to be used for declaring memory for
        # multiprocessing
        self._crop_size = cfg.TRAIN.CROP_SIZE
        self._expected_data_size = 3 * self._crop_size ** 2
        if split in ['test', 'val'] and cfg.TEST.TEN_CROP:
            self._expected_data_size = 10 * self._expected_data_size

        if split == 'train':
            self._get_data_perm()
        self.create_threads()
        # multi-processing is supported for imagenet and bigger datasets only.
        # For cifar like small datasets, it wouldn't help since entire data can
        # be loaded in memory.
        self._create_data_input()

    def _get_data_perm(self):
        """Randomly permute the database"""
        indices = self._perm
        random.shuffle(indices)
        self._perm = indices
        # set the current dataloader pointer to 0
        self._current = 0
        return None

    def get_worker_ids(self):
        # differentiate between train data and test data workers
        if self._split == 'train':
            return range(0, self._num_workers)
        else:
            return range(100, 100 + self._num_workers)

    def _create_data_input(self):
        (context_execution, fetch_func) = create_data_input(
            self._input_db, self._expected_data_size, self._num_processes,
            self._num_workers, self._split, self._batch_size
        )
        self._context_execution = context_execution
        self._minibatch_fetch_func = fetch_func
        worker_ids = self.get_worker_ids()
        self._context_execution(worker_ids)

    def minibatch_queue_size(self):
        return self._minibatch_queue.qsize()

    def get_blob_names(self):
        return list(self._blobs_idx_map.keys())

    def _get_next_minibatch_indices(self):
        """
        Randomly sample minibatch indices given the data perm. We will get
        information about data at these indices from the ImageDataset class.
        While selecting the minibatches, we also put a lock on the permuation
        so that multiple worker threads don't pick the same minibatch indices.
        """
        db_size = self._db_size
        with self._lock:
            if self._split == 'train':
                if ((self._current + self._batch_size) >= db_size):
                    self._get_data_perm()   # the current pointer will be 0
                db_indices = self._perm[
                    self._current:self._current + self._batch_size
                ]
                self._current += self._batch_size
                return db_indices
            elif self._split in ['test', 'val']:
                if (self._current >= db_size):
                    self._current = 0
                end_idx = min(self._current + self._batch_size, db_size)
                db_indices = self._perm[self._current:end_idx]
                self._current += self._batch_size
                return db_indices

    def _get_next_minibatch(self, worker_id):
        """
        Returns next blobs to be used for the next mini-batch queue
        """
        db_indices = self._get_next_minibatch_indices()
        minibatch_data, minibatch_labels, weights = self._minibatch_fetch_func(
            self._input_db, worker_id, self._batch_size, db_indices,
        )
        minibatch_blobs = {
            'data': minibatch_data,
            'labels': minibatch_labels,
            'db_indices': np.array(db_indices).astype(np.int32),
            'img_weight': weights,
        }
        return minibatch_blobs

    def minibatch_loader(self, worker_id):
        """Load mini-batches and put them into a queue in Device memory"""
        # for a given worker thread, we will have num_processes running sharing
        # memory buffer and putting data in it.
        with self.coordinator.stop_on_execution():
            while not self.coordinator.should_stop():
                mb_blobs = self._get_next_minibatch(worker_id)
                if (len(mb_blobs['data']) == 0 or len(mb_blobs['labels']) == 0):
                    logger.warn("Error in fetching minibatch, re-trying")
                    continue
                ordered_minibatch_blobs = OrderedDict()
                for key in self.get_blob_names():
                    ordered_minibatch_blobs[key] = mb_blobs[key]
                coordinated_put(
                    self.coordinator,
                    self._minibatch_queue,
                    ordered_minibatch_blobs,
                )
        logger.debug("Stopping mini-batch loader thread...")

    # this puts the given blob values to a GPU memory
    # multiple enqueuer threads per device
    def enqueue_blobs(self, device_id, enqueue_blobs_names, blob_values):
        # enqueue blob names is <blob_name>_enqueue_<enqueuer_thread_id>
        # for the current nameScope, feed blobs using the blob values
        prefix, device = helpers.get_prefix_and_device()
        enqueue_blobs_names = [
            '{}{}/{}'.format(
                prefix, device_id, enqueue_blob_name
            ) for enqueue_blob_name in enqueue_blobs_names
        ]
        deviceOption = core.DeviceOption(device, device_id)
        for (blob_name, blob) in zip(enqueue_blobs_names, blob_values):
            workspace.FeedBlob(blob_name, blob, device_option=deviceOption)

        workspace.RunOperatorOnce(
            core.CreateOperator(
                'EnqueueBlobs',
                [
                    '{}{}/{}'.format(prefix, device_id, self._blobs_queue_name)
                ] + enqueue_blobs_names,
                enqueue_blobs_names,
                device_option=deviceOption,
            )
        )

    def enqueue_blobs_thread(self, device_id, enqueue_blobs_names):
        """
        Transfer mini-batches from CPU mini-batch queue to a DEVICE BlobsQueue
        """
        with self.coordinator.stop_on_execution():
            while not self.coordinator.should_stop():
                blobs = coordinated_get(self.coordinator, self._minibatch_queue)
                self.enqueue_blobs(
                    device_id,
                    enqueue_blobs_names,
                    blobs.values(),
                )
        logger.debug("Stopping enqueuer thread...")

    def close_blobs_queue(self):
        """Close a BlobsQueue"""
        workspace.RunOperatorOnce(
            core.CreateOperator(
                'CloseBlobsQueue',
                [self._blobs_queue_name],
                []
            )
        )

    def create_blobs_queue(self, queue_name, num_blobs, capacity):
        """
        Create a BlobsQueue in the workspace to hold the mini-batches. Each
        DEVICE has its own workspace and we chose the namescope per DEVICE.
        The queue name has a uuid which is different for test/train data split.
        """
        workspace.RunOperatorOnce(
            core.CreateOperator(
                'CreateBlobsQueue',
                [], [queue_name],
                num_blobs=num_blobs,
                capacity=capacity,
            )
        )

    # minibatch loader threads: each thread builds minibatches and places them
    # into a queue in CPU memory
    def create_threads(self):
        # "worker" threads to construct (partial) minibatches and put them on
        # minibatch CPU queue in CPU memory (limited by queue size).
        self._worker_ids = self.get_worker_ids()
        self._workers = [
            threading.Thread(
                target=self.minibatch_loader,
                name='worker_{}'.format(worker_id),
                args=[worker_id],
            ) for worker_id in self._worker_ids
        ]

        # create one BlobsQueue per DEVICE which holds the training data in GPU
        # memory and feeds to the net
        prefix, device = helpers.get_prefix_and_device()
        # the root device id = 0
        for device_id in range(0, self._num_devices):
            with core.NameScope('{}{}'.format(prefix, device_id)):
                self.create_blobs_queue(
                    queue_name=self._blobs_queue_name,
                    num_blobs=len(self._blobs_idx_map),
                    capacity=self._device_blobs_queue_capacity
                )

        # launch enqueuer threads
        # Create one blob for each (blob_name, enqueuer_thread_id) pair:
        #  <train/test>_<blob_name>_enqueue_<enqueuer_thread_id>
        # The distinction between train/test here is important since when we use
        # EnqueueBlobs op, we need to distinguish otherwise data can get mixed.
        blob_names = self._blobs_idx_map.keys()
        enqueue_blobs_names = [
            ['{}_{}_enqueue_{}'.format(self._split, blob_name, idx)
                for blob_name in blob_names]
            for idx in range(self._num_enqueuers)
        ]
        for device_id in range(0, self._num_devices):
            # NameScope is prepended to all the blobs in the workspace
            with core.NameScope('{}{}'.format(prefix, device_id)):
                with core.DeviceScope(core.DeviceOption(device, device_id)):
                    for blob_list in enqueue_blobs_names:
                        for blob in blob_list:
                            scoped_blob_name = scope.CurrentNameScope() + blob
                            workspace.CreateBlob(scoped_blob_name)
        # create the enqueuer threads
        self._enqueuers = [
            threading.Thread(
                target=self.enqueue_blobs_thread,
                args=(device_id, enqueue_blobs_names[idx])
            )
            for device_id in range(0, self._num_devices)
            for idx in range(self._num_enqueuers)
        ]

    def prefill_minibatch_queue(self):
        logger.info('Pre-filling {} minibatch queue'.format(self._split))
        while(self.minibatch_queue_size() < self._minibatch_queue_capacity):
            time.sleep(1.0)
            logger.info('[{}/{}]'.format(
                self.minibatch_queue_size(), self._minibatch_queue_capacity))
        logger.info("{} minibatch queue pre-filled.".format(self._split))

    def start(self, prefill=False):
        for w in self._workers + self._enqueuers:
            w.daemon = True
            w.start()
        if prefill:
            self.prefill_minibatch_queue()

    def join(self):
        for w in self._workers + self._enqueuers:
            w.join()

    def shutdown_dataloader(self):
        self.coordinator.request_stop()
        self.coordinator.wait_for_stop()
        prefix, _ = helpers.get_prefix_and_device()
        for idx in range(0, self._num_devices):
            with core.NameScope("{}{}".format(prefix, idx)):
                self.close_blobs_queue()
        self.join()

    def register_sigint_handler(self):
        def signal_handler(signal, frame):
            logger.info("SIGINT: shutting down data loader threads and exiting")
            self.shutdown_dataloader()
        signal.signal(signal.SIGINT, signal_handler)
