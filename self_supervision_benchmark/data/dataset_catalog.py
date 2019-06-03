# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
Data and labels file for various datasets.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os

from self_supervision_benchmark.core.config import config as cfg


def get_data_labels_file(dataset, partition):
    data_type = 'TRAIN' if partition == 'train' else 'TEST'
    data_file = cfg[data_type].DATA_FILE
    labels_file = cfg[data_type].LABELS_FILE
    assert os.path.exists(data_file), "Please specify {} data file".format(data_type)
    assert os.path.exists(labels_file), "Please specify {} labels file".format(data_type)
    output = [data_file, labels_file]
    return output
