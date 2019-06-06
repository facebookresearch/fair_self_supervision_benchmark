# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
Environment helper for importing detectron ops - we use SigmoidCrossEntropyLoss
Op from this.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import sys
import yaml

# create the logger
logger = logging.getLogger(__name__)

# Detectron ops lib name as shipped by PyTorch conda binary
_DETECTRON_OPS_LIB = 'libcaffe2_detectron_ops_gpu.so'


def get_detectron_ops_lib():
    """
    Retrieve Detectron ops library.
    """
    # Candidate prefixes for the detectron ops lib path. We install pytorch
    # which ships caffe2 detectron ops.
    prefixes = sys.path
    subdirs = ['lib', 'torch/lib']
    # Search for detectron ops lib
    for prefix in prefixes:
        for subdir in subdirs:
            ops_path = os.path.join(prefix, subdir, _DETECTRON_OPS_LIB)
            if os.path.exists(ops_path):
                logger.info('=====>Found Detectron ops lib: {}'.format(ops_path))
                return ops_path
    raise Exception('Detectron ops lib not found')


# YAML load/dump function aliases
yaml_load = yaml.load
yaml_dump = yaml.dump
