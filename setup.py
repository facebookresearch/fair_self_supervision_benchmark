# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages


###############################################################################
# Pure python packages
###############################################################################
packages = find_packages()

setup(
    name='self_supervision_benchmark',
    packages=packages,
)
