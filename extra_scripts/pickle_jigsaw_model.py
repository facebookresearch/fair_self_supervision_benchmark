# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
Given a model, this script converts the blobs names to the detection blob names
so we can finetune the model. For the jigsaw pretext task, we keep blobs from
one tower, remove the other towers and also remove the suffix s{} from the name.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import argparse
import logging
import os
import re
import six
import sys
from six.moves import cPickle as pickle

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def pickle_jigsaw_model(opts, pickle_format=2):
    assert os.path.exists(opts.model_file), "Model file does NOT exist. Abort!"
    # load the weights first
    with open(opts.model_file, 'rb') as fopen:
        if six.PY2:
            data = pickle.load(fopen)
        else:
            data = pickle.load(fopen, encoding='latin1')
    data = data['blobs']

    output_blobs, count = {}, 0
    skip_suffixes = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    logger.info('Converting jigsaw model..................')
    for item in sorted(data.keys()):
        if 's0' in item:
            # remove s0 from the blob names.
            out_name = re.sub('_s[0-9]_', '_', item)
            logger.info('input_name: {} out_name: {}'.format(item, out_name))
            output_blobs[out_name] = data[item]
        elif any(x in item for x in skip_suffixes):
            # skip the blobs for the other towers
            logger.info('IGNORING: {}'.format(item))
            count += 1
        else:
            # add any other blobs to the output
            logger.info('adding: {}'.format(item))
            output_blobs[item] = data[item]

    logger.info('Original #blobs: {}'.format(len(data.keys())))
    logger.info('Output #blobs: {}'.format(len(output_blobs.keys())))
    logger.info('Removed #blobs: {}'.format(count))

    logger.info('Saving converted model.................')
    with open(opts.output_file, 'wb') as fopen:
        pickle.dump({'blobs': output_blobs}, fopen, pickle_format)
    logger.info('Wrote blobs to: {}'.format(opts.output_file))
    return output_blobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default=None,
                        help='Input model file .pkl')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output model file .pkl')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    pickle_jigsaw_model(args)


if __name__ == '__main__':
    main()
