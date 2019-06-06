# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

"""
Helper functions for input/output.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import errno
import logging
import os
import re
import six
import sys
from six.moves import cPickle as pickle
from six.moves import urllib
from uuid import uuid4

# create the logger
logger = logging.getLogger(__name__)


S3_BASE_URL = 'https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark'


def save_object(obj, file_name, pickle_format=2):
    """
    Credits:
    https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/io.py
    Save a Python object by pickling it. Unless specifically overridden, we want
    to save it in Pickle format=2 since this will allow other Python2 executables
    to load the resulting Pickle. When we want to completely remove Python2
    backward-compatibility, we can bump it  up to 3. We should never use
    pickle.HIGHEST_PROTOCOL as far as possible if the resulting file is
    manifested or used, external to the system.
    """
    file_name = os.path.abspath(file_name)
    # Avoid filesystem race conditions (particularly on network filesystems)
    # by saving to a random tmp file on the same filesystem, and then
    # atomically rename to the target filename.
    tmp_file_name = file_name + ".tmp." + uuid4().hex
    try:
        with open(tmp_file_name, 'wb') as f:
            pickle.dump(obj, f, pickle_format)
            f.flush()  # make sure it's written to disk
            os.fsync(f.fileno())
        os.rename(tmp_file_name, file_name)
    finally:
        # Clean up the temp file on failure. Rather than using os.path.exists(),
        # which can be unreliable on network filesystems, attempt to delete and
        # ignore os errors.
        try:
            os.remove(tmp_file_name)
        except EnvironmentError as e:  # parent class of IOError, OSError
            if getattr(e, 'errno', None) != errno.ENOENT:  # We expect ENOENT
                logger.info("Could not delete temp file %r",
                    tmp_file_name, exc_info=True)
                # pass through since we don't want the job to crash


def load_object(file_name):
    """
    Credits:
    https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/io.py
    """
    with open(file_name, 'rb') as f:
        # The default encoding used while unpickling is 7-bit (ASCII.) However,
        # the blobs are arbitrary 8-bit bytes which don't agree. The absolute
        # correct way to do this is to use `encoding="bytes"` and then interpret
        # the blob names either as ASCII, or better, as unicode utf-8. A
        # reasonable fix, however, is to treat it the encoding as 8-bit latin1
        # (which agrees with the first 256 characters of Unicode anyway.)
        if six.PY2:
            return pickle.load(f)
        else:
            return pickle.load(f, encoding='latin1')


def cache_url(params_file, cache_dir):
    """
    We download the urls to the cache_dir and use that.
    """
    is_url = re.match(r'^(?:http)s?://', params_file, re.IGNORECASE) is not None
    if not is_url:
        return params_file

    url = params_file
    cache_file_path = url.replace(S3_BASE_URL, cache_dir)
    if os.path.exists(cache_file_path):
        return cache_file_path

    cache_file_dir = os.path.dirname(cache_file_path)
    if not os.path.exists(cache_file_dir):
        os.makedirs(cache_file_dir)

    logger.info('Downloading remote file {} to {}'.format(url, cache_file_path))
    download_url(url, cache_file_path)
    return cache_file_path


def download_url(url, cache_file, chunk_size=8192):
    """
    Download url and write it to cache_file.
    Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    response = urllib.request.urlopen(url)
    if six.PY2:
        total_size = response.info().getheader('Content-Length').strip()
    else:
        total_size = response.info().get('Content-Length').strip()
    total_size = int(total_size)
    bytes_so_far = 0

    with open(cache_file, 'wb') as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            f.write(chunk)

    return bytes_so_far
