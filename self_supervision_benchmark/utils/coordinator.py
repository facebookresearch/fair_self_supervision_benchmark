# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import contextlib
import logging
import os
import threading
import traceback
from six.moves import queue as Queue

logger = logging.getLogger(__name__)


class Coordinator(object):

    def __init__(self):
        self.__event = threading.Event()

    def request_stop(self):
        logger.debug("Coordinator stop requested. Stopping")
        self.__event.set()

    def should_stop(self):
        return self.__event.is_set()

    def wait_for_stop(self):
        return self.__event.wait()

    @contextlib.contextmanager
    def stop_on_execution(self):
        try:
            yield
        except Exception:
            if not self.should_stop():
                traceback.print_exc()
                self.request_stop()
                os._exit(0)


def coordinated_get(coordinator, queue):
    while not coordinator.should_stop():
        try:
            return queue.get(block=True, timeout=0.1)
        except Queue.Empty:
            continue
    return None


def coordinated_put(coordinator, queue, element):
    while not coordinator.should_stop():
        try:
            queue.put(element, block=True, timeout=0.1)
            return
        except Queue.Full:
            continue

    logger.debug("Coordinator stopped during put()")
    return None
