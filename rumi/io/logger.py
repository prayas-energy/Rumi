# Copyright 2021 Prayas Energy Group(https://www.prayaspune.org/peg/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import functools
import logging
import logging.handlers
import datetime
from time import sleep
import multiprocessing
from rumi.io import filemanager
from rumi.io import config


levels = {"INFO": logging.INFO,
          "WARN": logging.WARN,
          "DEBUG": logging.DEBUG,
          "ERROR": logging.ERROR}


def get_log_filepath(spec_type):
    prefix = config.get_config_value("scenario")
    location = filemanager.get_output_path(spec_type)
    format_ = "%Y-%m-%d-%H-%M"
    timestamp = datetime.datetime.now().strftime(format_)
    return os.path.join(location, ".".join(["rumi", prefix, timestamp, "log"]))


def listener_configurer(filename, level):
    print("Redirecting log to ", filename)
    filehandler = logging.FileHandler(filename=filename, mode="w")
    formatter = logging.Formatter(
        '%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')

    filehandler.setLevel(levels[level])
    filehandler.setFormatter(formatter)

    # consol = logging.StreamHandler()
    # consol.setLevel(l['INFO'])
    logging.basicConfig(handlers=[filehandler], level=levels[level])


def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)


def listener_process(filename, level, queue, event: multiprocessing.Event):
    listener_configurer(filename, level)
    while not event.is_set():
        while not queue.empty():
            record = queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)
        sleep(1)


@functools.lru_cache(maxsize=None)
def get_queue():
    return multiprocessing.Manager().Queue(-1)


@functools.lru_cache(maxsize=None)
def get_event():
    return multiprocessing.Event()


def init_logger(spec_type, level="INFO"):
    queue = get_queue()
    stopevent = get_event()
    filename = get_log_filepath(spec_type)

    listener = multiprocessing.Process(target=listener_process,
                                       args=(filename,
                                             level,
                                             queue,
                                             stopevent))
    listener.start()
    worker_configurer(queue)
