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
import logging
import datetime
from rumi.io import filemanager
from rumi.io import config


def get_log_filepath(spec_type):
    prefix = config.get_config_value("scenario")
    location = filemanager.get_output_path(spec_type)
    format_ = "%Y-%m-%d-%H-%M"
    timestamp = datetime.datetime.now().strftime(format_)
    return os.path.join(location, ".".join(["rumi", prefix, timestamp, "log"]))


def init_logger(spec_type, level="INFO"):
    l = {"INFO": logging.INFO,
         "WARN": logging.WARN,
         "DEBUG": logging.DEBUG,
         "ERROR": logging.ERROR}
    filepath = get_log_filepath(spec_type)
    print("Redirecting log to ", filepath)
    filehandler = logging.FileHandler(filename=filepath, mode="w")
    formatter = logging.Formatter(
        '%(name)s : %(levelname)s : %(message)s')
    filehandler.setLevel(l[level])
    filehandler.setFormatter(formatter)

    #consol = logging.StreamHandler()
    # consol.setLevel(l['INFO'])
    logging.basicConfig(handlers=[filehandler], level=l[level])
