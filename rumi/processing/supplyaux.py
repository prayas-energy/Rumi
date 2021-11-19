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
"""Wrapper module over supply main script to provide API access.
"""
import sys
import os
import subprocess


def rumi_supply(model_instance_path,
                scenario,
                output_folder=None):
    """Supply processing 

    :param: model_instance_path -> Path of the model instance top-level folder
    :param: scenario -> Name of the scenario within specified model
    :param: output_folder -> Path of the output folder (optional)

    """

    cmd = " ".join(['rumi_supply',
                    "-m", model_instance_path,
                    "-s", scenario])
    if output_folder:
        cmd = " ".join([cmd, "-o", output_folder])

    p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    p.wait()
    print(p.stdout.read().decode())
    print(p.stderr.read().decode())


if __name__ == "__main__":
    pass
