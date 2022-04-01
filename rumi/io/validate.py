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
"""Python module to validate Rumi model instance data. 
"""
import logging
import sys
import click
from rumi.io import config
from rumi.io.logger import init_logger
from rumi.io import loaders

global logger


def rumi_validate(param_type: str,
                  model_instance_path: str,
                  scenario: str,
                  logger_level: str,
                  numthreads: int):
    """Validate Common, Demand or Supply module data

    :param: param_type: str -> one of Common, Demand, Supply
    :param: model_instance_path: str -> Path where model instance is stored
    :param: scenario: str -> Name of Scenario
    :param: logger_level: str: Level for logging,one of INFO, WARN, DEBUG or ERROR


    """
    loaders.rumi_validate(param_type=param_type,
                          model_instance_path=model_instance_path,
                          scenario=scenario,
                          logger_level=logger_level,
                          numthreads=numthreads)


@click.command()
@click.option("-p", "--param_type",
              help="Parameter type to validate. Can be one of Common, Demand or Supply.")
@click.option("-m", "--model_instance_path",
              help="Path where the model instance is located")
@click.option("-s", "--scenario",
              help="Name of the scenario")
@click.option("-l", "--logger_level",
              help="Level for logging: one of INFO, WARN, DEBUG or ERROR (default: INFO)",
              default="INFO")
@click.option("-t", "--numthreads",
              help="Number of threads/processes (default: 2)",
              default=2)
def _main(param_type: str,
          model_instance_path: str,
          scenario: str,
          logger_level: str,
          numthreads: int):
    """Command line interface for data validation.

    -m/--model_instance_path, -s/--scenario and -p/--param_type are compulsory
    named arguments. While others are optional.
    """
    rumi_validate(param_type=param_type,
                  model_instance_path=model_instance_path,
                  scenario=scenario,
                  logger_level=logger_level,
                  numthreads=numthreads)


def main():
    if len(sys.argv) == 1:
        print("To see valid options  run the command with --help")
        print("rumi_validate --help")
    else:
        _main()


if __name__ == "__main__":
    main()
