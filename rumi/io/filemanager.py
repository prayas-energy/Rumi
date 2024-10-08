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
"""folder structure and yaml files are handled by this module.
this module can depend only on config, customyaml and python
modules.
"""
from rumi.io import customyaml
from rumi.io import config
import os
import yaml
import functools


class FolderStructureError(Exception):
    pass


def find_location_(name, folder_structure, path):
    if name in folder_structure:
        return os.path.join(*path)

    for key in folder_structure:
        if isinstance(folder_structure[key], dict):
            x = find_location_(name, folder_structure[key], path + [key])
            if x:
                return x


def folder_structure():
    yaml_location = config.get_config_value("yaml_location")
    return customyaml.load_yaml(os.path.join(yaml_location,
                                             "folder_structure.yml"))


@functools.lru_cache(maxsize=4)
def _load_specs(type_):
    yaml_location = config.get_config_value("yaml_location")
    specs_file = os.path.join(yaml_location, f"{type_}.yml")
    with open(specs_file) as file:
        return yaml.full_load(file)


def common_specs():
    return _load_specs("Common")


def demand_specs():
    return _load_specs("Demand")


def supply_specs():
    return _load_specs("Supply")


@functools.lru_cache(maxsize=None)
def find_global_location(name):
    """finds location of parameter in Default Data
    this is not complete path, it is only relative
    to model instance
    """
    fs = folder_structure()
    path = find_location_(name, fs, [])
    if not path:
        raise FolderStructureError(f"path for {name} not found.")
    return path


def scenario_path():
    """only return complete path of scenario.
    """
    scenario_name = config.get_config_value("scenario")
    instance_path = config.get_config_value("model_instance_path")
    return os.path.join(instance_path, "Scenarios", scenario_name)


def scenario_location():
    """return complete path of scenario.
    also creates the folder if it does not exists.
    """
    path = scenario_path()
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_custom_output_path(spec_type, output_path):
    scenario_name = config.get_config_value("scenario")
    return os.path.join(output_path,
                        scenario_name,
                        spec_type,
                        'Output')


def get_output_path(spec_type):
    """depening spec_type returns output path
    """
    output_path = config.get_config_value("output")

    if output_path:
        output_path = get_custom_output_path(spec_type, output_path)
    else:
        scenario_path_ = scenario_location()
        output_path = os.path.join(scenario_path_, spec_type, 'Output')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path


def get_specs(name):
    if name in (common_specs() or {}):
        return common_specs()[name]
    elif name in (demand_specs() or {}):
        return demand_specs()[name]
    elif name in (supply_specs() or {}):
        return supply_specs()[name]
    else:
        raise FolderStructureError(f"Unknown parameter {name}")


def get_type_specs(name):
    if name == "Common":
        return common_specs()
    elif name == "Demand":
        return demand_specs()
    elif name == "Supply":
        return supply_specs()
    else:
        raise FolderStructureError(f"No such parameter type, {name}")


def filename(name, prefix=""):
    """Return filename for given parameter"""
    return "{}{}.{}".format(prefix, name, get_specs(name)['filetype'])


def find_filepath(name, *subfolder, fileprefix=""):
    """Finds path where the parameter is stored.

    Paramters
    ---------
      param_name: str
         Name of parameter

    Returns
    -------
    If parameter is overridded in scenarios, then path of 
    parameter from scenario is returned else global path
    is returned
    """
    prefix = config.get_config_value("model_instance_path")
    global_path = find_global_location(name)
    possible_path = global_path.replace("Default Data",
                                        scenario_location())
    filename_ = filename(name, fileprefix)

    filepath = os.path.join(possible_path, *subfolder, filename_)
    if os.path.exists(filepath):
        return filepath
    else:
        return os.path.join(prefix, global_path, *subfolder, filename_)


def get_config_parameter_path(param_name):
    """Finds path where Config parameter is stored.

    Paramters
    ---------
      param_name: str
         Name of parameter

    Returns
    -------
    If parameter is overridden in Model Instance, then path of 
    parameter from Model Instance is returned else platform path
    is returned
    """

    platform_config_dir = config.get_config_value("yaml_location")
    model_instance_path = config.get_config_value('model_instance_path')
    instance_config_dir = os.path.join(model_instance_path, 'Config')

    filename = ".".join([param_name, 'csv'])
    path = os.path.join(platform_config_dir, filename)
    possible_path = os.path.join(instance_config_dir, filename)

    if os.path.exists(possible_path):
        return possible_path
    else:
        return path


if __name__ == "__main__":
    pass
