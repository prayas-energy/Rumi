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
"""Framework for configuration management. this module can not depend
on any other rumi modules. this has to be standalone module that
can depend directly on python modules only.
"""
import os
import yaml
import pkg_resources


class ConfigurationError(Exception):
    pass


class Singleton(type):

    def __init__(self, *args, **kwargs):
        self._instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self._instance:
            return self._instance
        else:
            self._instance = super().__call__(*args, **kwargs)
            return self._instance


class ConfigManager(metaclass=Singleton):

    def __init__(self, conffile=None):
        self.config = {}
        self.prefix = "RUMI_"
        if conffile:
            with open(conffile) as f:
                data = yaml.full_load(f) or {}
                for key, value in data.items():
                    self.set_config_value(key, value)

    def get_config_value(self, conf):
        value = self.config.get(conf)
        if value:
            return value
        else:
            value = os.getenv(self.get_name_in_env(conf))
            return value if value else None

    def get_name_in_env(self, name):
        return self.prefix + name

    def set_config_value(self, name, value):
        if name in self.config:
            raise ConfigurationError(
                "Configuration values once set can not be modified.")

        self.config[name] = str(value)
        env_name = self.get_name_in_env(name)
        if os.getenv(env_name):
            raise ConfigurationError(
                "Configuration values once set can not be modified.")

        os.environ[env_name] = str(value)


def set_config(name, value):
    """ Set config value for global access

    allows setting configuration value which can be accessed
    anywhere in software later.
    """
    ConfigManager().set_config_value(name, value)


def _get_model_config(model_instance_path):
    return os.path.join(model_instance_path,
                        "Config",
                        "Config.yml")


def initialize_config(model_instance_path=None,
                      scenario_name=None):
    """ Initialize config as singletone instance.
    once initialized , it can not be changed

    After initialization by default config will have
    model_instance_path, scenario, yaml_location
    """
    model_config = _get_model_config(model_instance_path)
    yaml_location = pkg_resources.resource_filename("rumi",
                                                    "Config")

    if os.path.exists(model_config):
        conffile = model_config
    else:
        conffile = os.path.join(yaml_location, "Config.yml")
    # yaml files are stored in platform/Config

    cm = ConfigManager(conffile=conffile)

    cm.set_config_value("config_location", os.path.dirname(conffile))
    cm.set_config_value("model_instance_path", model_instance_path)
    cm.set_config_value("scenario", scenario_name)
    cm.set_config_value("yaml_location", yaml_location)


def get_config_value(conf_name):
    """get value of configuration from configuration"""
    return ConfigManager().get_config_value(conf_name)
