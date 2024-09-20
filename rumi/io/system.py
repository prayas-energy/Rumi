import time
import os
from rumi.io import config, filemanager
from rumi.io import logger as rumilogger


class ScenarioError(Exception):
    pass


class SystemLauncher(metaclass=config.Singleton):
    """A class to initialize config and logger for the system
    """

    def __init__(self,
                 model_instance_path,
                 scenario,
                 component_name,
                 logger_level,
                 create_scenario=True,
                 **kwargs):
        config.initialize_config(model_instance_path, scenario)
        for name, value in kwargs.items():
            if value:
                config.set_config(name, value)

        if not create_scenario:
            if not os.path.exists(filemanager.scenario_path()):
                print(f"Invalid scenario: {scenario}")
                raise ScenarioError(f"{scenario} does not exist")

        rumilogger.init_logger(component_name, logger_level)

    @staticmethod
    def shutdown():
        """waits for logger to finish writing and then shuts down the logger process
        """
        while not rumilogger.get_queue().empty():
            time.sleep(1)
        rumilogger.get_event().set()
