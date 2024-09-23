import os
import pytest
import shutil
from rumi.io import filemanager, config
from rumi.io.test_config import configmanager
import pkg_resources


@pytest.fixture()
def clear_filemanager_cache():
    filemanager.find_global_location.cache_clear()
    filemanager._load_specs.cache_clear()


def test_find_filepath(clear_filemanager_cache, configmanager):
    assert filemanager.find_filepath(
        "ModelPeriod") == os.path.join("Test Instance", "Default Data", "Common", "Parameters", "ModelPeriod.csv")


    global_path = os.path.join(filemanager.find_global_location("ModelPeriod"),
                               filemanager.filename("ModelPeriod"))
    instance_path = config.get_config_value("model_instance_path")
    source_path = os.path.join(instance_path, global_path)
    dest_path = global_path.replace("Default Data",
                                    filemanager.scenario_location())
    basedir = os.path.dirname(dest_path)
    os.makedirs(basedir, exist_ok=True)
    shutil.copyfile(source_path, dest_path)

    assert filemanager.find_filepath(
        "ModelPeriod") == os.path.join("Test Instance", "Scenarios", "Scenario1", "Common", "Parameters", "ModelPeriod.csv")

    os.unlink(dest_path)

    assert filemanager.find_filepath(
       "ModelPeriod") == os.path.join("Test Instance", "Default Data", "Common", "Parameters", "ModelPeriod.csv")
       
      
def test_find_filepath_empty_global(clear_filemanager_cache, tmp_path, monkeypatch):
    model_instance = tmp_path / "model_instance"
    model_instance.mkdir()

    def get_config_value(name):
        if name == "scenario":
            return "test_scenario"
        elif name == "model_instance_path":
            return model_instance.absolute()
        elif name == "yaml_location":
            yaml_location = pkg_resources.resource_filename("rumi",
                                                            "Config")
            return yaml_location
        elif name == "config_location":
            yaml_location = pkg_resources.resource_filename("rumi",
                                                            "Config")
            # for this test conf will be taken from package not from model instance
            os.path.join(yaml_location, "Config.yml")

    monkeypatch.setattr(config, "get_config_value", get_config_value)
    global_data = model_instance/"Default Data"
    global_data.mkdir()
    scenarios = model_instance / "Scenarios"
    scenarios.mkdir()
    test_scenario = scenarios / "test_scenario"
    test_scenario.mkdir()
    common = test_scenario / "Common"
    common.mkdir()
    parameters = common / "Parameters"
    parameters.mkdir()
    modelgeography = parameters/"ModelGeography.csv"
    modelgeography.write_text("INDIA")
    assert os.path.abspath(filemanager.find_filepath(
        "ModelGeography")) == str(modelgeography.absolute())


@pytest.fixture()
def output():
    output_path = "WESQWQK_output"
    yield output_path
    shutil.rmtree(output_path)


def test_get_output_path(clear_filemanager_cache, configmanager, output):
    path = os.path.join("Test Instance",
                        "Scenarios",
                        config.get_config_value("scenario"),
                        "Supply",
                        "Output")
    assert filemanager.get_output_path("Supply") == path

    config.set_config("output", output)
    path = os.path.join(output,
                        config.get_config_value("scenario"),
                        "Supply",
                        "Output")
    assert filemanager.get_output_path("Supply") == path
