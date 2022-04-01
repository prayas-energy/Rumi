import os
import pytest
import shutil
from rumi.io import filemanager, config
from rumi.io.test_config import configmanager


def test_find_filepath(configmanager):
    assert filemanager.find_filepath(
        "ModelPeriod") == "Test Instance/Global Data/Common/Parameters/ModelPeriod.csv"

    global_path = os.path.join(filemanager.find_global_location("ModelPeriod"),
                               filemanager.filename("ModelPeriod"))
    instance_path = config.get_config_value("model_instance_path")
    source_path = os.path.join(instance_path, global_path)
    dest_path = global_path.replace("Global Data",
                                    filemanager.scenario_location())
    basedir = os.path.dirname(dest_path)
    os.makedirs(basedir, exist_ok=True)
    shutil.copyfile(source_path, dest_path)

    assert filemanager.find_filepath(
        "ModelPeriod") == "Test Instance/Scenarios/Scenario1/Common/Parameters/ModelPeriod.csv"

    os.unlink(dest_path)

    assert filemanager.find_filepath(
        "ModelPeriod") == "Test Instance/Global Data/Common/Parameters/ModelPeriod.csv"


@pytest.fixture()
def output():
    output_path = "WESQWQK_output"
    yield output_path
    shutil.rmtree(output_path)


def test_get_output_path(configmanager, output):
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
