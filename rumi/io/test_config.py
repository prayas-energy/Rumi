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
import tempfile
from rumi.io import config
import pytest
import yaml
import multiprocessing
import pkg_resources


def create_model_instance(model_instance_path):
    ModelPeriod = """StartYear,EndYear
2021,2031"""
    ModelGeography = """INDIA"""
    SubGeography1 = """ER,WR,NR,SR,NER"""
    SubGeography2 = """ER,BR,JH,OD,WB
WR,CG,GJ,MP,MH,GA,UT
NR,DL,HR,HP,JK,PB,RJ,UP,UK
SR,AP,KA,KL,TN,TS
NER,AS,NE"""
    GDP = """Year,ModelGeography,SubGeography1,GDP
2020,INDIA,ER,23738390.47
2021,INDIA,ER,21574529.28
2022,INDIA,ER,23326940.94
2023,INDIA,ER,24628388.83
2024,INDIA,ER,25972207.66
2025,INDIA,ER,27355270.59
2026,INDIA,ER,28791258.46
2027,INDIA,ER,30275178.16
2028,INDIA,ER,31833078.64
2029,INDIA,ER,33468499.91
2030,INDIA,ER,35185142.66
2031,INDIA,ER,36986875.11
2020,INDIA,NER,5384212.504
2021,INDIA,NER,4957274.804
2022,INDIA,NER,5431071.616
2023,INDIA,NER,5811467.238
2024,INDIA,NER,6212657.498
2025,INDIA,NER,6634769.09
2026,INDIA,NER,7082067.686
2027,INDIA,NER,7554401.887
2028,INDIA,NER,8059467.036
2029,INDIA,NER,8599628.342
2030,INDIA,NER,9177431
2031,INDIA,NER,9795614.735
2020,INDIA,NR,52308265.32
2021,INDIA,NR,48035436.21
2022,INDIA,NR,52479339.19
2023,INDIA,NR,55986787.3
2024,INDIA,NR,59660409.39
2025,INDIA,NR,63497270.05
2026,INDIA,NR,67533652.8
2027,INDIA,NR,71763026.69
2028,INDIA,NR,76252842.52
2029,INDIA,NR,81018821.35
2030,INDIA,NR,86077610.37
2031,INDIA,NR,91446835.84
2020,INDIA,SR,57575301.12
2021,INDIA,SR,53151524.57
2022,INDIA,SR,58374303.23
2023,INDIA,SR,62602178.89
2024,INDIA,SR,67058182.18
2025,INDIA,SR,71741962.32
2026,INDIA,SR,76697640.92
2027,INDIA,SR,81921314.32
2028,INDIA,SR,87493813.65
2029,INDIA,SR,93437893.24
2030,INDIA,SR,99777750.38
2031,INDIA,SR,106539113.7
2020,INDIA,WR,57496667.88
2021,INDIA,WR,53133354.07
2022,INDIA,WR,58421415.26
2023,INDIA,WR,62732541.01
2024,INDIA,WR,67291911.45
2025,INDIA,WR,72101941.77
2026,INDIA,WR,77210018.19
2027,INDIA,WR,82615618.59
2028,INDIA,WR,88404053.36
2029,INDIA,WR,94602746.76
2030,INDIA,WR,101241105.6
2031,INDIA,WR,108350663.1"""

    data = {
        "ModelPeriod": ModelPeriod,
        "ModelGeography": ModelGeography,
        "SubGeography1": SubGeography1,
        "SubGeography2": SubGeography2,
        "GDP": GDP
    }

    commonpath = os.path.join(model_instance_path,
                              "Global Data",
                              "Common",
                              "Parameters")
    if not os.path.exists(commonpath):
        os.makedirs(commonpath)

    for item, text in data.items():
        with open(os.path.join(commonpath,
                               ".".join([item, "csv"])), "w") as f:
            f.write(text)


@pytest.fixture()
def configmanager(monkeypatch):
    common = """
ModelPeriod:
  filetype: csv
  axis : column
  columns:
    StartYear:
      type: int
      min: 2000
      max: 2100
    EndYear:
      type: int
      min: 2000
      max: 2100
  validation:
    - code: all(EndYear > StartYear)
      message: "Change this message"

# nation
ModelGeography:
  filetype: csv
  axis: row
  noheader: True

# regions
SubGeography1:
  filetype: csv
  axis: row
  noheader: True
  optional: True

# states
SubGeography2:
  filetype: csv
  axis: row
  noheader: True
  optional: True
  map: True

# Districts
SubGeography3:
  filetype: csv
  axis: row
  noheader: True
  optional: True
  map: True


GDP:
  filetype: csv
  axis: row
  columns:
    Year:
      type: int
      min: 2000
      max: 2100
    GDP:
      type: float
      min: 0
    ModelGeography:
      type: str
    SubGeography1:
      type: str
      optional: True
    SubGeography2:
      type: str
      optional: True
    SubGeography3:
      type: str
      optional: True

    
global_validation:
  module: rumi.io.common
  validation:
    - code: unique(SubGeography1)
      message: "Change this message"
    - code: unique(SubGeography2.keys())
      message: "some message" """

    folder_structure = """
Config:
Documentation:
Global Data:
  Common:
    Parameters: !include Common.yml
    Source:
Scenarios:"""

    testdir = "test_config_"
    commonfile = os.path.join(testdir, "Common.yml")
    configfile = os.path.join(testdir, "Config.yml")
    fs = os.path.join(testdir, "folder_structure.yml")
    model_instance_path = "Test Instance"

    if not os.path.exists(testdir):
        os.mkdir(testdir)

    with open(commonfile, "w") as f:
        f.write(common)

    with open(configfile, "w") as f:
        f.write("""config1: value1
config2: value2""")

    with open(fs, "w") as f:
        f.write(folder_structure)

    create_model_instance("Test Instance")

    monkeypatch.setattr(pkg_resources,
                        'resource_filename',
                        lambda x, y: f"{testdir}")

    try:
        config.initialize_config(model_instance_path, "Scenario1")
    except config.ConfigurationError as ec:
        print(config.get_config_value("model_instance_path"))
        print(config.get_config_value("yaml_location"))
        print(ec)

    yield configfile

    # os.unlink(commonfile)
    # os.unlink(configfile)
    # os.unlink(fs)
    # os.removedirs(testdir)


def test_initialize_config(configmanager):
    get_config = config.get_config_value
    assert get_config("config1") == "value1"
    assert get_config("config2") == "value2"
    assert get_config("model_instance_path") == "Test Instance"
    assert get_config("scenario") == "Scenario1"
    assert get_config("yaml_location") == "test_config_"
    assert get_config("undefined") == None


def test_singleton(configmanager):
    assert id(config.ConfigManager()) == id(config.ConfigManager())


def test_multiprocess(configmanager):
    with multiprocessing.Pool(4) as p:
        c = p.map(config.get_config_value, ['config1',
                                            'config2',
                                            'model_instance_path',
                                            'scenario',
                                            'yaml_location',
                                            'undefined'])
        assert c == ['value1', 'value2', 'Test Instance',
                     'Scenario1', 'test_config_', None]


def test_set_config(configmanager):
    config.set_config("x", "y")
    assert config.get_config_value("x") == "y"
    assert config.get_config_value("config1") == "value1"


def test_immutability(configmanager):
    config.set_config("unique", "x")
    with pytest.raises(config.ConfigurationError) as ce:
        config.set_config("unique", "y")
    assert "Configuration values once set can not be modified." in str(
        ce.value)
