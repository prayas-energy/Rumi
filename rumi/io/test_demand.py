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
from rumi.io import demand
from rumi.io import loaders
import pandas as pd


def get_parameter(param_name):
    if param_name == "DS_ES_ST_Map":
        return [["DS1", "ES1", "ST1", "ST2", "ST3", "ST4"]]
    elif param_name == "ST_Info":
        return pd.DataFrame({'ServiceTech': ['ST1', 'ST2', 'ST3', 'ST4'],
                             'EnergyCarrier': ['EC', 'EC', 'EC', 'EC1'],
                             'EnergyServiceUnit': ["Hours"]*4,
                             'NumEfficiencyLevels': [1]*4})


def test_get_corresponding_sts(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)

    assert set(demand.get_corresponding_sts(
        "DS1", "ES1", "ST2")) == {'ST1', 'ST2', 'ST3'}
