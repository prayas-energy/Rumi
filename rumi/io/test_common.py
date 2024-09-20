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
from rumi.io import common
import pandas as pd


def test_drop_columns():
    a = [[10, 20, 30, 40],
         [21, 22, 23, 24],
         [31, 32, 33, 34],
         [41, 42, 43, 44]]
    assert common.drop_columns(a, 1) == [[20, 30, 40],
                                         [22, 23, 24],
                                         [32, 33, 34],
                                         [42, 43, 44]]
    assert common.drop_columns(a, 2) == [[30, 40],
                                         [23, 24],
                                         [33, 34],
                                         [43, 44]]


def test_valid_geography(monkeypatch):
    def get_parameter(name, *kwargs):
        if name == "ModelGeography":
            return 'INDIA'
        elif name == "SubGeography1":
            return ['NORTH', 'SOUTH']
        elif name == "SubGeography2":
            return {'NORTH': ['UP', 'BIHAR'],
                    'SOUTH': ['KERAL', 'KARNATAKA']}
        else:
            return {'UP': ['UP1', 'UP2'],
                    'BIHAR': ['BIHAR1', 'BIHAR2'],
                    'KERAL': ['KL1', 'KL2', 'KL3'],
                    'KARNATAKA': ['KA1', 'KA2']}
    from rumi.io import loaders

    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'ModelGeography': ['INDIA']*4,
                       'SubGeography1': ['NORTH', 'NORTH', 'SOUTH', 'SOUTH'],
                       'SubGeography2': ['UP', 'BIHAR', 'KERAL', 'KARNATAKA']})
    assert common.valid_geography(df)

    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       "Year":[2021]*4,
                       'ModelGeography': ['INDIA']*4,
                       'SubGeography1': ['NORTH', 'NORTH', 'SOUTH', 'SOUTH'],
                       'SubGeography2': ['UP', 'BIHAR', 'KERAL', 'KARNATAKA']})
    assert common.valid_geography(df)

    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       "Year": [2021]*4,
                       'ModelGeography': ['INDIA']*4,
                       'SubGeography1': ['NORTH', 'SOUTH']*2,
                       'SubGeography2': ['BIHAR', 'KERAL', 'UP', 'KARNATAKA']})
    assert common.valid_geography(df)
