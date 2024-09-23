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
import pytest
import pandas as pd
from rumi.io import loaders
from rumi.io import filemanager
from rumi.io import config
from rumi.io.test_config import configmanager
from rumi.io.test_filemanager import clear_filemanager_cache
import pandas as pd


def function_for_loading():
    return pd.DataFrame({"StartYear": [2021], "EndYear": [2025]})


@pytest.fixture()
def specs(clear_filemanager_cache, configmanager):
    loaders.get_parameter.cache_clear()

    s = filemanager.common_specs()
    s['TestOptional'] = dict(s['ModelPeriod'])
    s['TestOptional']['loader'] = "rumi.io.test_loaders.function_for_loading"
    yield s


def test_circular():
    assert loaders.circular([0, 1, 2, 3], range(4))
    assert not loaders.circular([1, 0, 2, 3], range(4))
    assert loaders.circular([2, 3, 4, 5, 0, 1], range(6))
    assert loaders.circular([4, 6, 8, 1, 3], range(9))


def test_x_in_y():
    assert loaders.x_in_y(x=[1, 4], y=[1, 2, 3, 4, 5])
    assert not loaders.x_in_y(x=[0, 1, 2], y=[1, 2, 3, 4, 5])
    assert not loaders.x_in_y(x=[0, 1, 2], y=[3, 4, 5, 6])


def test_valid_date():
    assert loaders.valid_date([12], [31])
    assert not loaders.valid_date([2], [30])


def test_unique():
    assert loaders.unique([1, 2, 3, 4])
    assert not loaders.unique([1, 1, 1, 2])


def test_column():
    assert loaders.column([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 0) == [1, 4, 7]
    assert loaders.column([[1], [1, 2], [1, 2, 3]], 1) == ["", 2, 2]


def test_transpose():
    assert loaders.transpose([[1, 2, 3]]) == [[1], [2], [3]]


def test_cancat():
    a = [[10, 20, 30, 40],
         [21, 22, 23, 24]]

    assert loaders.concat(*a) == [10, 20, 30, 40, 21, 22, 23, 24]


def test_validate_params(specs):
    print(config.get_config_value("model_instance_path"), "X"*10)
    print(loaders.validate_params("Common"))


def assert_model_years(ModelPeriod):
    print(ModelPeriod, "Y"*10)
    return ModelPeriod.StartYear[0] == 2021 and ModelPeriod.EndYear[0] == 2031


def test_load_param(specs):
    assert loaders.load_param("ModelGeography")
    with pytest.raises(filemanager.FolderStructureError) as e:
        assert loaders.load_param("TestOptional") is None

    ModelPeriod = loaders.load_param('ModelPeriod')
    assert assert_model_years(ModelPeriod)


def test_validate_each_item(specs):
    assert loaders.validate_each_item("ModelGeography",
                                      specs['ModelGeography'],
                                      [['INDIA']])
    ModelPeriod = pd.DataFrame({"StartYear": [2019],
                                "EndYear": [2030]})
    assert loaders.validate_each_item("ModelPeriod",
                                      specs['ModelPeriod'],
                                      ModelPeriod)
    ModelPeriod = pd.DataFrame({"StartYear": [1996],
                                "EndYear": [2030]})
    assert not loaders.validate_each_item("ModelPeriod",
                                          specs['ModelPeriod'],
                                          ModelPeriod)
    ModelPeriod = pd.DataFrame({"StartYear": [2019],
                                "EndYear": [3000]})
    assert not loaders.validate_each_item("ModelPeriod",
                                          specs['ModelPeriod'],
                                          ModelPeriod)


def test_validate_param(specs):
    module = "rumi.io.common"
    assert loaders.validate_param("ModelGeography",
                                  specs['ModelGeography'],
                                  [['INDIA']], module)
    ModelPeriod = pd.DataFrame({"StartYear": [2019],
                                "EndYear": [2030]})
    assert loaders.validate_param("ModelPeriod",
                                  specs['ModelPeriod'],
                                  ModelPeriod, module)
    ModelPeriod = pd.DataFrame({"StartYear": [1996],
                                "EndYear": [2030]})
    assert not loaders.validate_param("ModelPeriod",
                                      specs['ModelPeriod'],
                                      ModelPeriod, module)
    ModelPeriod = pd.DataFrame({"StartYear": [2019],
                                "EndYear": [3000]})
    assert not loaders.validate_param("ModelPeriod",
                                      specs['ModelPeriod'],
                                      ModelPeriod, module)


def test_load_module():
    env = {}
    loaders.load_module("rumi.io.common", env)
    assert "drop_columns" in env
    assert isinstance(env['drop_columns'], type(test_cancat))


def test_create_namespace():
    namespace = {
        "x2": "[v*2 for v in x]",
        "y": "concat([1, 1, 1], [2, 2, 2])"
    }
    env = {
        "x": [1, 1, 1],
        "concat": loaders.concat
    }
    loaders.load_namespace(namespace, env)
    assert "x2" in env
    assert "y" in env
    assert env['x2'] == [2, 2, 2]
    assert env['y'] == [1, 1, 1, 2, 2, 2]


def test_global_validation(specs, monkeypatch):
    namespace = {
        "x2": "[v*2 for v in x]",
        "y": "concat([1, 1, 1], [2, 2, 2])"
    }
    validations = [{"code": "x_in_y(x=x2, y=y)",
                    "message": "test"}]
    env = {
        "x": [1, 1, 1],
        "concat": loaders.concat,
        "x_in_y": loaders.x_in_y
    }
    global_validation = {"namespace": namespace,
                         "validation": validations,
                         "module": "rumi.io.common"}
    assert loaders.global_validation(env, global_validation)

    GDP = pd.DataFrame({"Year": [2019, 2019, 2020, 2020],
                        "GDP": [234324324324, 43243242344]*2,
                        "ModelGeography": ["INDIA"]*4,
                        "SubGeography1": ["NORTH", "SOUTH"]*2})

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

    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    env['GDP'] = GDP
    validations = [
        {"code": "valid_geography(GDP)",
         "message": "Testing global validation"}
    ]
    global_validation['validation'] = validations
    assert loaders.global_validation(env, global_validation)

    global_validation['validation'] = [{"code": "x_in_y(x=[0,0,0], y=y)",
                                        "message": "test"}]
    assert not loaders.global_validation(env, global_validation)


def test_get_parameter(specs):

    ModelPeriod = loaders.get_parameter("ModelPeriod")
    assert_model_years(ModelPeriod)

    assert isinstance(loaders.get_parameter("ModelGeography"), str)
    assert isinstance(loaders.get_parameter("SubGeography1"), list)
    assert isinstance(loaders.get_parameter("SubGeography2"), dict)
    assert loaders.get_parameter("TestOptional").StartYear.iloc[0] == 2021
    assert loaders.get_parameter("TestOptional").EndYear.iloc[0] == 2025


def test_filter_param(monkeypatch):
    def get_parameter(param_name):
        if param_name == "ModelPeriod":
            return pd.DataFrame({"StartYear": [2021],
                                 "EndYear": [2022]})

    def get_specs(param_name):
        return {'filetype': 'csv',
                'axis': 'row',
                'columns': {'Year':
                            {'type': 'int',
                             'min': 2000,
                             'max': 2100},
                            'GDP': {'type': 'float',
                                    'min': 0},
                            'ModelGeography': {'type': 'str'},
                            'SubGeography1': {'type': 'str',
                                              'optional': True},
                            'SubGeography2': {'type': 'str',
                                              'optional': True},
                            'SubGeography3': {'type': 'str',
                                              'optional': True}},
                'dependencies': ['ModelPeriod'],
                'filterqueries': ["Year >= {ModelPeriod.StartYear.iloc[0]}",
                                  "Year <= {ModelPeriod.EndYear.iloc[0]}"]}

    monkeypatch.setattr(filemanager, "get_specs", get_specs)
    monkeypatch.setattr(loaders, "get_parameter", get_parameter)
    data = pd.DataFrame({"Year": [i+2019 for i in range(10)],
                         'GDP': [(i+1)*10.0 for i in range(10)],
                         'ModelGeography': ['INDIA']*10})
    d = loaders.filter_param("GDP", data)
    assert len(d) == 2
    assert d.Year.values == pytest.approx([2021, 2022])


def test_reformat_headerless():
    d = loaders.reformat_headerless("test", {"map": True}, [["a", "b", "c", "", ""],
                                                            ["b", "WS", "WA", ""],
                                                            ["c", "AS", "DE"]])
    assert d == {"a": ["b", "c"],
                 "b": ["WS", "WA"],
                 "c": ["AS", "DE"]}

    d = loaders.reformat_headerless("test", {"list": True}, [
                                    ["a", "b", "c", "", ""]])
    assert d == ["a", "b", "c"]

    assert loaders.reformat_headerless("test",
                                       {},
                                       [["A", "B", "C", ""],
                                        ["X", "Y"]]) == [["A", "B", "C"],
                                                         ["X", "Y"]]


def test_strip_trailing():
    assert loaders.strip_trailing(["","a","b","c"]) == ["", "a", "b", "c"]
    assert loaders.strip_trailing(["","a","b",""]) == ["", "a", "b"]
    assert loaders.strip_trailing(["","a","b","","",""]) == ["", "a", "b"]
    assert loaders.strip_trailing(["","","",""]) == []
