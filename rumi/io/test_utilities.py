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
from rumi.io import common
from rumi.io import utilities
from rumi.io import constant
from rumi.io import loaders
import pandas as pd
import pytest
import numpy as np


def get_parameter(name, **kwargs):
    if name == "ModelPeriod":
        return pd.DataFrame({"StartYear": [2021],
                             "EndYear": [2022]})
    elif name == "Seasons":
        return utilities.make_dataframe("""Season,StartMonth,StartDate
SUMMER,4,1
MONSOON,6,1
AUTUMN,9,1
WINTER,11,1
SPRING,2,1""")
    elif name == "DayTypes":
        return utilities.make_dataframe("""DayType,Weight
ALLDAYS,1""")
    elif name == "DaySlices":
        return utilities.make_dataframe("""DaySlice,StartHour
EARLY,6
MORN,9
MID,12
AFTERNOON,15
EVENING,18
NIGHT,22"""
                                        )
    elif name == "ModelGeography":
        return "INDIA"
    elif name == "SubGeography1":
        return 'ER,WR,NR,SR,NER'.split(",")
    elif name == "SubGeography2":
        return {"ER": 'BR,JH,OD,WB'.split(","),
                "WR": 'CG,GJ,MP,MH,GA,UT'.split(","),
                "NR": 'DL,HR,HP,JK,PB,RJ,UP,UK'.split(","),
                "SR": 'AP,KA,KL,TN,TS'.split(","),
                "NER": 'AS,NE'.split(",")}

    elif name == "PhysicalPrimaryCarriers":
        return pd.DataFrame({"EnergyCarrier": ['COKING_COAL', "COAL"],
                             "BalancingArea": ['MODELGEOGRAPHY', 'SUBGEOGRAPHY2'],
                             "BalancingTime": ['YEAR', 'DAYTYPE'],
                             "PhysicalUnit": ["tonne"]*2,
                             "EnergyUnit": ["MJ"]*2,
                             "DomEnergyDensity": [21526.84]*2,
                             "ImpEnergyDensity": [24058.0]*2})
    elif name == "PhysicalDerivedCarriers":
        return pd.DataFrame({"EnergyCarrier": []})
    elif name == "NonPhysicalDerivedCarriers":
        return pd.DataFrame({"EnergyCarrier": []})


def test_group_daytype(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    v = 0.3
    data = utilities.base_dataframe_time(constant.TIME_SLICES,
                                         colname='VALUE',
                                         val=v).reset_index()

    df = utilities.group_daytype(data, [], 'VALUE')
    assert set(df.columns) == set(
        constant.TIME_COLUMNS['DAYTYPE'] + ['VALUE'])

    assert len(df) == 2*5
    assert df.iloc[0]['VALUE'] == pytest.approx(v*6)  # number of dayslices


def test_group_season(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    v = 0.3
    data = utilities.base_dataframe_time(constant.TIME_SLICES,
                                         colname='VALUE',
                                         val=v).reset_index()

    df = utilities.group_season(data, [], 'VALUE')

    assert set(df.columns) == set(
        constant.TIME_COLUMNS['SEASON'] + ['VALUE'])

    assert len(df) == 2*5
    # number of dayslices
    assert sum(df['VALUE'].values) == pytest.approx(
        2*sum([v*6*days for s, days in common.seasons_size().items()]))


def test_groupby_time(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    v = 0.3
    data = utilities.base_dataframe_time(constant.TIME_SLICES,
                                         colname='VALUE',
                                         val=v).reset_index()

    # =================Year=============================
    df = utilities.groupby_time(data, [], 'YEAR', 'VALUE')

    assert set(df.columns) == set(
        constant.TIME_COLUMNS['YEAR'] + ['VALUE'])

    assert len(df) == 2
    # number of dayslices
    assert sum(df['VALUE'].values) == pytest.approx(
        2*sum([v*6*days for s, days in common.seasons_size().items()]))

    # =================Season=============================
    df = utilities.groupby_time(data, [], 'SEASON', 'VALUE')

    assert set(df.columns) == set(
        constant.TIME_COLUMNS['SEASON'] + ['VALUE'])

    assert len(df) == 2*5
    # number of dayslices
    assert sum(df['VALUE'].values) == pytest.approx(
        2*sum([v*6*days for s, days in common.seasons_size().items()]))

    # =================DayType=============================
    df = utilities.groupby_time(data, [], 'DAYTYPE', 'VALUE')
    assert set(df.columns) == set(
        constant.TIME_COLUMNS['DAYTYPE'] + ['VALUE'])

    assert len(df) == 2*5
    assert df.iloc[0]['VALUE'] == pytest.approx(v*6)  # number of dayslices


def test_filter_empty():
    d = pd.DataFrame({"x": range(5),
                      "y": [""]*5,
                      "z": [""]*5,
                      "a": [np.NAN]*5,
                      "b": [""]*2 + ["B"]*3,
                      "c": [np.NaN]*3 + ["C"]*2})
    fd = utilities.filter_empty(d)
    assert set(fd.columns) == set(['x', 'b', 'c'])


def test_get_cols_from_dataframe():
    d = pd.DataFrame({"Year": [2],
                      'Season': ['AUTUMN'],
                      'DayType': ['ALLDAYS'],
                      'DaySlice': ['MORNING']})

    assert utilities.get_cols_from_dataframe(
        d, 'T') == list(constant.TIME_SLICES)
    assert utilities.get_cols_from_dataframe(d, 'G') == []

    d = pd.DataFrame({"ModelGeography": ['INDIA'],
                      'SubGeography1': ['ER'],
                      'SubGeography2': ['MS'],
                      'SubGeography3': ['PU']})
    assert utilities.get_cols_from_dataframe(
        d, 'G') == list(constant.GEOGRAPHIES)

    assert utilities.get_cols_from_dataframe(d, 'T') == []


def test_unique_across(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    df = pd.DataFrame({"a": [1, 2]*5,
                       "b": range(20, 30),
                       "c": list("abcedfghji")}
                      )
    assert not utilities.unique_across(df, ['a'])
    assert utilities.unique_across(df, ['a', 'b'])
    assert utilities.unique_across(df, ['a', 'b', 'c'])

    basedf = utilities.base_dataframe_geography(
        ['ModelGeography', 'SubGeography1'],
        colname='EnergyCarrier',
        val='COAL').reset_index()

    assert utilities.unique_across(basedf, ['EnergyCarrier'])
    assert not utilities.unique_across(pd.concat([basedf]*2),
                                       ['EnergyCarrier'])
