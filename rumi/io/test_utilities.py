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
    elif name == "DS_Cons1_Map":
        return {'DS1': ['SUBGEOGRAPHY2', 'YEAR', 'URBAN', 'RURAL'],
                'DS2': ['SUBGEOGRAPHY2', 'YEAR', 'URBAN', 'RURAL']}


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

    basedf = utilities.base_dataframe(
        ['Year', 'ModelGeography', 'SubGeography1', 'ConsumerType1'],
        demand_sector='DS1',
        colname='EnergyCarrier',
        val='COAL').reset_index()

    assert utilities.unique_across(basedf, ['EnergyCarrier'])
    assert not utilities.unique_across(pd.concat([basedf]*2),
                                       ['EnergyCarrier'])


def test_base_dataframe(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)

    # if order of geocols, conscols, timecols was different from those
    # defined in constant it was giving wrong base dataframe
    df1 = utilities.base_dataframe_all(geocols=['SubGeography1', 'ModelGeography'],
                                       timecols=['Year'])
    df2 = utilities.base_dataframe_all(geocols=['ModelGeography', 'SubGeography1'],
                                       timecols=['Year'])

    assert df2.equals(df1)


def test_compute_product_geo(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    with pytest.raises(utilities.InValidColumnsError):
        utilities.compute_product_geo(['SubGeography1'])


def test_compute_product_consumers(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', lambda _: {
                        "DS": ["x", "y", "RURAL", "URBAN"]})
    with pytest.raises(utilities.InValidColumnsError):
        utilities.compute_product_consumers(['ConsumerType2'], "DS")

    monkeypatch.setattr(loaders, 'get_parameter', lambda _: None)
    with pytest.raises(TypeError):
        utilities.compute_product_consumers(['ConsumerType1'], "DS")


def test_compute_product_time(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    with pytest.raises(utilities.InValidColumnsError):
        utilities.compute_product_time(['Year', 'DaySlice'])


def test_groupby(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    v = 0.3
    data = utilities.base_dataframe_time(constant.TIME_SLICES,
                                         colname='VALUE',
                                         val=v).reset_index()

    # =================Year=============================
    df = utilities.groupby(data, ['Year'], 'VALUE').reset_index()

    assert set(df.columns) == set(
        constant.TIME_COLUMNS['YEAR'] + ['VALUE'])

    assert len(df) == 2
    # number of dayslices
    assert sum(df['VALUE'].values) == pytest.approx(
        2*sum([v*6*days for s, days in common.seasons_size().items()]))

    # =================Season=============================
    df = utilities.groupby(data, ['Year', 'Season'], 'VALUE').reset_index()

    assert set(df.columns) == set(
        constant.TIME_COLUMNS['SEASON'] + ['VALUE'])

    assert len(df) == 2*5
    # number of dayslices
    assert sum(df['VALUE'].values) == pytest.approx(
        2*sum([v*6*days for s, days in common.seasons_size().items()]))

    # =================DayType=============================
    df = utilities.groupby(
        data, constant.TIME_COLUMNS['DAYTYPE'], 'VALUE').reset_index()
    assert set(df.columns) == set(
        constant.TIME_COLUMNS['DAYTYPE'] + ['VALUE'])

    assert len(df) == 2*5
    assert df.iloc[0]['VALUE'] == pytest.approx(v*6)  # number of dayslices

    # =====================================================
    v = 0.3
    data = utilities.base_dataframe_time(constant.TIME_COLUMNS['DAYTYPE'],
                                         colname='VALUE',
                                         val=v).reset_index()
    df = utilities.groupby(
        data, constant.TIME_COLUMNS['SEASON'], 'VALUE').reset_index()

    assert len(df) == 2*5
    assert sum(df['VALUE'].values) == pytest.approx(
        2*sum([v*days for s, days in common.seasons_size().items()]))

    df = utilities.groupby(
        data, constant.TIME_COLUMNS['YEAR'], 'VALUE').reset_index()

    assert len(df) == 2
    assert sum(df['VALUE'].values) == pytest.approx(
        2*sum([v*days for s, days in common.seasons_size().items()]))

    df2 = utilities.groupby(
        df, constant.TIME_COLUMNS['YEAR'], 'VALUE').reset_index()

    assert df2.equals(df)

    # =============no time cols but geography cols given======
    data = utilities.base_dataframe_all(timecols=['Year'],
                                        geocols=['ModelGeography',
                                                 'SubGeography1'],
                                        colname='VALUE',
                                        val=v).reset_index()
    df = utilities.groupby(
        data, ['Year', 'ModelGeography'], 'VALUE').reset_index()
    assert len(df) == len(data) / 5  # SubGeography1 values
    assert len(df) == len(utilities.get_years())
    assert df.VALUE.sum() == pytest.approx(v*5*len(utilities.get_years()))


def test_check_source_dest(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)

    def test_grouping(function='groupby'):
        v = 0.3
        for s, gran in enumerate(constant.TIME_SLICES):
            source = utilities.base_dataframe_time(constant.TIME_COLUMNS[gran.upper()],
                                                   colname='VALUE',
                                                   val=v).reset_index()
            for d in range(s+1):
                destcols = list(constant.TIME_SLICES[:d+1])
                balacing_time = constant.TIME_SLICES[d].upper()
                # print(gran, destcols[-1])
                if function == "groupby_time":
                    dest = utilities.groupby_time(source,
                                                  [],
                                                  balacing_time,
                                                  'VALUE')
                else:
                    dest = utilities.groupby(source,
                                             destcols,
                                             'VALUE').reset_index()
                # print(source, dest)
                if (gran in ['DayType', 'DaySlice']) and\
                   destcols[-1] in ['Season', 'Year']:
                    if gran == 'DayType':
                        s = 1
                    else:
                        s = 6
                    assert sum(dest['VALUE'].values) == pytest.approx(
                        2*s*sum([v*days for s, days in common.seasons_size().items()]))
                elif gran == 'Season' and destcols[-1] == 'Year':
                    assert len(dest) == len(utilities.get_years())
                    assert dest['VALUE'].values == pytest.approx(
                        [v*len(common.seasons_size()) for y in utilities.get_years()])
                elif gran == 'DaySlice' and destcols[-1] == 'DayType':
                    assert len(dest) == len(source)/6  # number of day slices
                    assert dest['VALUE'].values == pytest.approx(
                        [v*6]*len(common.seasons_size()) *
                        len(utilities.get_years()))
                else:
                    # for same source and dest granularities
                    assert len(dest) == len(source)
                    assert dest.equals(source)

    test_grouping()
    test_grouping("groupby_time")


def test_subset_multi(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    data = utilities.base_dataframe_all(geocols=['ModelGeography'],
                                        timecols=['Year'],
                                        val=0,
                                        extracols_df=pd.DataFrame({'ST': ["A", "B"]})).reset_index()

    s = utilities.subset_multi(data, ['ST', 'Year'], ["A", 2021])
    d = data.query("ST=='A' & Year==2021")
    assert set(s.itertuples(index=False, name=None)) == set(
        d.itertuples(index=False, name=None))


def test_check_CGT_validity(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    data = utilities.base_dataframe_of_granularity(CGRAN='CONSUMERTYPE1',
                                                   GGRAN='SUBGEOGRAPHY1',
                                                   TGRAN='DAYSLICE',
                                                   demand_sector='DS1',
                                                   val=1.0,
                                                   extracols_df=pd.DataFrame({"Entity": ["A", "B"]})).reset_index()
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', 'Entity', "C", demand_sector='DS1', exact=True)
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', 'Entity', "G", demand_sector='DS1', exact=True)
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', 'Entity', "T", demand_sector='DS1', exact=True)

    data = utilities.base_dataframe_of_granularity(CGRAN='CONSUMERTYPE1',
                                                   GGRAN='SUBGEOGRAPHY1',
                                                   TGRAN='DAYSLICE',
                                                   demand_sector='DS1',
                                                   val=1.0,
                                                   extracols_df=pd.DataFrame({"Entity": ["A"]})).reset_index()
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', 'Entity', "C", demand_sector='DS1', exact=True)
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', 'Entity', "G", demand_sector='DS1', exact=True)
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', 'Entity', "T", demand_sector='DS1', exact=True)

    data = utilities.base_dataframe_of_granularity(CGRAN='CONSUMERTYPE1',
                                                   GGRAN='SUBGEOGRAPHY1',
                                                   TGRAN='DAYSLICE',
                                                   demand_sector='DS1',
                                                   val=1.0).reset_index()
    data['Entity'] = None
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', 'Entity', "C", demand_sector='DS1', exact=True)
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', 'Entity', "G", demand_sector='DS1', exact=True)
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', 'Entity', "T", demand_sector='DS1', exact=True)

    data = utilities.base_dataframe_of_granularity(CGRAN='CONSUMERTYPE1',
                                                   GGRAN='SUBGEOGRAPHY1',
                                                   TGRAN='DAYSLICE',
                                                   demand_sector='DS1',
                                                   val=1.0).reset_index()
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', [], "C", demand_sector='DS1', exact=True)
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', [], "G", demand_sector='DS1', exact=True)
    assert utilities.check_CGT_validity(
        data, 'TESTPARAM', [], "T", demand_sector='DS1', exact=True)
    assert not utilities.check_CGT_validity(data.query("Year != 2021"),
                                            'TESTPARAM',
                                            [],
                                            "T",
                                            demand_sector='DS1', exact=True)
    assert not utilities.check_CGT_validity(data.query("SubGeography1 != 'NR'"),
                                            'TESTPARAM',
                                            [],
                                            "G",
                                            demand_sector='DS1', exact=True)
    assert not utilities.check_CGT_validity(data.query("ConsumerType1 != 'RURAL'"),
                                            'TESTPARAM',
                                            [],
                                            "C",
                                            demand_sector='DS1', exact=True)


def test_override_dataframe_with_check(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    data = utilities.base_dataframe_all(geocols=['ModelGeography'],
                                        timecols=['Year'],
                                        val=0,
                                        extracols_df=pd.DataFrame({"ConsumerType1": ["X", "Y"]})).reset_index()

    base = utilities.base_dataframe_all(geocols=['ModelGeography'],
                                        conscols=['ConsumerType1'],
                                        timecols=['Year'],
                                        demand_sector="DS1",
                                        val=0,
                                        extracols_df=None).reset_index()

    with pytest.raises(utilities.InvalidCGTDataError):
        indexcols = ['ConsumerType1', 'ModelGeography', 'Year']
        utilities.override_dataframe_with_check(
            base, data, indexcols, "TESTPARAM", "DS1", "ES1")

    data = utilities.base_dataframe_all(geocols=['ModelGeography'],
                                        timecols=['Year'],
                                        val=2.0,
                                        extracols_df=None).reset_index()

    base = utilities.base_dataframe_all(geocols=['ModelGeography'],
                                        timecols=['Year'],
                                        demand_sector="DS1",
                                        val=0,
                                        extracols_df=None).reset_index()
    indexcols = ['ModelGeography', 'Year']
    d = utilities.override_dataframe_with_check(
        base, data, indexcols, "TESTPARAM")
    assert d.dummy.sum() == pytest.approx(len(base) * 2.0)
    assert utilities.get_set(
        d[indexcols]) == utilities.get_set(base[indexcols])


def test_check_duplicates(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    df = utilities.base_dataframe_all(geocols=['ModelGeography', 'SubGeography1'],
                                      timecols=['Year'],
                                      conscols=['ConsumerType1'],
                                      demand_sector="DS1",
                                      val=0,
                                      extracols_df=None).reset_index()

    indexcols = ['ModelGeography', 'SubGeography1', 'Year', 'ConsumerType1']
    assert not utilities.check_duplicates(df, indexcols, "TESTPARAM")
    df2 = pd.concat([df, df])
    assert utilities.check_duplicates(df2, indexcols, "TESTPARAM")


def test_compute_intervals():
    seasons = utilities.make_dataframe("""Season,StartMonth,StartDate
SUMMER,4,1
MONSOON,6,1
AUTUMN,9,1
WINTER,11,1
SPRING,2,1""")
    assert utilities.compute_intervals(seasons) == {
        'SUMMER': 61, 'MONSOON': 92, 'AUTUMN': 61, 'WINTER': 92, 'SPRING': 59}
    seasons = utilities.make_dataframe("""Season,StartMonth,StartDate
SUMMER,4,1
REST,10,1""")
    assert utilities.compute_intervals(seasons) == {
        'SUMMER': 183, 'REST': 182}
    seasons = utilities.make_dataframe("""Season,StartMonth,StartDate
SUMMER,4,1
REST,3,31""")
    assert utilities.compute_intervals(seasons) == {
        'SUMMER': 364, 'REST': 1}
    seasons = utilities.make_dataframe("""Season,StartMonth,StartDate
SUMMER,4,1
REST,4,1
XXX,5,1""")
    assert utilities.compute_intervals(seasons) == {
        'SUMMER': 0, 'REST': 30, 'XXX': 335}
