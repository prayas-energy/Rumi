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
import pandas as pd
import numpy as np
from rumi.io import supply
from rumi.io import loaders
from rumi.io import constant
from rumi.io import common
import pytest
from rumi.io import utilities


def test_check_granularity():
    data = """EnergyService,EnergyCarrier,ConsumerType1,ModelGeography,SubGeography1,SubGeography2,Year,Elasticity
AGRI_ALL,LPG,ALL,INDIA,,,2021,4.50871871278
AGRI_ALL,LPG,ALL,INDIA,,,2022,4.05784684150
AGRI_ALL,LPG,ALL,INDIA,,,2023,3.65206215735
AGRI_ALL,LPG,ALL,INDIA,,,2024,3.28685594162
AGRI_ALL,LPG,ALL,INDIA,,,2025,2.95817034746
AGRI_ALL,LPG,ALL,INDIA,,,2026,2.66235331271
AGRI_ALL,LPG,ALL,INDIA,,,2027,2.39611798144
AGRI_ALL,LPG,ALL,INDIA,,,2028,2.15650618330
AGRI_ALL,LPG,ALL,INDIA,,,2029,1.94085556497
AGRI_ALL,LPG,ALL,INDIA,,,2030,1.74677000847
AGRI_ALL,LPG,ALL,INDIA,,,2031,1.57209300762
AGRI_ALL,HSD,ALL,INDIA,,,2021,0.45187797897
AGRI_ALL,HSD,ALL,INDIA,,,2022,0.44735919918
AGRI_ALL,HSD,ALL,INDIA,,,2023,0.44288560719
AGRI_ALL,HSD,ALL,INDIA,,,2024,0.43845675112
AGRI_ALL,HSD,ALL,INDIA,,,2025,0.43407218361
AGRI_ALL,HSD,ALL,INDIA,,,2026,0.42973146177
AGRI_ALL,HSD,ALL,INDIA,,,2027,0.42543414715
AGRI_ALL,HSD,ALL,INDIA,,,2028,0.42117980568
AGRI_ALL,HSD,ALL,INDIA,,,2029,0.41696800762
AGRI_ALL,HSD,ALL,INDIA,,,2030,0.41279832755
AGRI_ALL,HSD,ALL,INDIA,,,2031,0.40867034427
AGRI_ALL,PP_OTHER,ALL,INDIA,,,2021,0.82669293702
AGRI_ALL,PP_OTHER,ALL,INDIA,,,2022,0.82669293702
AGRI_ALL,PP_OTHER,ALL,INDIA,,,2023,0.82669293702
AGRI_ALL,PP_OTHER,ALL,INDIA,,,2024,0.82669293702
AGRI_ALL,PP_OTHER,ALL,INDIA,,,2025,0.82669293702
AGRI_ALL,PP_OTHER,ALL,INDIA,,,2026,0.82669293702
AGRI_ALL,PP_OTHER,ALL,INDIA,,,2027,0.82669293702
AGRI_ALL,PP_OTHER,ALL,INDIA,,,2028,0.82669293702
AGRI_ALL,PP_OTHER,ALL,INDIA,,,2029,0.82669293702
AGRI_ALL,PP_OTHER,ALL,INDIA,,,2030,0.82669293702
AGRI_ALL,PP_OTHER,ALL,INDIA,,,2031,0.82669293702
AGRI_ALL,ELECTRICITY,ALL,INDIA,NR,DL,2021,0.817161019
AGRI_ALL,ELECTRICITY,ALL,INDIA,NR,DL,2022,0.804903604
AGRI_ALL,ELECTRICITY,ALL,INDIA,ER,BR,2021,0.817161019
AGRI_ALL,ELECTRICITY,ALL,INDIA,NR,HR,2021,0.817161019
AGRI_ALL,ELECTRICITY,ALL,INDIA,ER,JH,2021,0.817161019
AGRI_ALL,ELECTRICITY,ALL,INDIA,NR,HP,2021,0.817161019
AGRI_ALL,ELECTRICITY,ALL,INDIA,ER,OD,2021,0.817161019
AGRI_ALL,ELECTRICITY,ALL,INDIA,NR,JK,2021,0.817161019"""
    granularity = """EnergyCarrier,TimeGranularity,GeographicGranularity
LPG,YEAR,MODELGEOGRAPHY
HSD,YEAR,MODELGEOGRAPHY
PP_OTHER,YEAR,MODELGEOGRAPHY
ELECTRICITY,YEAR,SUBGEOGRAPHY2
"""
    data = utilities.make_dataframe(data)
    granularity = utilities.make_dataframe(granularity)

    assert supply.check_granularity(
        data, granularity, 'EnergyCarrier', GSTAR=True, TSTAR=True)

    data = """EnergyService,EnergyCarrier,ConsumerType1,ModelGeography,SubGeography1,SubGeography2,Year,Elasticity
AGRI_ALL,LPG,ALL,INDIA,NR,,2021,4.50871871278
AGRI_ALL,LPG,ALL,INDIA,NR,,2022,4.05784684150"""
    data = utilities.make_dataframe(data)
    assert not supply.check_granularity(
        data, granularity, 'EnergyCarrier', GSTAR=True, TSTAR=True)


def test_find_EC(monkeypatch):
    def get_parameter(name, **kwargs):
        if name == 'EnergyStorTechnologies':
            d = """EnergyStorTech,StoredEC,DomOrImp,MaxChargeRate,MaxDischargeRate,StorPeriodicity
BESS_6HR,ELECTRICITY,EC_DOM,0.166666667,0.166666667,NEVER
BESS_4HR,ELECTRICITY,EC_DOM,0.25,0.25,NEVER"""

            return utilities.make_dataframe(d)

        elif name == "EnergyConvTechnologies":
            d = """EnergyConvTech,CapacityUnit,InputEC,OutputDEC,AnnualEnergyPerUnitCapacity
ET_COAL,MW,NONCOKING_COAL,ELECTRICITY,8760
ET_CCGT,MW,NATGAS,ELECTRICITY,8760
ET_OCGT,MW,NATGAS,ELECTRICITY,8760
ET_PHWR,MW,ATOMIC,ELECTRICITY,8760"""
            return utilities.make_dataframe(d)
        else:
            return None

    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    assert supply.find_EC('EnergyConvTech', "ET_COAL") == 'ELECTRICITY'
    assert supply.find_EC('EnergyStorTech', 'BESS_4HR') == 'ELECTRICITY'


def test_filter_on_geography(monkeypatch):
    def get_parameter(name, **kwargs):
        if name == "ModelGeography":
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
                                 "BalancingArea": ['MODELGEOGRAPHY', 'SUBGEOGRAPHY1'],
                                 "BalancingTime": ['YEAR']*2,
                                 "PhysicalUnit": ["tonne"]*2,
                                 "EnergyUnit": ["MJ"]*2,
                                 "DomEnergyDensity": [21526.84]*2,
                                 "ImpEnergyDensity": [24058.0]*2})
        elif name == "PhysicalDerivedCarriers":
            return pd.DataFrame({"EnergyCarrier": []})
        elif name == "NonPhysicalDerivedCarriers":
            return pd.DataFrame({"EnergyCarrier": []})

    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)

    v = 0.3
    d = utilities.base_dataframe_geography(['ModelGeography',
                                            'SubGeography1',
                                            'SubGeography2'],
                                           colname='FixedTaxOH',
                                           val=v).reset_index()

    d['EnergyCarrier'] = ['COKING_COAL']*len(d)
    d['X'] = ["X"]*len(d)
    df = supply.filter_on_geography(d, "fine", "Carriers")
    df1 = df.set_index(["ModelGeography"])
    assert df1.loc['INDIA']['FixedTaxOH'] == pytest.approx(len(d)*v)
    df.set_index(["ModelGeography", 'SubGeography1'], inplace=True)
    assert set(df.columns) == set(
        c for c in d.columns if c not in constant.GEOGRAPHIES)

    v = 0.3
    d = utilities.base_dataframe_geography(['ModelGeography'],
                                           colname='FixedTaxOH',
                                           val=v).reset_index()

    d['EnergyCarrier'] = ['COAL']*len(d)
    d['X'] = ["X"]*len(d)
    df = supply.filter_on_geography(d, 'coarse', 'Carriers')
    geocols = [c for c in df.columns if c in constant.GEOGRAPHIES]
    assert geocols == ['ModelGeography', 'SubGeography1']
    assert df['FixedTaxOH'].values == pytest.approx([v]*len(df))

    df.set_index(["ModelGeography", 'SubGeography1'], inplace=True)
    assert set(df.columns) == set(
        c for c in d.columns if c not in constant.GEOGRAPHIES)

    # ===========================================================
    # carrier with finest balancing area is given
    # but data is given coarser than  balancing area
    # this is the case of DEC_ImpConstraints. It should come out
    # as it is except that it should have the empty missing columns
    d = utilities.base_dataframe_geography(['ModelGeography'],
                                           colname='FixedTaxOH',
                                           val=v).reset_index()
    d['EnergyCarrier'] = ['COAL']*len(d)
    d['X'] = ["X"]*len(d)
    df = supply.filter_on_geography(d, 'fine', 'Carriers')
    assert len(d) == len(df)
    assert d['FixedTaxOH'].values == pytest.approx(df['FixedTaxOH'].values)
    assert 'SubGeography1' in df.columns
    assert all(df['SubGeography1'].values == [""]*len(d))


def test_filter_on_time(monkeypatch):
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
SPRING,2,1"""
                                            )
        elif name == "DayTypes":
            return utilities.make_dataframe("""DayType,Weight
ALLDAYS,1"""
                                            )

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

    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)

    v = 0.3
    d = utilities.base_dataframe_time(['Year',
                                       'Season',
                                       'DayType'],
                                      colname='FixedTaxOH',
                                      val=v).reset_index()

    d['EnergyCarrier'] = ['COKING_COAL']*len(d)
    d['X'] = ["X"]*len(d)
    df = supply.filter_on_time(d, "fine", "Carriers")
    df1 = df.set_index("Year")
    assert df1.loc[2021]['FixedTaxOH'] == pytest.approx(sum(
        common.seasons_size().values()) * v)
    df.set_index(['Year', 'Season', 'DayType'], inplace=True)
    assert set(df.columns) == set(
        c for c in d.columns if c not in constant.TIME_SLICES)

    v = 0.3
    d = utilities.base_dataframe_time(['Year'],
                                      colname='FixedTaxOH',
                                      val=v).reset_index()

    d['EnergyCarrier'] = ['COAL']*len(d)
    d['X'] = ["X"]*len(d)
    df = supply.filter_on_time(d, 'coarse', 'Carriers')
    timecols = [c for c in df.columns if c in constant.TIME_SLICES]
    assert timecols == ['Year', 'Season', 'DayType']
    assert df['FixedTaxOH'].values == pytest.approx([v]*len(df))
    assert set([c for c in df.columns if c not in timecols]) == set(
        c for c in d.columns if c not in constant.TIME_SLICES)

    # ==========================================================
    d = utilities.base_dataframe_time(['Year'],
                                      colname='FixedTaxOH',
                                      val=v).reset_index()
    d['EnergyCarrier'] = ['COKING_COAL']*len(d)
    d['X'] = ["X"]*len(d)
    df = supply.filter_on_time(d, 'fine', 'Carriers')
    timecols = [c for c in df.columns if c in constant.TIME_SLICES]
    assert timecols == ['Year', 'Season', 'DayType']
    assert df['FixedTaxOH'].values == pytest.approx([v]*len(df))
    assert set([c for c in df.columns if c not in timecols]) == set(
        c for c in d.columns if c not in constant.TIME_SLICES)

    # ===========================================================
    # carrier with finest balancing area is given
    # data is at coarser level, so it is upposed to expand.
    d = utilities.base_dataframe_time(['Year'],
                                      colname='FixedTaxOH',
                                      val=v).reset_index()

    d['ModelGeography'] = ['INDIA']*len(d)
    d['EnergyCarrier'] = ['COAL']*len(d)
    d = supply.filter_on_geography(d, 'coarse', 'Carriers')
    d = supply.filter_on_time(d, 'coarse', 'Carriers')

    assert set(d.columns) == {'Year', 'Season', 'DayType', 'ModelGeography',
                              'SubGeography1', 'SubGeography2', 'FixedTaxOH',
                              'EnergyCarrier'}
    assert d.isnull().sum().sum() == 0
    assert d['FixedTaxOH'].values == pytest.approx([v]*len(d))
    assert len(d) == 25*2*5
    # ===========================================================

    # ===========================================================
    # carrier with coarsest balancing area is given
    # data is at coarser level, so it is upposed to expand to
    # finest balancing area. it means it will have empty fields
    # for all those geographic levels which are finer than balancing area
    d = utilities.base_dataframe_time(['Year'],
                                      colname='FixedTaxOH',
                                      val=v).reset_index()

    d['ModelGeography'] = ['INDIA']*len(d)
    d['EnergyCarrier'] = ['COKING_COAL']*len(d)

    d = supply.filter_on_geography(d, 'coarse', 'Carriers')
    d = supply.filter_on_time(d, 'coarse', 'Carriers')
    assert set(d.columns) == {'Year', 'Season', 'DayType', 'ModelGeography',
                              'SubGeography1', 'SubGeography2', 'FixedTaxOH',
                              'EnergyCarrier'}
    assert d['FixedTaxOH'].values == pytest.approx([v]*len(d))
    assert (d['SubGeography1'] == "").sum() == len(d)
    assert (d['SubGeography2'] == "").sum() == len(d)
    assert len(d) == 2
    # ===========================================================

    # ===========================================================
    # carrier with coarsest balancing area is given
    # data is at finer geographic and time level, so it is supposed to group to
    # balancing area but should have columns of finest balancing area.
    # it means it will have empty fields
    # for all those geographic levels which are finer than balancing area
    d = utilities.base_dataframe_geography(['ModelGeography', 'SubGeography1'],
                                           colname='FixedTaxOH',
                                           val=v).reset_index()
    d['Year'] = [2021]*len(d)
    d1 = d.copy()
    d1['Year'] = [2022]*len(d)
    d = pd.concat([d, d1])
    ds = []
    for s in common.seasons_size():
        d1 = d.copy()
        d1['Season'] = [s]*len(d)
        ds.append(d1)
    d = pd.concat(ds)

    d['EnergyCarrier'] = ['COKING_COAL']*len(d)

    d = supply.filter_on_geography(d, 'fine', 'Carriers')

    d = supply.filter_on_time(d, 'fine', 'Carriers')

    assert set(d.columns) == {'Year', 'Season', 'DayType', 'ModelGeography',
                              'SubGeography1', 'SubGeography2', 'FixedTaxOH',
                              'EnergyCarrier'}

    assert d['FixedTaxOH'].values == pytest.approx(
        [sum([v*5 for s, days in common.seasons_size().items()])]*len(d))
    assert len(d) == 2
    # ===========================================================
    # carrier with finest balancing time is given
    # but data is given coarser than  balancing time
    # this is the case of DEC_ImpConstraints. It should come out
    # as it is except that it should have the empty missing columns
    d = utilities.base_dataframe_time(['Year'],
                                      colname='FixedTaxOH',
                                      val=v).reset_index()
    d['EnergyCarrier'] = ['COAL']*len(d)
    d['X'] = ["X"]*len(d)
    df = supply.filter_on_time(d, 'fine', 'Carriers')
    assert len(d) == len(df)
    assert d['FixedTaxOH'].values == pytest.approx(df['FixedTaxOH'].values)
    assert 'DayType' in df.columns
    assert all(df['DayType'].values == [""]*len(d))


def test_filter_empty_column():
    d = pd.DataFrame({"x": range(5),
                      "y": [""]*5,
                      "z": [""]*5,
                      "b": [""]*2 + ["B"]*3,
                      "a": [np.NAN]*5})
    fd = supply.filter_empty_columns(d, ['x', 'y', 'z', 'a'])
    assert set(fd.columns) == set(['x', 'b'])

    fd = supply.filter_empty_columns(d, ['x', 'y', 'a'])
    assert set(fd.columns) == set(['x', 'b', 'z'])

    fd = supply.filter_empty_columns(d, ['x', 'y', 'z'])
    assert set(fd.columns) == set(['x', 'a', 'b'])

    fd = supply.filter_empty_columns(d, ['x', 'y', 'z', 'b'])
    assert set(fd.columns) == set(['x', 'a', 'b'])


def test_validate_units_config(monkeypatch):
    def config_param(name):
        if name == 'EnergyUnitConversion':
            return pd.DataFrame({"units": ['a', 'b', 'c'],
                                 "a": [1.0, 2.0, 3.0],
                                 "b": [0.5, 1.0, 4.0],
                                 "c": [0.33333, 0.25, 1.0]})

    monkeypatch.setattr(loaders, "get_config_parameter", config_param)
    assert supply.validate_units_config('EnergyUnitConversion')
