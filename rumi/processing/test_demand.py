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
from rumi.processing import demand
from rumi.io import loaders
from rumi.io import demand as demandio

import pytest
from rumi.io import utilities as ioutils
from rumi.io import filemanager
import os


def get_GDP1(dummy):
    index = pd.MultiIndex.from_product([['INDIA'],
                                        ['ER', 'WR'],
                                        [11, 12, 13, 14, 15]],
                                       names=['ModelGeography',
                                              'SubGeography1',
                                              'Year'])
    return pd.DataFrame({'GDP': [1, 2, 4, 8, 4]*2}, index=index).reset_index()


def get_GDP2(dummy):
    index = pd.MultiIndex.from_product([['INDIA'],
                                        ['ER'],
                                        ['UT', 'BP'],
                                        [11, 12, 13, 14, 15]],
                                       names=['ModelGeography',
                                              'SubGeography1',
                                              'SubGeography2',
                                              'Year'])
    return pd.DataFrame({'GDP': [1, 2, 4, 8, 4]*2}, index=index).reset_index()


def test_clean_output(tmp_path, monkeypatch):
    outputfolder = tmp_path / "Output"
    outputfolder.mkdir()
    monkeypatch.setattr(filemanager, "get_output_path",
                        lambda x: str(outputfolder))

    demand.clean_output()

    def create_file(folder, filename):
        f = folder / filename
        f.write_text(",".join(list("abcd")))

    for item in "ABCDEF":
        create_file(outputfolder, f"{item}.csv")

    create_file(outputfolder, "something.log")

    with monkeypatch.context() as m:
        m.setattr('builtins.input', lambda prompt: "y")
        demand.clean_output()

    with monkeypatch.context() as m:
        m.setattr('builtins.input', input)
        demand.clean_output()

    for item in "ABCDEF":
        create_file(outputfolder, f"{item}.csv")

    with pytest.raises(OSError):
        demand.clean_output()

    assert os.listdir(outputfolder)  # it will have log files


def test_compute_gdp_rate(monkeypatch):

    monkeypatch.setattr(loaders, 'get_parameter', get_GDP1)
    GDP = loaders.get_parameter('GDP')
    rate = demand.compute_gdp_rate(GDP)

    rate = rate.pivot(index=['ModelGeography', 'SubGeography1'],
                      columns=['Year'],
                      values='GDP_RATE')
    assert all(rate.loc[('INDIA', 'ER')].values == [1.0, 1.0, 1.0, -0.5])
    assert all(rate.loc[('INDIA', 'WR')].values == [1.0, 1.0, 1.0, -0.5])
    assert all(rate.columns == [12, 13, 14, 15])

    monkeypatch.setattr(loaders, 'get_parameter', get_GDP2)
    GDP = loaders.get_parameter('GDP')
    rate = demand.compute_gdp_rate(GDP)
    print(loaders.get_parameter('GDP'))
    print(rate)
    rate = rate.pivot(index=['ModelGeography', 'SubGeography1', 'SubGeography2'],
                      columns=['Year'],
                      values='GDP_RATE')
    assert all(rate.loc[('INDIA', 'ER', 'UT')].values == [1.0, 1.0, 1.0, -0.5])
    assert all(rate.loc[('INDIA', 'ER', 'BP')].values == [1.0, 1.0, 1.0, -0.5])
    assert all(rate.columns == [12, 13, 14, 15])


def get_parameter_gdp_demand(param_name, **kwargs):
    if param_name == "GDP":
        return get_GDP2(param_name)
    elif param_name == "DS_List":
        return ['DS']
    elif param_name == "DS_ES_EC_Map":
        return pd.DataFrame([{'DemandSector': 'DS',
                              'EnergyService': 'ES',
                              'EnergyCarrier': 'EC',
                              'ConsumerGranularity': 'CONSUMERTYPE1',
                              'GeographicGranularity': 'SUBGEOGRAPHY2',
                              'TimeGranularity': 'YEAR'}])
    elif param_name == 'DS_Cons1_Map':
        return {'DS': ['SUBGEOGRAPHY2', 'YEAR', 'URBAN', 'RURAL']}


def get_demand_index():
    index = pd.MultiIndex.from_product([['ES'],
                                        ['EC'],
                                        ['URBAN', 'RURAL'],
                                        ['INDIA'],
                                        ['ER'],
                                        ['UT']],
                                       names=['EnergyService',
                                              'EnergyCarrier',
                                              'ConsumerType1',
                                              'ModelGeography',
                                              'SubGeography1',
                                              'SubGeography2'])
    return index


def get_elasticity_index():
    index = pd.MultiIndex.from_product([['ES'],
                                        ['EC'],
                                        ['URBAN', 'RURAL'],
                                        ['INDIA'],
                                        ['ER'],
                                        ['UT'],
                                        [12, 13]],
                                       names=['EnergyService',
                                              'EnergyCarrier',
                                              'ConsumerType1',
                                              'ModelGeography',
                                              'SubGeography1',
                                              'SubGeography2',
                                              'Year'])
    return index


def _get_base_year_demand(*args):
    index = get_demand_index()
    return pd.DataFrame({'BaseYearDemand': [1000.0]*2,
                         'Year': [2020]*2},
                        index=index).reset_index()


def _get_demand_elasticity(*args):
    index = get_elasticity_index()
    return pd.DataFrame({'Elasticity': [5.0, 6.0, 3.0, 4.0]},
                        index=index).reset_index()


def test_compute_demand_growth_rate(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter_gdp_demand)
    monkeypatch.setattr(demand, 'get_demand_elasticity',
                        _get_demand_elasticity)

    gr = demand.compute_demand_growth_rate('DS', 'ES', 'EC')
    print(gr)
    assert all(gr.query("ConsumerType1 == 'RURAL' & Year==12 & SubGeography2=='UT'")
               ['GROWTH_RATE'].values == [3.0])
    assert all(gr.query("ConsumerType1 == 'RURAL' & Year==13 & SubGeography2=='UT'")
               ['GROWTH_RATE'].values == [4.0])
    assert all(gr.query("ConsumerType1 == 'URBAN' & Year==12 & SubGeography2=='UT'")
               ['GROWTH_RATE'].values == [5.0])
    assert all(gr.query("ConsumerType1 == 'URBAN' & Year==13 & SubGeography2=='UT'")
               ['GROWTH_RATE'].values == [6.0])


def test_compute_gdpelasticity_demand(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter_gdp_demand)
    monkeypatch.setattr(demand, 'get_demand_elasticity',
                        _get_demand_elasticity)
    monkeypatch.setattr(demand, 'get_base_year_demand',
                        _get_base_year_demand)
    d = demand.compute_gdpelasticity_demand('DS', 'ES', 'EC')
    print(d)
    assert d.query('ConsumerType1 == "URBAN"')[
        'EnergyDemand'].values == pytest.approx([6000.0, 42000.0, 42000.0, 42000.0])
    assert d.query('ConsumerType1 == "RURAL"')[
        'EnergyDemand'].values == pytest.approx([4000.0, 20000.0, 20000.0, 20000.0])


def test_get_cons_columns(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter',
                        get_parameter_granularity_coarsest)
    conscols = demand.get_cons_columns("DS1", "ES1", "EC1")
    assert conscols == []
    conscols = demand.get_cons_columns("DS1", "ES1", "EC3")
    assert conscols == ['ConsumerType1']


def get_parameter_granularity(param_name, **kwargs):
    if param_name == "DS_ES_EC_Map":
        return pd.DataFrame([{"DemandSector": "DS1",
                              "EnergyService": "ES1",
                              "EnergyCarrier": "EC1",
                              "ConsumerGranularity": "CONSUMERALL",
                              "GeographicGranularity": "SUBGEOGRAPHY1",
                              "TimeGranularity": "SEASON"}])
    elif param_name == "DS_List":
        return ["DS1"]


def get_parameter_granularity_coarsest(param_name, **kwargs):
    if param_name == "DS_ES_EC_Map":
        return pd.DataFrame([{"DemandSector": "DS1",
                              "EnergyService": "ES1",
                              "EnergyCarrier": "EC1",
                              "ConsumerGranularity": "CONSUMERALL",
                              "GeographicGranularity": "SUBGEOGRAPHY1",
                              "TimeGranularity": "SEASON"},
                             {"DemandSector": "DS1",
                              "EnergyService": "ES1",
                              "EnergyCarrier": "EC2",
                              "ConsumerGranularity": "CONSUMERALL",
                              "GeographicGranularity": "SUBGEOGRAPHY2",
                              "TimeGranularity": "YEAR"},
                             {"DemandSector": "DS1",
                              "EnergyService": "ES1",
                              "EnergyCarrier": "EC3",
                              "ConsumerGranularity": "CONSUMERTYPE1",
                              "GeographicGranularity": "MODELGEOGRAPHY",
                              "TimeGranularity": "SEASON"}])
    elif param_name == 'DS_Cons1_Map':
        return {'DS1': ['SUBGEOGRAPHY2', 'YEAR', 'URBAN', 'RURAL']}
    elif param_name == 'DS_List':
        return ['DS1']


def test_get_geographics_columns(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter_granularity)
    geogran = demandio.get_geographic_granularity("DS1",
                                                  "ES1",
                                                  "EC1")
    assert geogran == 'SUBGEOGRAPHY1'
    geocols = demand.get_geographic_columns(geogran)
    assert geocols == ['ModelGeography', 'SubGeography1']


def test_get_time_columns(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter_granularity)
    timegran = demand.get_time_granularity("DS1",
                                           "ES1",
                                           "EC1")
    assert timegran == 'SEASON'
    cols = demand.get_time_columns(timegran)
    assert cols == ['Year', 'Season']


def test_coarsest(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter',
                        get_parameter_granularity_coarsest)
    DS_ES_EC_Map = loaders.get_parameter(
        'DS_ES_EC_Map')
    cols = demand.coarsest(DS_ES_EC_Map, True)
    assert cols == ['Year',  'ModelGeography']
    cols = demand.coarsest(DS_ES_EC_Map, False)
    assert cols == ['Year',  'ModelGeography']


def get_parameter_bottomup(param_name, **kwargs):
    # TODO
    pass


def test_BottomupDemand(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter',
                        get_parameter_bottomup)
    # TODO


def get_parameter_gtprofile(param_name, **kwargs):
    if param_name == "GTProfile" and kwargs['demand_sector'] == 'DS' and kwargs['energy_service'] == "ES" and kwargs['energy_carrier'] == 'EC':
        return ioutils.base_dataframe_all(geocols=['ModelGeography', 'SubGeography1'], timecols=['Year'], colname='GTProfile', val=1/55).reset_index()
    elif param_name == "ModelPeriod":
        return pd.DataFrame({"StartYear": [2021], "EndYear": [2031]})
    elif param_name == "ModelGeography":
        return "INDIA"
    elif param_name == "SubGeography1":
        return ["NE", "NR", "SR", "WR", "ER"]


def get_demand(demand_sector, energy_service, energy_carrier):
    return ioutils.base_dataframe_all(geocols=['ModelGeography'],
                                      timecols=['Year'], colname='EnergyDemand',
                                      val=100.0).reset_index()


def test_apply_demand_profile(monkeypatch):
    monkeypatch.setattr(loaders, "get_parameter", get_parameter_gtprofile)
    d1 = get_demand("DS", "ES", "EC")
    func = demand.demand_profile(get_demand)
    d2 = func("DS", "ES", "EC")
    assert d1['EnergyDemand'].sum() == pytest.approx(d2['EnergyDemand'].sum())
    assert d2.groupby(['ModelGeography', 'Year']
                      ).sum(numeric_only=True).reset_index().equals(d1)


def test_get_coarsest(monkeypatch):
    def compute_demand(ds, es, ec):
        cols = {("DS", "ES", "EC1"): {"conscols": [],
                                      "geocols": ["ModelGeography"],
                                      "timecols": ['Year']},
                ("DS", "ES", "EC2"): {"conscols": [],
                                      "geocols": ["ModelGeography", 'SubGeography1'],
                                      "timecols": ['Year']},
                ("DS", "ES", "EC3"): {"conscols": [],
                                      "geocols": ["ModelGeography", 'SubGeography1'],
                                      "timecols": ['Year', 'Season']}}
        return ioutils.base_dataframe_all(**cols[(ds, es, ec)]).reset_index()

    def get_parameter(param_name, **kwargs):
        if param_name == "ModelPeriod":
            return pd.DataFrame([{"StartYear": 2021, "EndYear": 2022}])
        elif param_name == "DayTypes":
            return pd.DataFrame({"DayType": ['A', 'B'],
                                 "Weight": [0.4, 0.6]})
        elif param_name == "Seasons":
            return pd.DataFrame({"Season": ["SUMMER", "MONSOON"],
                                 "StartMonth": [4, 5],
                                 "StartDate": [1, 31]})
        elif param_name == "ModelGeography":
            return "INDIA"
        elif param_name == "SubGeography1":
            return ["NR", "SR", "NER", "WR", "ER"]
        elif param_name == "DaySlices":
            return ioutils.make_dataframe("""DaySlice,StartHour
H0,0
H1,13""")

    monkeypatch.setattr(loaders, "get_parameter", get_parameter)

    assert demand.get_coarsest([("DS", "ES", "EC1"),
                                ("DS", "ES", "EC2"),
                                ("DS", "ES", "EC3")],
                               data_loader=compute_demand) == ['Year', 'ModelGeography']

    assert demand.get_finest([("DS", "ES", "EC1"),
                              ("DS", "ES", "EC2"),
                              ("DS", "ES", "EC3")],
                             data_loader=compute_demand) == ['Year', 'Season',
                                                             'ModelGeography', 'SubGeography1']


def test_sum_series():
    ss = [pd.Series([1]*10) for i in range(5)]
    s = pd.Series([5]*10)

    assert demand.sum_series(ss).values == pytest.approx([5]*10)

    assert demand.sum_series([]) == 0


def test_get_output_filepath(monkeypatch, tmp_path):
    outputfolder = tmp_path / "Output"
    outputfolder.mkdir()

    monkeypatch.setattr(filemanager, "get_output_path",
                        lambda x: str(outputfolder))

    entity_names = ("ServiceTech",)
    entity_values = ("ST",)
    expected = outputfolder / "TotalNumInstances" / "TotalNumInstances_ST.csv"
    assert demand.get_output_filepath(
        entity_names, entity_values) == str(expected)

    entity_names = ("EnergyCarrier",)
    entity_values = ("EC",)
    expected = outputfolder / "EnergyCarrier" / "EC_Demand.csv"
    assert demand.get_output_filepath(
        entity_names, entity_values) == str(expected)

    entity_names = ("DemandSector", "EnergyService", "EnergyCarrier")
    entity_values = ("DS", "ES", "EC")
    expected = outputfolder / "DemandSector" / "DS" / "ES" / "DS_ES_EC_Demand.csv"
    assert demand.get_output_filepath(
        entity_names, entity_values) == str(expected)

    entity_names = ("DemandSector", "EnergyService",
                    "ServiceTech", "EnergyCarrier")
    entity_values = ("DS", "ES", "ST", "EC")
    expected = outputfolder / "DemandSector" / \
        "DS" / "ES" / "DS_ES_ST_EC_Demand.csv"
    assert demand.get_output_filepath(
        entity_names, entity_values) == str(expected)

    entity_names = ("EnergyService", "EnergyCarrier")
    entity_values = ("ES", "EC")
    expected = outputfolder / "EnergyService" / "ES" / "ES_EC_Demand.csv"
    assert demand.get_output_filepath(
        entity_names, entity_values) == str(expected)

    expected = outputfolder / "EndUseDemandEnergy.csv"
    assert demand.get_output_filepath(None, None) == str(expected)

    entity_names = ("DemandSector", "EnergyCarrier")
    entity_values = ("DS", "EC")
    expected = outputfolder / "DemandSector" / "DS" / "DS_EC_Demand.csv"
    assert demand.get_output_filepath(
        entity_names, entity_values) == str(expected)

    entity_names = ("DemandSector", "EnergyCarrier")
    entity_values = ("DS", "EC")
    expected = outputfolder / "DemandSector" / "DS" / "DS_EC_ES_ST_Demand.csv"
    assert demand.get_output_filepath(
        entity_names, entity_values, "ES_ST") == str(expected)
