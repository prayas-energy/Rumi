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


def get_parameter_gdp_demand(param_name):
    if param_name == "GDP":
        return get_GDP2(param_name)
    elif param_name == "DS_ES_EC_DemandGranularity_Map":
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
    assert conscols == ['ConsumerType1']
    conscols = demand.get_cons_columns("DS1", "ES1", "EC3")
    assert conscols == ['ConsumerType1']


def get_parameter_granularity(param_name):
    if param_name == "DS_ES_EC_DemandGranularity_Map":
        return pd.DataFrame([{"DemandSector": "DS1",
                              "EnergyService": "ES1",
                              "EnergyCarrier": "EC1",
                              "ConsumerGranularity": "CONSUMERALL",
                              "GeographicGranularity": "SUBGEOGRAPHY1",
                              "TimeGranularity": "SEASON"}])


def get_parameter_granularity_coarsest(param_name):
    if param_name == "DS_ES_EC_DemandGranularity_Map":
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
    DS_ES_EC_DemandGranularity_Map = loaders.get_parameter(
        'DS_ES_EC_DemandGranularity_Map')
    cols = demand.coarsest(DS_ES_EC_DemandGranularity_Map, True)
    assert cols == ['Year',  'ModelGeography', 'ConsumerType1']
    cols = demand.coarsest(DS_ES_EC_DemandGranularity_Map, False)
    assert cols == ['Year',  'ModelGeography']


def get_parameter_bottomup(param_name, **kwargs):
    # TODO
    pass


def test_BottomupDemand(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter',
                        get_parameter_bottomup)
    # TODO
