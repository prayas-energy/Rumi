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
from rumi.io import utilities
from rumi.io import loaders
import pytest

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
                       'Year': [2021]*4,
                       'ModelGeography': ['INDIA']*4,
                       'SubGeography1': ['NORTH', 'NORTH', 'SOUTH', 'SOUTH'],
                       'SubGeography2': ['UP', 'BIHAR', 'KERAL', 'KARNATAKA']})
    assert common.valid_geography(df)


def test_get_base_energy_density(monkeypatch):
    def get_parameter(name, *kwargs):
        if name == "ModelPeriod":
            return pd.DataFrame({"StartYear":[2021],
                                 "EndYear":[2025]})
        elif name == "PhysicalDerivedCarriers":
            return utilities.make_dataframe("""EnergyCarrier,BalancingArea,BalancingTime,EnergyUnit,PhysicalUnit,EnergyDensity
MS,MODELGEOGRAPHY,YEAR,PJ,MT,44.79876
HSD,MODELGEOGRAPHY,YEAR,PJ,MT,43.33338
ATF,MODELGEOGRAPHY,YEAR,PJ,MT,44.58942""")
        else:
            return None

    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    df = common.get_base_energy_density("PhysicalDerivedCarriersEnergyDensity")
    ms = df.query("EnergyCarrier == 'MS'")
    ms_edensity =  loaders.get_parameter("PhysicalDerivedCarriers").query("EnergyCarrier == 'MS'").loc[0,['EnergyDensity']].values[0]
    assert len(ms) == 5
    assert ms.equals(pd.DataFrame({"EnergyCarrier":['MS']*5,
                                   "EnergyDensity": [ms_edensity]*5,
                                   "Year": utilities.get_years()}))
    assert len(df) == len(loaders.get_parameter("PhysicalDerivedCarriers"))*5


def test_expand_energy_density(monkeypatch):
    def get_parameter(name, *kwargs):
        if name == "ModelPeriod":
            return pd.DataFrame({"StartYear":[2021],
                                 "EndYear":[2025]})
        elif name == "PhysicalPrimaryCarriers":
            return utilities.make_dataframe("""EnergyCarrier,BalancingArea,BalancingTime,PhysicalUnit,EnergyUnit,DomEnergyDensity,ImpEnergyDensity
COKING_COAL,SUBGEOGRAPHY1,YEAR,MT,PJ,20.9792705,28.0285326
STEAM_COAL,SUBGEOGRAPHY1,YEAR,MT,PJ,17.35076799,22.60872
NATGAS,MODELGEOGRAPHY,YEAR,BCM,PJ,48.6812,37.6812
BIOGAS,MODELGEOGRAPHY,YEAR,BCM,PJ,22.8,22.8
BIOMASS,MODELGEOGRAPHY,YEAR,MT,PJ,15.56,15.56
CRUDE,MODELGEOGRAPHY,YEAR,MT,PJ,43.12404,43.12404""")

    monkeypatch.setattr(loaders, "get_parameter", get_parameter)
    d = common.expand_energy_density("PhysicalPrimaryCarriersEnergyDensity",
                                     utilities.make_dataframe("""EnergyCarrier,Year,ImpEnergyDensity,DomEnergyDensity
COKING_COAL,2023,21.0,30.0"""))
    coal = d.query('EnergyCarrier == "COKING_COAL"')
    assert coal.Year.sum() == sum([2021, 2022, 2023, 2024, 2025])
    c1 = coal[coal.Year < 2023]
    assert c1.ImpEnergyDensity.sum() == pytest.approx(28.0285326*2)
    assert c1.DomEnergyDensity.sum() == pytest.approx(20.9792705*2)
    c2 = coal[coal.Year >= 2023]
    assert c2.ImpEnergyDensity.sum() == pytest.approx(21.0*3)
    assert c2.DomEnergyDensity.sum() == pytest.approx(30.0*3)
    natgas = d.query('EnergyCarrier == "NATGAS"')
    assert natgas.Year.sum() == sum([2021, 2022, 2023, 2024, 2025])
    assert natgas.ImpEnergyDensity.sum() == pytest.approx(37.6812*5)
    assert natgas.DomEnergyDensity.sum() == pytest.approx(48.6812*5)
    d = common.expand_energy_density("PhysicalPrimaryCarriersEnergyDensity",
                                     utilities.make_dataframe("""EnergyCarrier,Year,ImpEnergyDensity,DomEnergyDensity
COKING_COAL,2024,25.0,31.0
COKING_COAL,2023,21.0,30.0"""))
    coal = d.query('EnergyCarrier == "COKING_COAL"')
    assert coal.Year.sum() == sum([2021, 2022, 2023, 2024, 2025])
    c1 = coal[coal.Year < 2023]
    assert c1.ImpEnergyDensity.sum() == pytest.approx(28.0285326*2)
    assert c1.DomEnergyDensity.sum() == pytest.approx(20.9792705*2)
    c2 = coal[coal.Year == 2023]
    assert c2.ImpEnergyDensity.sum() == pytest.approx(21.0)
    assert c2.DomEnergyDensity.sum() == pytest.approx(30.0)
    c2 = coal[coal.Year >= 2024]
    assert c2.ImpEnergyDensity.sum() == pytest.approx(25.0*2)
    assert c2.DomEnergyDensity.sum() == pytest.approx(31.0*2)


def test_expand_carrier_emissions(monkeypatch):
    def get_parameter(name, *kwargs):
        if name == "ModelPeriod":
            return pd.DataFrame({"StartYear":[2021],
                                 "EndYear":[2025]})

    monkeypatch.setattr(loaders, "get_parameter", get_parameter)
        
    PrimaryCarrierEmissions = utilities.make_dataframe("""EnergyCarrier,EmissionType,Year,DomEmissionFactor,ImpEmissionFactor
COKING_COAL,CO,2021,21.0,30.0
COKING_COAL,CO2,2021,20.9792705,28.0285326
COKING_COAL,CO2,2024,25.0,31.0
COKING_COAL,CO2,2023,21.0,30.0""")
    d = common.expand_carrier_emissions("PrimaryCarrierEmissions",
                                        PrimaryCarrierEmissions)
    coal = d.query('EmissionType == "CO2"')
    assert coal.Year.sum() == sum([2021, 2022, 2023, 2024, 2025])
    c1 = coal[coal.Year < 2023]
    assert c1.ImpEmissionFactor.sum() == pytest.approx(28.0285326*2)
    assert c1.DomEmissionFactor.sum() == pytest.approx(20.9792705*2)
    c2 = coal[coal.Year == 2023]
    assert c2.ImpEmissionFactor.sum() == pytest.approx(30.0)
    assert c2.DomEmissionFactor.sum() == pytest.approx(21.0)
    c2 = coal[coal.Year >= 2024]
    assert c2.ImpEmissionFactor.sum() == pytest.approx(31.0*2)
    assert c2.DomEmissionFactor.sum() == pytest.approx(25.0*2)

    coal = d.query('EmissionType == "CO"')
    assert coal.Year.sum() == sum([2021, 2022, 2023, 2024, 2025])
    assert coal.ImpEmissionFactor.sum() == pytest.approx(30.0*5)
    assert coal.DomEmissionFactor.sum() == pytest.approx(21.0*5)


def test_first_year_present(monkeypatch):
    def get_parameter(name, *kwargs):
        if name == "ModelPeriod":
            return pd.DataFrame({"StartYear":[2021],
                                 "EndYear":[2025]})

    monkeypatch.setattr(loaders, "get_parameter", get_parameter)
    PrimaryCarrierEmissions = utilities.make_dataframe("""EnergyCarrier,EmissionType,Year,DomEmissionFactor,ImpEmissionFactor
COKING_COAL,CO2,2023,21.0,30.0""")
    assert not common.first_year_present(PrimaryCarrierEmissions)
    PrimaryCarrierEmissions = utilities.make_dataframe("""EnergyCarrier,EmissionType,Year,DomEmissionFactor,ImpEmissionFactor
COKING_COAL,CO2,2021,21.0,30.0
COKING_COAL,CO,2023,21.0,30.0""")
    assert not common.first_year_present(PrimaryCarrierEmissions)
    PrimaryCarrierEmissions = utilities.make_dataframe("""EnergyCarrier,EmissionType,Year,DomEmissionFactor,ImpEmissionFactor
COKING_COAL,CO2,2021,21.0,30.0
COKING_COAL,CO,2021,21.0,30.0""")
    assert common.first_year_present(PrimaryCarrierEmissions)
