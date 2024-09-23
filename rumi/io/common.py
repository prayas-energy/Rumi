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
"""Functions to be used by common. all functions from this module
are available in Common.yml validation code.
"""
import datetime
import functools
import pandas as pd

from rumi.io import utilities
from rumi.io import loaders
from rumi.io import constant
from rumi.io.utilities import seasons_size, compute_intervals
from rumi.io.utilities import get_geographic_columns
from rumi.io.utilities import get_time_columns
from rumi.io.utilities import valid_geography
from rumi.io.utilities import balancing_area
from rumi.io.utilities import balancing_time
from rumi.io import functionstore as fs


class CommonValidationError(Exception):
    pass


def drop_columns(tabular, n):
    """drops first n items from each row and returns new tabular data

    >>> drop_columns([[1, 2, 3],
                      [21, 22, 23],
                      [31, 32, 33]],
                     1)
    [[2, 3], [22, 23], [32, 33]]
    """
    return [row[n:] for row in tabular]


def concat(*args):
    return sum([list(item) for item in args], [])


def find_hours(end, start):
    if end < start:
        return 24-start + end
    else:
        return end - start


def dayslice_size():
    DaySlices = loaders.get_parameter("DaySlices")
    dayslices = DaySlices.to_dict(orient='records')
    dayslices.append(dayslices[0])
    return {s['DaySlice']: find_hours(dayslices[i+1]['StartHour'], s['StartHour'])
            for i, s in enumerate(dayslices[:-1])}


def sum_of_durations(Seasons):
    intervals = compute_intervals(Seasons)
    return sum(intervals.values())


def get_base_energy_density(param_name):
    Year = pd.Series(utilities.get_years(),
                     name='Year')
    if param_name == "PhysicalDerivedCarriersEnergyDensity":
        PhysicalDerivedCarriers = loaders.get_parameter(
            "PhysicalDerivedCarriers")
        basedata = PhysicalDerivedCarriers[['EnergyCarrier',
                                            'EnergyDensity']]
    else:
        PhysicalPrimaryCarriers = loaders.get_parameter(
            "PhysicalPrimaryCarriers")
        basedata = PhysicalPrimaryCarriers[['EnergyCarrier',
                                            'ImpEnergyDensity',
                                            'DomEnergyDensity']]

    return basedata.merge(Year, how="cross")


def running_override(a, b, v):
    """
    >>> running_override([1,2,3,4,5], [1,3], [10,30])
    [10,10,30,30,30]
    """
    j = 0
    values = []

    for i, item in enumerate(b, start=-1):
        val = v[i]
        while a[j] != item and j < len(a):
            values.append(val)
            j += 1

    diff = len(a) - len(values)
    return values + [v[-1]]*diff


def check_year_data(param_name, entity):
    """for given param, checks if data given is within ModelPeriod
    """
    data = loaders.get_parameter(param_name)
    if fs.isnone(data):
        return True
    return utilities.check_time_validity(data,
                                         param_name,
                                         entity,
                                         checkunique=True,
                                         exact=False)


def first_year_value(baseyear, basevalues, refinedyear, refinedvalues):
    if len(refinedyear) == 0 and len(refinedvalues) == 0:
        return list(baseyear), list(basevalues)
    if refinedyear[0] != baseyear[0]:
        refinedyear = [baseyear[0]] + list(refinedyear)
        refinedvalues = [basevalues[0]] + list(refinedvalues)

    return refinedyear, refinedvalues


def override_energy_density(param_name, basedata, data):
    a = basedata.Year.values
    b = data.Year.values
    if param_name == "PhysicalDerivedCarriersEnergyDensity":
        v = data['EnergyDensity'].values
        b, v = first_year_value(a, basedata.EnergyDensity.values, b, v)
        values = running_override(a, b, v)
        return pd.DataFrame({"EnergyCarrier": basedata.EnergyCarrier.copy(),
                             "Year": basedata.Year.copy(),
                             "EnergyDensity": values})
    else:
        v = data['ImpEnergyDensity'].values
        b, v = first_year_value(a, basedata.ImpEnergyDensity.values, b, v)
        ImpValues = running_override(a, b, v)
        b = data.Year.values
        v = data['DomEnergyDensity'].values
        b, v = first_year_value(a, basedata.DomEnergyDensity.values, b, v)
        DomValues = running_override(a, b, v)
        return pd.DataFrame({"EnergyCarrier": basedata.EnergyCarrier.copy(),
                             "Year": basedata.Year.copy(),
                             "ImpEnergyDensity": ImpValues,
                             "DomEnergyDensity": DomValues})


def expand_carrier_emissions(param_name: str,
                             data: pd.DataFrame) -> pd.DataFrame:
    """If Imp/Dom EmissionFatcor is given for few year it expands
    it to complete time spectrum in years
    """
    if data is None:
        return data
    data = data.sort_values(by=['EnergyCarrier', 'EmissionType', 'Year'])
    dfs = []
    Year = pd.Series(utilities.get_years(),
                     name='Year')
    a = Year.values
    size = len(a)
    for EnergyCarrier, EmissionType in data.set_index(['EnergyCarrier',
                                                       'EmissionType']).index.unique():
        q = f"EnergyCarrier == '{EnergyCarrier}' & EmissionType == '{EmissionType}'"
        d_subset = data.query(q)
        b = d_subset.Year.values
        v = d_subset.ImpEmissionFactor.values
        b, v = first_year_value(a, [v[0]]*size, b, v)
        ImpValues = running_override(a, b, v)

        b = d_subset.Year.values
        v = d_subset.DomEmissionFactor.values
        b, v = first_year_value(a, [v[0]]*size, b, v)
        DomValues = running_override(a, b, v)
        data_ = pd.DataFrame({"EnergyCarrier": [EnergyCarrier]*size,
                             "EmissionType": [EmissionType]*size,
                              "Year": Year.copy(),
                              "ImpEmissionFactor": ImpValues,
                              "DomEmissionFactor": DomValues})
        dfs.append(data_)

    return pd.concat(dfs)


def expand_energy_density(param_name: str, data: pd.DataFrame) -> pd.DataFrame:
    """If EnergyDensity or Imp/Dom EnergyDensity is given few year it expands
    it to complete time spectrum in years
    """
    basedata = get_base_energy_density(param_name)
    if fs.isnone(data):
        return basedata

    data = data.sort_values(by=['EnergyCarrier', 'Year'])
    dfs = []
    for EnergyCarrier in basedata.EnergyCarrier.unique():
        q = f"EnergyCarrier == '{EnergyCarrier}'"
        bdf = basedata.query(q)
        if EnergyCarrier in data.EnergyCarrier.values:
            d_subset = data.query(q)
            dfs.append(override_energy_density(param_name, bdf, d_subset))
        else:
            dfs.append(bdf)

    return pd.concat(dfs)


def first_year_present(data: pd.DataFrame) -> bool:
    def check_first_year(items):
        q = " & ".join([f'{name} == "{value}"' for name, value in zip(indexcols,
                                                                      items)])
        subset = data.query(q)
        return first_year in subset.Year.values

    first_year = utilities.get_years()[0]
    indexcols = ['EnergyCarrier', 'EmissionType']
    return all([check_first_year(items) for items in data.set_index(indexcols).index.unique()])
