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
from rumi.io import filemanager
from rumi.io import config
from rumi.io import common
from rumi.io import constant
from rumi.io import loaders
from rumi.io import utilities
import logging
import os
import functools
import numpy as np
import itertools
import math

logger = logging.getLogger(__name__)


def load_param(param_name, subfolder):
    """Loader function to be used by yaml framework. do not use this
    directly.
    """
    filepath = filemanager.find_filepath(param_name, subfolder)
    logger.debug(f"Reading {param_name} from file {filepath}")
    df = loaders.read_csv(param_name, filepath)
    return df


def get_filtered_parameter(param_name):
    """Returns supply parameter at balancing time and balancing area.
    This function will do necessary collapsing and expansion of
    parameter data. It will do this operation on all float64 columns.
    other columns will be treated as categorical.

    :param: param_name
    :returns: DataFrame

    """
    param_data_ = loaders.get_parameter(param_name)
    if not isinstance(param_data_, pd.DataFrame) and param_data_ == None:
        return param_data_
    original_order = [c for c in param_data_.columns]
    param_data = utilities.filter_empty(param_data_)  # for test data
    specs = filemanager.supply_specs()
    if param_name in specs:
        param_specs = specs[param_name]
        folder = param_specs.get("nested")
        geographic = param_specs.get("geographic")
        time = param_specs.get("time")
        if geographic:
            param_data = filter_on_geography(
                param_data, geographic, folder)

        if time:
            param_data = filter_on_time(param_data, time, folder)

        param_data = preserve_column_order(
            param_data, original_order)
    return param_data.fillna("")


def preserve_column_order(dataframe, original_order):
    class DummyDFColumns:
        """A class to simulate df.columns from pa.DataFrame
        """

        def __init__(self, cols):
            self.columns = list(cols)

    def indexof_geo(oldcols):
        subset_cols = utilities.get_geographic_columns_from_dataframe(
            oldcols)
        return oldcols.columns.index(subset_cols[-1])+1

    def indexof_time(oldcols):
        subset_cols = utilities.get_time_columns_from_dataframe(oldcols)
        return oldcols.columns.index(subset_cols[-1])+1

    def extra_geo(dataframe, oldcols):
        geo = utilities.get_geographic_columns_from_dataframe(dataframe)
        return [c for c in geo if c not in oldcols.columns]

    def extra_time(dataframe, oldcols):
        time = utilities.get_time_columns_from_dataframe(dataframe)
        return [c for c in time if c not in oldcols.columns]

    def new_order(dataframe, oldcols):
        cols = [c for c in oldcols]
        oldcols_ = DummyDFColumns(cols)
        if utilities.get_geographic_columns_from_dataframe(oldcols_):
            for i, c in enumerate(extra_geo(dataframe, oldcols_),
                                  start=indexof_geo(oldcols_)):
                cols.insert(i, c)

            oldcols_ = DummyDFColumns(cols)

        if utilities.get_time_columns_from_dataframe(oldcols_):
            for i, c in enumerate(extra_time(dataframe, oldcols_),
                                  start=indexof_time(oldcols_)):
                cols.insert(i, c)

        return cols

    return dataframe.reindex(columns=new_order(dataframe, original_order))


def filter_empty_columns(data, filtercols):
    rows = len(data)
    empty = [c for c in filtercols if data[c].isnull(
    ).sum() == rows or (data[c] == "").sum() == rows]
    return data[[c for c in data.columns if c not in empty]]


def filter_empty_geography(data):
    """filter out empty geographic columns"""
    return filter_empty_columns(data,
                                utilities.get_geographic_columns_from_dataframe(data))


def filter_empty_time(data):
    """filter out empty time columns"""
    return filter_empty_columns(data,
                                utilities.get_time_columns_from_dataframe(data))


def finest_geography_from_balancing(entities):
    g = [common.get_geographic_columns(
        common.balancing_area(e)) for e in entities]
    return max(g, key=len)


@functools.lru_cache(maxsize=1)
def get_all_carriers():
    carrriers = ["PhysicalPrimaryCarriers",
                 "PhysicalDerivedCarriers", "NonPhysicalDerivedCarriers"]
    allcarriers = []
    for carrrier in carrriers:
        allcarriers.extend(
            list(loaders.get_parameter(carrrier)['EnergyCarrier']))
    return allcarriers


def finest_time_from_balancing(entities):
    t = [common.get_time_columns(common.balancing_time(e)) for e in entities]
    return max(t, key=len)


@functools.lru_cache(maxsize=16)
def find_EC(entity, value):
    if entity == 'EnergyCarrier':
        return value
    elif entity == 'EnergyConvTech':
        EnergyConvTechnologies = loaders.get_parameter(
            'EnergyConvTechnologies')
        ect = EnergyConvTechnologies.set_index('EnergyConvTech')
        return ect.loc[value]['OutputDEC']
    else:
        EnergyStorTechnologies = loaders.get_parameter(
            'EnergyStorTechnologies')
        est = EnergyStorTechnologies.set_index('EnergyStorTech')
        return est.loc[value]['StoredEC']


def get_entity_type(folder):
    if folder == "Carriers":
        return 'EnergyCarrier'
    elif folder == "Storage":
        return 'EnergyStorTech'
    else:
        return 'EnergyConvTech'


def filter_on_time(data, granularity, folder):
    """granularity is either 'fine' or 'coarse' and folder is one of 'Carriers',
    'Technologies', 'Storage'
    """
    entity = get_entity_type(folder)
    entities = get_all_carriers()
    timecols = finest_time_from_balancing(entities)

    dfs = []

    if granularity == "fine":
        for item in data[entity].unique():
            q = f"{entity} == '{item}'"
            d = filter_empty_time(data.query(q))
            balancing_time = common.balancing_time(find_EC(entity, item))
            d = group_by_time(d, balancing_time, timecols)
            dfs.append(d)

    else:
        for item in data[entity].unique():
            q = f"{entity} == '{item}'"
            d = filter_empty_time(data.query(q))
            balancing_time = common.balancing_time(find_EC(entity, item))
            d = expand_by_time(d, entity, balancing_time, timecols)
            dfs.append(d)

    return pd.concat(dfs).reset_index(drop=True)


def get_nontime_columns(d):
    return [c for c in d.columns if (not pd.api.types.is_float_dtype(d[c])) and c not in constant.TIME_SLICES]


def group_by_time(d, balancing_time, superset_cols):
    timecols_ = common.get_time_columns(balancing_time)
    othercols = get_nontime_columns(d)
    d = utilities.groupby_time(d.fillna(""), othercols, balancing_time).copy()

    rows = len(d)
    diff = [c for c in superset_cols if c not in timecols_]

    for c in diff:
        d[c] = pd.Series([""]*rows, dtype=str, name=c)
    return d[superset_cols + [c for c in d.columns if c not in superset_cols]]


def expand_by_time(d, entity, balancing_time, superset_cols):
    timecols_ = common.get_time_columns(balancing_time)
    label = d[entity].unique()[0]
    base = utilities.base_dataframe_time(timecols_,
                                         colname=entity,
                                         val=label).reset_index()

    d = d.merge(base, how='left')

    rows = len(d)
    diff = [c for c in superset_cols if c not in timecols_]

    for c in diff:
        d[c] = pd.Series([""]*rows, dtype=str, name=c)
    return d


def filter_on_geography(data, granularity, folder):
    """granularity is either 'fine' or 'coarse' and folder is one of 'Carriers',
    'Technologies', 'Storage'
    """
    entity = get_entity_type(folder)
    entities = get_all_carriers()
    geocols = finest_geography_from_balancing(entities)

    dfs = []

    if granularity == "fine":
        for item in data[entity].unique():
            q = f"{entity} == '{item}'"
            d = filter_empty_geography(data.query(q))
            balancing_area = common.balancing_area(find_EC(entity, item))
            d = group_by_geographic(d, balancing_area, geocols)
            dfs.append(d)

    else:
        for item in data[entity].unique():
            q = f"{entity} == '{item}'"
            d = filter_empty_geography(data.query(q))
            balancing_area = common.balancing_area(find_EC(entity, item))
            d = expand_by_geographic(d, entity, balancing_area, geocols)
            dfs.append(d)

    return pd.concat(dfs).reset_index(drop=True)


def expand_by_geographic(d, entity, balancing_area, superset_cols):
    geocols_ = common.get_geographic_columns(balancing_area)

    label = d[entity].unique()[0]
    base = utilities.base_dataframe_geography(geocols_,
                                              colname=entity,
                                              val=label).reset_index()
    d = d.merge(base, how='left')

    rows = len(d)
    diff = [c for c in superset_cols if c not in geocols_]

    for c in diff:
        d[c] = pd.Series([""]*rows, dtype=str, name=c)
    return d


def get_nongeographic_columns(d):
    """get non geographics non numeric columns"""
    return [c for c in d.columns if (not pd.api.types.is_float_dtype(d[c])) and c not in constant.GEOGRAPHIES]


def group_by_geographic(d, balancing_area, superset_cols):
    geocols_ = common.get_geographic_columns(balancing_area)
    othercols = get_nongeographic_columns(d)
    d = d.groupby(othercols + geocols_).sum().reset_index()

    rows = len(d)
    diff = [c for c in superset_cols if c not in geocols_]

    for c in diff:
        d[c] = pd.Series([""]*rows, dtype=str, name=c)

    return d[superset_cols + [c for c in d.columns if c not in superset_cols]]


def find_C_G_columns(data):
    cstar_gstar = constant.TIME_SLICES + constant.GEOGRAPHIES
    return [c for c in data.columns if c in cstar_gstar]


def check_granularity(data, granularity, entity, GSTAR=None, TSTAR=None):
    """
    Checks whether given data follows granularity as specified in granularity map.
    """
    if GSTAR == None and TSTAR == None:
        raise Exception(
            "check_granularity function must have valid GSTAR/TSTAR argument")

    valid = True
    granularity = granularity.set_index(entity)
    for item in data[entity].unique():
        q = f"{entity} == '{item}'"
        d = utilities.filter_empty(data.query(q))
        g = granularity.loc[item]
        GeographicGranularity, TimeGranularity = None, None
        if GSTAR:
            GeographicGranularity = g['GeographicGranularity']
        if TSTAR:
            TimeGranularity = g['TimeGranularity']
        valid = valid and utilities.check_granularity_per_entity(d,
                                                                 item,
                                                                 GeographicGranularity,
                                                                 TimeGranularity)

    return valid


def check_balancing_area(param_name, data, entity):
    """
    check if given data is spcified at geographics granularity of corresponding
    balacing area
    """

    for entity_value in data[entity].unique():
        entity_ = find_EC(entity, entity_value)
        geogran = utilities.balancing_area(entity_)
        geocols = utilities.get_geographic_columns(geogran)
        subset = utilities.filter_empty(
            data.query(f"{entity} == '{entity_value}'"))
        geocols_ = utilities.get_geographic_columns_from_dataframe(subset)
        valid = set(geocols) == set(geocols_)
        if not valid:
            logger.error(
                f"In {param_name} , data for {entity}, {entity_} is not specified at balancing area level. The data should be specified at geographic granularity {geogran}")
            return False

    return True


def check_balancing_time(param_name, data, entity):
    """
    check if given data is spcified at time granularity of corresponding
    balancing time. comp specifies what kind of check to perform, equal, coarser or finer.
    """

    for entity_ in data[entity].unique():
        timegran = utilities.balancing_time(entity_)
        timecols = utilities.get_time_columns(timegran)
        subset = utilities.filter_empty(data.query(f"{entity} == '{entity_}'"))
        timecols_ = utilities.get_time_columns_from_dataframe(subset)
        valid = set(timecols) == set(timecols_)
        if not valid:
            logger.error(
                f"In {param_name} , data for {entity}, {entity_} is not specified at balacing time level. The data should be specified at time granularity {timegran}")
            return False

    return True


def check_balancing_area_gran(param_name,
                              granmap,
                              entity,
                              comp,
                              find_EC_=find_EC):
    return utilities.check_balancing_area_gran(param_name, granmap, entity, comp, find_EC_)


def check_balancing_time_gran(param_name,
                              granmap,
                              entity,
                              comp,
                              find_EC_=find_EC):
    return utilities.check_balancing_time_gran(param_name, granmap, entity, comp, find_EC_)


def get_geo_cols(data, postfix="Src"):
    """used to find geographic columns with given postfix.
    used for ECT_Transfers parameter
    """
    subset = data[[c for c in data.columns if c.endswith(postfix)]]
    subset = subset.rename(
        columns={c: c.replace(postfix, "") for c in subset.columns})
    return utilities.get_geographic_columns_from_dataframe(subset)


def check_balancing_area_src_dest(param_name, data, entity):
    """
    check if given data is spcified at source/destination geographic granularity of corresponding balacing area
    """
    for entity_ in data[entity].unique():
        geogran = utilities.balancing_area(entity_)
        geocols = utilities.get_geographic_columns(geogran)
        subset = utilities.filter_empty(data.query(f"{entity} == '{entity_}'"))

        geocols_src = get_geo_cols(subset)
        geocols_dest = get_geo_cols(subset, "Dest")

        if len(geocols_dest) != len(geocols_src):
            logger.error(
                "In {param_name}, source and destination do not have same geographic granularity for {entity_}.")
            return False
        if len(geocols) != len(geocols_dest):
            logger.error(
                "In {param_name}, geographic granularity should be of granularity {geogran} for {entity_}.")
            return False

    return True


def validate_units_config(param_name):
    """Validate units configuration parameter.
    It checks if the parameter is given in matrix format, where
    first row and column of matrix is names of units and it
    those should be same if compared as a vector.

    next nth row and nth column (neglecting names column and row)
    have cross product which results in a vector with ones.
    Parameters
    -----------
    param_name: str
       name of parameter to test.

    Returns
    -------
    True or False
    """
    data = loaders.get_config_parameter(param_name)
    if not all(list(data.columns[1:]) == data.iloc[:, 0]):
        logger.error(
            f"For config parameter {param_name}, column names should be same as rows in first column")
        return False
    values = data.iloc[:, 1:].values
    for col, row in zip([data[c].values for c in data.columns[1:]], values):
        if not np.allclose(col*row, np.ones_like(col), rtol=0.01, atol=0.001):
            logger.error(
                f"For config parameter {param_name}, values in Xth column and Xth row should be inverse of each other.")
            return False
    return True


def get_valid_stor_periodicity():
    """get valid strings for stor periodicity
    checks time levels defined in common and accordingly
    returns valid strings for stor periodicity.

    Returns
    -------
    a list containing valid strings for stor periodicity
    """
    periodicity = ('DAILY', 'SEASONAL', 'ANNUAL', 'NEVER')
    if isinstance(loaders.get_parameter("DayTypes"), pd.DataFrame):
        periodicity_ = periodicity
    elif isinstance(loaders.get_parameter("Seasons"), pd.DataFrame):
        periodicity_ = periodicity[1:]
    else:
        periodicity_ = periodicity[2:]

    return periodicity_


def check_maxtransit():
    """checks if source and destination geographies are same maxtransit can not be zero.

    Returns
    -------
    False if maxtransit is zero for same source and destination geographies else returns True
    """

    EC_Transfers = loaders.get_parameter("EC_Transfers")
    for entity in EC_Transfers['EnergyCarrier'].unique():
        subset = utilities.filter_empty(
            EC_Transfers.query(f"EnergyCarrier == '{entity}'"))

        geocols_src = get_geo_cols(subset)
        geocols_dest = get_geo_cols(subset, "Dest")

        q = " & ".join([f"{src}Src == {dest}Dest" for src,
                        dest in zip(geocols_src, geocols_dest)])
        max_transit0 = subset.query(" & ".join([q, "MaxTransit==0.0"]))
        if len(max_transit0) > 0:
            logger.error(
                f"Following lines in EC_Transfers must not have zero MaxTransit")
            logger.error(",".join([c for c in max_transit0.columns]))
            for row in max_transit0.values:
                logger.error(",".join([str(item) for item in row]))
            return False

        if check_absent_lines(subset, geocols_src, entity):
            continue
        else:
            return False

    return True


def check_absent_lines(max_transit0, geocols_src, entity):
    """checks if some row for src==dest is absent in data.
    it assumes that data given for only one EnergyCarrier
    """
    for year in max_transit0.Year.unique():
        df = max_transit0.query(f"Year == {year}")
        s = [f"{item}Src" for item in geocols_src]
        df = df.rename(columns=dict(
            zip(s, geocols_src))).reset_index(drop=True)
        basedf = utilities.base_dataframe_geography(geocols_src).reset_index()
        del basedf['dummy']

        check = (~basedf.isin(df[geocols_src]))
        filterq = check.sum(axis=1) == len(geocols_src)
        notpresent = basedf[filterq]
        if len(notpresent) > 0:
            logger.error(
                f"Following geographis with same source and destination in EC_Transfers are absent for {entity}, {year}")
            logger.error(",".join([c for c in notpresent.columns]))
            for row in notpresent.values:
                logger.error(",".join([str(item) for item in row]))
            return False
    return True


def check_geographic_validity_ec_transfers():
    """checks geographic validity of source and destination geographies
    """

    EC_Transfers = loaders.get_parameter("EC_Transfers")
    geo_cols = get_geo_cols(EC_Transfers)

    source = [c+"Src" for c in geo_cols]
    dest = [c+"Dest" for c in geo_cols]
    v = utilities.check_geographic_validity(EC_Transfers.rename(columns=dict(zip(source, geo_cols))),
                                            "EC_Transfers",
                                            ['EnergyCarrier'] + dest, False)
    if not v:
        logger.error(
            f"For EC_Transfers geographic validity for source geographies failed")
        return False

    v = utilities.check_geographic_validity(EC_Transfers.rename(columns=dict(zip(dest, geo_cols))),
                                            "EC_Transfers",
                                            ['EnergyCarrier'] + source, True)
    if not v:
        logger.error(
            f"For EC_Transfers geographic validity for destination geographies failed")
        return False

    return True


def check_time_validity_ec_transfers():
    """checks geographic validity of source and destination geographies
    """

    EC_Transfers = loaders.get_parameter("EC_Transfers")
    geo_cols = get_geo_cols(EC_Transfers)

    source = [c+"Src" for c in geo_cols]
    dest = [c+"Dest" for c in geo_cols]
    v = utilities.check_time_validity(EC_Transfers,
                                      "EC_Transfers",
                                      ['EnergyCarrier'] + source + dest, True)
    if not v:
        logger.error(
            f"For EC_Transfers, validity of time columns for has failed")
        return False

    return True


def check_retirement_capacity(ECT_LegacyRetirement, ECT_LegacyCapacity):
    geocols = utilities.get_geographic_columns_from_dataframe(
        ECT_LegacyCapacity)
    groupcols = geocols + ["EnergyConvTech"]
    retirement_plan = ECT_LegacyRetirement.groupby(groupcols).sum()[
        'RetCapacity']
    capacity = ECT_LegacyCapacity.groupby(groupcols).sum()['LegacyCapacity']
    capacity_subset = capacity.loc[retirement_plan.index.values]
    notexceed = retirement_plan <= capacity_subset
    if all(notexceed):
        return True
    else:
        logger.error(
            f"Sum of retirement capacity for following EnergyConTechs exceeds than original capacity specified in ECT_LegacyCapacity")
        evt = retirement_plan[~notexceed].index.values
        for items in evt:
            logger.error(",".join(list(items)))
        return False


def check_unique_ec_transfers(EC_Transfers):
    geo_cols = get_geo_cols(EC_Transfers)
    source = [c+"Src" for c in geo_cols]
    dest = [c+"Dest" for c in geo_cols]

    return utilities.unique_across(EC_Transfers,
                                   ['EnergyCarrier', 'Year'] + source + dest)


def check_periodicity_coarseness(EnergyStorTechnologies):
    """Checks if evergy StoredEC has periodicity specified is coarser
    than balancing time of corresponding EnergyCarrier


    Parameters
    -----------
    EnergyStorTech: pd.DataFrame

    Returns
    -------
    True or False
    """
    EnergyStorTechnologies = EnergyStorTechnologies.set_index('EnergyStorTech')

    def _coarser(peridicity, balancing_time):
        time_grans = ['NEVER'] + list(constant.TIME_COLUMNS.keys())
        p_grans = {'DAILY': 'DAYTYPE',
                   'SEASONAL': 'SEASON',
                   'ANNUAL': 'YEAR',
                   'NEVER': 'NEVER'}
        return time_grans.index(p_grans[peridicity]) <= time_grans.index(balancing_time)

    valid = True
    for stor_tech in EnergyStorTechnologies.index:
        energy_carrier = EnergyStorTechnologies.loc[stor_tech]['StoredEC']
        balancing_time = utilities.balancing_time(energy_carrier)
        peridicity = EnergyStorTechnologies.loc[stor_tech]['StorPeriodicity']
        if not _coarser(peridicity, balancing_time):
            logger.error(
                f"In EnergyStorTech, StorPeriodicity for {stor_tech} should be coarser than {balancing_time}")
            valid = False
    return valid


def check_stortech_daytypes(EnergyStorTechnologies):
    """cheks if daytypes given in common are consistent 
    with peridicity given in EnergyStorTech

    Parameters
    -----------
    EnergyStorTech: pd.DataFrame
       description for arg      

    Returns
    -------
    True or False
    """
    EnergyStorTechnologies = EnergyStorTechnologies.set_index('EnergyStorTech')
    valid = True
    for stor_tech in EnergyStorTechnologies.index:
        energy_carrier = EnergyStorTechnologies.loc[stor_tech]['StoredEC']
        balancing_time = utilities.balancing_time(energy_carrier)
        peridicity = EnergyStorTechnologies.loc[stor_tech]['StorPeriodicity']
        if balancing_time in ["DAYSLICE", 'DAYTYPE'] and peridicity != 'DAILY':
            if len(utilities.get_daytypes()) > 1:
                logger.error(
                    f"In EnergyStorTech, StorPeriodicity for {energy_carrier} indicates that there can not be more than one DAYTYPEs")
                valid = False
    return valid
