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
from rumi.io import common
from rumi.io import constant
from rumi.io import loaders
from rumi.io import utilities
from rumi.io import functionstore
import logging
import re
import os
import functools
import numpy as np
import itertools
import csv
import string
from pyomo.environ import Var

logger = logging.getLogger(__name__)
print = functools.partial(print, flush=True)

CONSTRAINT_END_KEYWORD = "BOUNDS"
COMMENT_KEYWORD = "COMMENT"
called_from_rumi_validate = True


def load_param(param_name, subfolder):
    """Loader function to be used by yaml framework. do not use this
    directly.
    """
    filepath = filemanager.find_filepath(param_name, subfolder)
    logger.debug(f"Reading {param_name} from file {filepath}")
    df = loaders.read_csv(param_name, filepath)
    return df


def get_filtered_parameter(param_name, **kwargs):
    """Returns supply parameter at balancing time and balancing area.
    This function will do necessary collapsing and expansion of
    parameter data. It will do this operation on all float64 columns.
    other columns will be treated as categorical.

    :param: param_name
    :returns: DataFrame

    """
    param_data_ = loaders.get_parameter(param_name, **kwargs)
    if not isinstance(param_data_, pd.DataFrame) or param_data_ is None:
        return param_data_
    original_order = [c for c in param_data_.columns]
    param_data = utilities.filter_empty(param_data_)  # for test data
    specs = filemanager.supply_specs()
    if param_name in specs:
        param_specs = specs[param_name]
        folder = param_specs.get("nested")
        geographic = param_specs.get("geographic")
        time = param_specs.get("time")
        granularity_exception = param_specs.get("granularity_exception")

        if granularity_exception:
            geographic = "fine"
            time = "fine"
            # this is for two parameters DEC_ImpConstraints and PEC_ProdImpConstraints
            # for these parameters we will assume that data will always come finer or equal
            # to BA/BT but it is actually not. in the function call group_by_geographic/
            # group_by_time it will be handled that if granularity is coarser then skip
            # grouping.
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

    if is_finer(d, timecols_):
        d = utilities.groupby_time(
            d.fillna(""), othercols, balancing_time).copy()
    else:
        d = d.fillna("").copy()

    rows = len(d)
    diff = [c for c in superset_cols if c not in d.columns]

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


def is_finer(dataframe, grancols):
    """returns True if dataframe is finer or equal to granuarity
    """
    return all([c in dataframe.columns for c in grancols])


def group_by_geographic(d, balancing_area, superset_cols):
    geocols_ = common.get_geographic_columns(balancing_area)
    othercols = get_nongeographic_columns(d)

    if is_finer(d, geocols_):
        d = d.groupby(
            othercols + geocols_).sum(numeric_only=True).reset_index()

    rows = len(d)

    diff = [c for c in superset_cols if c not in d.columns]

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


def check_balancing_area(param_name, data, entity, equal=True):
    """
    check if geographic granularity of data is coarser or equal to corresponding balacing area

    if equal is true then it checks if data is exactly at balacing area.
    """

    for entity_value in data[entity].unique():
        entity_ = find_EC(entity, entity_value)
        geogran = utilities.balancing_area(entity_)
        geocols = utilities.get_geographic_columns(geogran)
        subset = utilities.filter_empty(
            data.query(f"{entity} == '{entity_value}'"))
        geocols_ = utilities.get_geographic_columns_from_dataframe(subset)
        if equal:
            valid = set(geocols) == set(geocols_)
            msg = "at granuarity which is not equal to balacing area. The data should be provided at granularity"
        else:
            valid = set(geocols) >= set(geocols_)
            msg = "at finer than balancing area. The data should be coarser or equal to"
        if not valid:
            logger.error(
                f"In {param_name} , data for {entity}, {entity_} is specified {msg} {geogran}")
            return False

    return True


def check_balancing_time(param_name, data, entity, equal=True):
    """
    check if time granularity of data is coarser or equal to balacing time.
    if equal if true then it checks id data is provided exactly at balancing time.
    """

    for entity_ in data[entity].unique():
        timegran = utilities.balancing_time(entity_)
        timecols = utilities.get_time_columns(timegran)
        subset = utilities.filter_empty(data.query(f"{entity} == '{entity_}'"))
        timecols_ = utilities.get_time_columns_from_dataframe(subset)
        if equal:
            valid = set(timecols) == set(timecols_)
            msg = "at granuarity which is not equal to balacing time. The data should be procided at granularity"
        else:
            valid = set(timecols) >= set(timecols_)
            msg = "at finer than balancing time. The data should be coarser or equal to"
        if not valid:
            logger.error(
                f"In {param_name} , data for {entity}, {entity_} is specified {msg} {timegran}")
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
                f"In {param_name}, source and destination do not have same geographic granularity for {entity_}.")
            return False
        if len(geocols) != len(geocols_dest):
            logger.error(
                f"In {param_name}, geographic granularity should be of granularity {geogran} for {entity_}.")
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
    retirement_plan = ECT_LegacyRetirement.groupby(groupcols).sum(numeric_only=True)[
        'RetCapacity']
    capacity = ECT_LegacyCapacity.groupby(groupcols).sum(
        numeric_only=True)['LegacyCapacity']
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


def check_enduse(checkname, checkfunc, endusedemand, *args):
    """Performs given check using checkfunc on EndUseDemandEnergy.
    Because EndUseDemandEnergy can have multiple granularities
    present for single EC, it needs special handling. It is done
    using this wrapper function
    """
    def process_levels(data):
        ds = []
        cols = []
        for l in data.colindicator.unique():
            d = utilities.filter_empty(data.query(f"colindicator == '{l}'"))
            ds.append(d)

        return all([checkfunc(d, *args) for d in ds])

    valid = True
    for ec in endusedemand.EnergyCarrier.unique():
        d = endusedemand.query(f"EnergyCarrier == '{ec}'").copy()
        colindicator = d.T.eq("").apply(
            lambda x: x.apply(lambda y: str(int(y)))).sum()
        d['colindicator'] = colindicator
        v = process_levels(d)
        if not v:
            logger.error(f"{checkname} failed for EndUseDemandEnergy for {ec}")
        valid = valid and v

    return valid


def process_range(element):
    """process element as a range list if integers if given else do nothing
    """
    pattern = re.compile(r"(?P<start>\d+)-(?P<end>\d+)")
    match = pattern.match(element)
    if match:
        start = int(match.groupdict()['start'])
        end = int(match.groupdict()['end'])
        if start > end:
            logger.error(
                f"Integer range {element}, given in UserConstraints has wrong values.")
            return element
        else:
            return ",".join(str(i) for i in range(start, end+1))
    else:
        return element


def float_(textdata):
    try:
        return float(textdata)
    except ValueError:
        if textdata.strip().lower() == 'none':
            return None
        else:
            return textdata


def process_element(element):
    element = process_range(element)

    def numeric(item: str):
        if item.isdigit():
            return int(item)
        return float_(item)

    return [numeric(item) for item in element.split(",")]


def validate_variable(tokens, model, linenum):
    """validates line to be expanded in UserConstraints parameter
    """
    prefix = f"In UserConstraints line no. {linenum}"
    variable_name = tokens[0][0]
    if not isinstance(variable_name, str):
        logger.error(f"{prefix}, first item must be text")
        return False
    if not isinstance(tokens[-1][0], (int, float)):
        logger.error(f"{prefix}, last item must be numeric value")
        return False
    if not hasattr(model, variable_name):
        logger.error(
            f"{prefix}, the first item '{variable_name}' is not a valid model attribute")
        return False
    attrib = getattr(model, variable_name)
    if not isinstance(attrib, Var):
        logger.error(
            f"{prefix}, the first item '{variable_name}' is not a valid model output")
        return False
    if attrib.dim() != len(tokens)-2:
        logger.error(
            f"{prefix}, number of items provided for '{variable_name}' is incorrect")
        return False
    return True


def expand_ALL(tokens, model, linenum):
    """Expands ALL keyword from UserConstraints
    """
    prefix = f"In UserConstraints, line no. {linenum}"
    variable_name = tokens[0][0]
    if not hasattr(model, variable_name):
        logger.error(
            f"{prefix} has invalid variable name {variable_name}")
        return None
    dataframe = pd.DataFrame(getattr(model, variable_name)._data.keys())
    # this is to change column names from integers to text
    # assert len(dataframe.columns) <= 52
    dataframe = dataframe.rename(columns=dict(zip(
        dataframe.columns, string.ascii_uppercase + string.ascii_lowercase)))

    if not check_valid_index(dataframe, tokens[1:-1]):
        logger.error(f"{prefix} has invalid fields")
        return None
    q = make_filter_query(tokens[1:-1], list(dataframe.columns))
    subset = dataframe.query(q)
    if len(subset) == 0:
        logger.error(
            f"In UserConstraints, line {linenum} could not be expanded")
        logger.error(",".join(tokens))
        return None
    subset.insert(loc=0, column='constraint_name', value=variable_name)
    n = len(subset.columns)
    subset.insert(loc=n, column='float_value', value=tokens[-1][0])
    return list(subset.itertuples(index=False, name=None))


def check_valid_index(dataframe, tokens):
    columns = dataframe.columns
    if len(dataframe.columns) != len(tokens):
        return False
    colswithnoALL = {col: item for col, item in zip(
        columns, tokens) if 'ALL' not in item}

    if not colswithnoALL:
        return True
    subset = dataframe[list(colswithnoALL.keys())]
    modeldata = set(subset.itertuples(index=False, name=None))
    # here we expand the tokens, if there are any ranges or
    # comma seperated fields
    expanded = tuple(itertools.product(*list(colswithnoALL.values())))
    return len([items for items in expanded if items in modeldata]) == len(expanded)


def make_filter_query(tokens, columns):
    queries = []
    for item, col in zip(tokens, columns):
        if "ALL" not in item:
            # this or is for all ranges and comma seperated items
            # for any token
            q = " | ".join(f"{col} == '{i}'" if isinstance(
                i, str) else f"{col} == {i}" for i in item)
            queries.append(f"({q})")

    # if the query is empty then
    return " & ".join(queries) or f"{columns[0]} == {columns[0]}"


def parse_user_constraints(filepath, model=None):
    encoding = functionstore.get_encoding(filepath)
    with open(filepath, encoding=encoding) as f:
        reader = csv.reader(f)
        constraints = []
        constraint_rows = []
        for n, line in enumerate(reader, start=1):
            if len(line) >= 1:
                if (line[0].strip().startswith(COMMENT_KEYWORD)):
                    continue
                if (line[0] == CONSTRAINT_END_KEYWORD):
                    constraint = {"BOUNDS": tuple(float_(item) for item in line[1:]),
                                  "VECTORS": constraint_rows}
                    if not constraint_rows:
                        logger.error(
                            f"In UserConstraints, Line {n}, possible consecutive BOUNDS lines")
                    constraint_rows = []
                    constraints.append(constraint)
                else:
                    tokens = [process_element(item) for item in line]

                    if any("ALL" in t for t in tokens):
                        # we skip doing the product here as it will
                        # be taken care by expand_ALL function
                        valid = validate_variable(tokens, model, n)
                        if valid:
                            expanded = expand_ALL(tokens, model, n)
                        if valid and expanded:
                            constraint_rows.extend(expanded)
                        else:
                            constraint_rows.append("ERROR")
                    else:
                        rows = tuple(itertools.product(*tokens))
                        constraint_rows.extend(rows)
        if constraint_rows:
            logger.warning(
                "In UserConstraints, last constraint did not end with BOUNDS")
            logger.warning(
                "In UserConstraints, last constraint will be ignored")
        if [e for c in constraints for e in c['VECTORS'] if e == "ERROR"]:
            raise loaders.LoaderError(
                "There are errors in UserConstraints parameter")

        return constraints


def read_user_constraints(*args, **kwargs):
    filepath = filemanager.find_filepath("UserConstraints")
    return parse_user_constraints(filepath, model=kwargs['model'])


def user_constraints_message():
    if called_from_rumi_validate:
        path = filemanager.find_filepath("UserConstraints")
        if os.path.exists(path) and os.path.isfile(path):
            message = "UserConstraints parameter validation is skipped at this stage, it will be validated before optimization run"
            print(message)
            logger.warning(message)
    return True


def unset_call_from_rumi_validate_flag():
    global called_from_rumi_validate
    called_from_rumi_validate = False


def set_call_from_rumi_validate_flag():
    global called_from_rumi_validate
    called_from_rumi_validate = True


def validate_param(param_name, model):
    specs = filemanager.get_specs(param_name)
    try:
        data = loaders.get_parameter(param_name, model=model)
    except loaders.LoaderError as l:
        logger.error("Validation failed for UserConstraints.")
        logger.exception(l)
        return False
    if data is not None:
        return loaders.validate_param(param_name,
                                      specs,
                                      data,
                                      "rumi.io.supply",
                                      model=model)

    return True


def constraints_loop(code, user_constraints, model=None):
    superscript = {1: "st", 2: "nd", 3: "rd"}
    for i, c in enumerate(user_constraints, start=1):
        try:
            if not eval(code, globals(), locals()):
                logger.error(
                    f"In UserConstraints, possible error in {i}{superscript.get(i, 'th')} constraint")
                return False
        except Exception as e:
            logger.exception(e)
            return False
    return True


def bounds_loop(code, user_constraints, model=None):
    """executes code for each bound for each user constraint
    """
    bounds = {1: "lower", 2: "upper"}
    superscript = {1: "st", 2: "nd", 3: "rd"}

    for i, c in enumerate(user_constraints, start=1):
        for j, b in enumerate(c['BOUNDS'], start=1):
            try:
                if not eval(code, globals(), locals()):
                    logger.error(
                        f"In UserConstraints, possible error in {bounds[j]} bound in {i}{superscript.get(i, 'th')} constraint")
                    logger.error(
                        f"In UserConstraints, possible error in this BOUNDS line BOUNDS, {c['BOUNDS'][0]},{c['BOUNDS'][1]}")
                    return False
            except Exception as e:
                logger.exception(e)
                return False
    return True


def vectors_loop(code, user_constraints, model=None):
    """executes code for each item in tuple for each user constraint
    """
    superscript = {1: "st", 2: "nd", 3: "rd"}

    for i, c in enumerate(user_constraints, start=1):
        for j, v in enumerate(c['VECTORS'], start=1):
            try:
                if not eval(code, globals(), locals()):
                    logger.error(
                        f"In UserConstraints, possible error in {j}{superscript.get(j, 'th')} expanded line provided for {i}{superscript.get(i, 'th')} constraint")
                    logger.error(
                        f"In UserConstraints, possible error in line {v}")

                    return False
            except Exception as e:
                logger.exception(e)
                return False
    return True
