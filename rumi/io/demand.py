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
"""demand io layer. mainly data loading and validation functions.
Function and variable from this module are available in Demand.yml
for validation.
Few function are also used by processing layer.

Important note: Some functions are just imported and not used
are for yaml validation. All definations(including those which are
imported from other modules) from this module are available in
yaml validation.
"""
import functools
import itertools
import os
import logging
from rumi.io import functionstore as fs
from rumi.io import loaders
from rumi.io import filemanager
from rumi.io import config
from rumi.io import constant
from rumi.io import common
from rumi.io import utilities
import pandas as pd
import numpy as np
from rumi.io.common import balancing_area, balancing_time
from rumi.io.utilities import check_consumer_validity
from rumi.io.utilities import check_geographic_validity
from rumi.io.utilities import check_time_validity
from rumi.io.multiprocessutils import execute_in_process_pool
logger = logging.getLogger(__name__)
print = functools.partial(print, flush=True)


def get_consumer_levels(ds):
    """get number of consumer levels defined
    for given demand sector

    Parameters
    ----------
    ds: str
       Demand sector name

    Returns
    -------
       1 or 2
    """
    DS_Cons1_Map = loaders.get_parameter("DS_Cons1_Map", demand_sector=ds)
    if DS_Cons1_Map is None or ds not in DS_Cons1_Map:
        return 0
    type1 = DS_Cons1_Map[ds][-1]
    Cons1_Cons2_Map = loaders.get_parameter(
        "Cons1_Cons2_Map", demand_sector=ds)
    if Cons1_Cons2_Map and Cons1_Cons2_Map.get(type1, None):
        return 2
    return 1


def get_cons_columns(ds):
    """get maximum consumer columns for given demand sector


    Parameters
    -----------
    ds: str
       Demand sector name

    Returns
    -------
       a list of consumer columns for given demand sector
    """

    return list(constant.CONSUMER_TYPES[:get_consumer_levels(ds)])


def get_ds_es_ec_map():
    DS_ES_EC_Map = get_combined(
        "DS_ES_EC_Map")
    return DS_ES_EC_Map.set_index(['DemandSector',
                                   'EnergyService',
                                   'EnergyCarrier'])


def get_ds_es_stc_map():
    DS_ES_STC_DemandGranularityMap = get_combined(
        "DS_ES_STC_DemandGranularityMap")
    return DS_ES_STC_DemandGranularityMap.set_index(['DemandSector',
                                                     'EnergyService',
                                                     'ServiceTechCategory'])


def get_combined(name):
    DS_List = loaders.get_parameter("DS_List")
    dfs = []
    for ds in DS_List:
        combined_map = loaders.get_parameter(name,
                                             demand_sector=ds)
        if combined_map is not None:
            dfs.append(combined_map)

    if dfs and isinstance(dfs[0], pd.DataFrame):
        return pd.concat(dfs).reset_index(drop=True)
    elif dfs and isinstance(dfs[0], list):
        return fs.concat(*dfs)
    elif dfs and isinstance(dfs[0], dict):
        return {k: v for map_ in dfs for k, v in map_.items()}
    else:
        return None


def expand_STC_ST_Map():
    STC_ST_Map = loaders.load_param("STC_ST_Map")
    DS_ES_STC_DemandGranularityMap = get_combined(
        "DS_ES_STC_DemandGranularityMap")
    if fs.is_empty_or_none(DS_ES_STC_DemandGranularityMap):
        return STC_ST_Map
    STC = list(DS_ES_STC_DemandGranularityMap.ServiceTechCategory.unique())
    existing_STC = [row[0] for row in STC_ST_Map]
    return STC_ST_Map + [[stc, stc] for stc in STC if stc not in existing_STC]


def empty_if_None(data):
    if not fs.isnone(data):
        return data
    else:
        return pd.DataFrame()


def get_combined_granularity_map():
    """returns combined granularity map for all bootomup and nonbottomup <DS,ES,EC>
    this ignores NonPhysicalPrimaryCarriers if any
    """
    DS_ES_EC_Map = empty_if_None(get_combined("DS_ES_EC_Map"))
    DS_ES_STC_DemandGranularityMap = empty_if_None(
        get_combined("DS_ES_STC_DemandGranularityMap"))

    newdata = []
    for i in range(len(DS_ES_STC_DemandGranularityMap)):
        row = DS_ES_STC_DemandGranularityMap.iloc[i, :]
        ECs = fs.flatten([ST_to_ECs(ST)
                          for ST in STC_to_STs(row['ServiceTechCategory'])])
        ECs = [EC for EC in ECs if not is_nppc(EC)]

        for EC in ECs:
            d = row.to_dict()
            del d['ServiceTechCategory']
            d['EnergyCarrier'] = EC
            newdata.append(d)

    d_st = pd.DataFrame(newdata)
    collapsed = []
    names = ["DemandSector", "EnergyService", "EnergyCarrier"]
    for values in set([(item['DemandSector'],
                        item['EnergyService'],
                        item['EnergyCarrier']) for item in newdata]):
        q = " & ".join([f"{name} == '{value}'" for name, value in zip(names,
                                                                      values)])
        ds, es, ec = values
        d = d_st.query(q)
        row = dict(zip(names, values))
        row['ConsumerGranularity'] = min(d.ConsumerGranularity.values,
                                         key=lambda x: len(
                                             constant.CONSUMER_COLUMNS[x]))
        row['GeographicGranularity'] = min(d.GeographicGranularity.values,
                                           key=lambda x: len(constant.GEO_COLUMNS[x]))
        row['TimeGranularity'] = min(d.TimeGranularity.values,
                                     key=lambda x: len(constant.TIME_COLUMNS[x]))
        collapsed.append(row)

    return pd.concat([DS_ES_EC_Map, pd.DataFrame(collapsed)])


def get_geographic_granularity(demand_sector,
                               energy_service,
                               energy_carrier):
    """return geographic granuarity of nonbottomup combination
    """
    DS_ES_EC_Map = loaders.get_parameter(
        "DS_ES_EC_Map",
        demand_sector=demand_sector)
    granularity_map = DS_ES_EC_Map.set_index(['DemandSector',
                                              'EnergyService',
                                              'EnergyCarrier'])
    return granularity_map.loc[(demand_sector,
                                energy_service,
                                energy_carrier)]['GeographicGranularity']


def get_type(demand_sector, energy_service):
    """find type of service BOTTOMUP,EXOGENOUS,GDPELASTICITY or RESIDUAL
    """
    DS_ES_Map = loaders.get_parameter('DS_ES_Map',
                                      demand_sector=demand_sector)
    DS_ES_Map = DS_ES_Map.set_index(['DemandSector', 'EnergyService'])
    return DS_ES_Map.loc[(demand_sector, energy_service)]['InputType']


def get_BaseYearDemand(demand_sector):
    """loader function for parameter BaseYearDemand
    """
    return get_demand_sector_parameter('BaseYearDemand',
                                       demand_sector)


def get_DemandElasticity(demand_sector):
    """loader function for parameter DemandElasticity
    """
    return get_demand_sector_parameter('DemandElasticity',
                                       demand_sector)


def get_ExogenousDemand(demand_sector):
    """loader function for parameter ExogenousDemand
    """

    exogenous = get_demand_sector_parameter('ExogenousDemand',
                                            demand_sector)
    return exogenous


def get_ST_SEC(demand_sector):
    """ST_SEC loader function
    """
    return get_demand_sector_parameter("ST_SEC",
                                       demand_sector)


def get_ST_EmissionDetails(demand_sector):
    """ST_EmissionDetails loader function
    """
    return get_demand_sector_parameter("ST_EmissionDetails",
                                       demand_sector)


def get_ResidualDemand(demand_sector):
    """loader function for parameter ResidualDemand
    """
    return get_demand_sector_parameter("ResidualDemand",
                                       demand_sector)


def get_NumConsumers(demand_sector):
    """loader function for parameter NumConsumers
    """
    return get_demand_sector_parameter('NumConsumers',
                                       demand_sector)


def get_NumInstances(demand_sector, energy_service):
    """loader function for parameter NumInstances
    """
    return get_DS_ES_parameter('NumInstances',
                               demand_sector,
                               energy_service)


def get_TechSplitRatio(demand_sector, energy_service):
    """loader function for parameter NumInstances
    """
    return get_DS_ES_parameter('TechSplitRatio',
                               demand_sector,
                               energy_service)


def get_gran_map(param_name):
    """Retrurns granularity_map followed by given parameter
    """
    if param_name in ['BaseYearDemand', 'DemandElasticity', 'ExogenousDemand']:
        g = get_combined("DS_ES_EC_Map")
        if g is not None:
            return g.set_index(['DemandSector', 'EnergyService', 'EnergyCarrier'])
    else:
        g = get_combined("DS_ES_STC_DemandGranularityMap")
        if g is not None:
            return g.set_index(['DemandSector', 'EnergyService', 'ServiceTechCategory'])


def create_empty_dataframe(param_name):
    specs = filemanager.demand_specs()[param_name]
    return pd.DataFrame({c: pd.Series(dtype=d['type']) for c, d in specs['columns'].items()})


def get_all_STC_ST_pairs(demand_sector, energy_service):
    """generates all pairs of STC,ST for given demand_sector and energy_service
    """
    pairs = [(row[0], r) for row in loaders.get_parameter("STC_ST_Map")
             for r in row[1:]]
    q = f"DemandSector=='{demand_sector}' & EnergyService=='{energy_service}'"
    stcs = loaders.get_parameter("DS_ES_STC_DemandGranularityMap",
                                 demand_sector=demand_sector).query(q)[
        'ServiceTechCategory'].values
    return [(STC, ST) for STC, ST in pairs if STC in stcs]


def fill_missing_rows_with_zero_(param_name,
                                 data,
                                 base_dataframe_all=utilities.base_dataframe_all,
                                 entity_values=None,
                                 **kwargs):
    """core function which actually replaces missing rows with zero.
    Approach is to generate full data frame with zero values. then
    override it with data given by user. So automatically rows
    that are missing in user data will become zero.
    """
    conscols = utilities.get_consumer_columns_from_dataframe(data)
    if conscols:
        demand_sector = kwargs['demand_sector']
    else:
        demand_sector = None
    timecols = utilities.get_time_columns_from_dataframe(data)
    geocols = utilities.get_geographic_columns_from_dataframe(data)

    allstructural_cols = conscols+timecols+geocols
    rest_cols = [c for c in data.columns if c not in allstructural_cols]

    column = rest_cols[0]

    base = base_dataframe_all(conscols=conscols,
                              geocols=geocols,
                              timecols=timecols,
                              demand_sector=demand_sector,
                              colname=column,
                              val=0.0).reset_index()
    for c in rest_cols[1:]:
        base[c] = 0.0
    indexcols = conscols + geocols + timecols
    demand_sector = kwargs['demand_sector'] if 'demand_sector' in kwargs else ""
    energy_service = kwargs['energy_service'] if 'energy_service' in kwargs else ""
    return utilities.override_dataframe_with_check(dataframe1=base,
                                                   dataframe2=data,
                                                   index_cols=indexcols,
                                                   param_name=param_name,
                                                   entity_values=entity_values,
                                                   demand_sector=demand_sector,
                                                   energy_service=energy_service)


def fill_missing_rows_with_zero__(param_name,  data, entities,
                                  base_dataframe_all=utilities.base_dataframe_all,
                                  **kwargs):
    """Helper function to fill missing rows
    """
    if entities:
        datai = data.set_index(entities)
        dfs = []
        index = datai.index
        for items in index.unique():
            if not isinstance(items, tuple):
                items = (items,)
            subset = data.query(" & ".join(
                [f"{entity} == '{item}'" for entity, item in zip(index.names, items)]))
            subset = utilities.filter_empty(subset)
            for e, v in zip(index.names, items):
                del subset[e]
            d = fill_missing_rows_with_zero_(param_name,
                                             subset,
                                             base_dataframe_all=base_dataframe_all,
                                             entity_values=items,
                                             **kwargs)
            diff = len(d) - len(subset)
            if diff > 0:
                logger.info(
                    f"Filled {diff} missing rows with zero for {param_name} and {items}")
            for e, v in zip(index.names, items):
                d[e] = v

            dfs.append(d)

        return pd.concat(dfs).reset_index(drop=True)
    else:
        return fill_missing_rows_with_zero_(param_name,
                                            data,
                                            base_dataframe_all=base_dataframe_all,
                                            **kwargs)


def fill_missing_rows_with_zero(param_name, data, **kwargs):
    """For given data fills in zero if some row is missing
    """
    if fs.isnone(data):
        # If parameter does not exists
        return data

    specs = filemanager.get_specs(param_name)
    entities = specs.get('entities', [])
    return fill_missing_rows_with_zero__(param_name, data, entities, **kwargs)


class BaseYearDemandBaseData(utilities.BaseDataFrame):
    """BaseDataFrame with one year less than modelperiod"""

    def get_years(self):
        ModelPeriod = loaders.get_parameter("ModelPeriod")
        return (ModelPeriod.StartYear.iloc[0]-1,)

    def product_time(self):
        if set(self.timecols) == {'Year'}:
            return self.get_years()
        elif set(self.timecols) == {'Year', 'Season'}:
            return list(itertools.product(self.get_years(),
                                          utilities.get_seasons()))
        elif set(self.timecols) == {'Year', 'Season', 'DayType'}:
            return list(itertools.product(self.get_years(),
                                          utilities.get_seasons(),
                                          utilities.get_daytypes()))
        elif set(self.timecols) == {'Year', 'Season', 'DayType', 'DaySlice'}:
            return list(itertools.product(self.get_years(),
                                          utilities.get_seasons(),
                                          utilities.get_daytypes(),
                                          utilities.get_dayslices()))
        else:
            raise utilities.InValidColumnsError("Invalid time slices",
                                                self.timecols)


def fill_missing_rows_with_zero_baseyeardemand(param_name, data, **kwargs):

    def baseyear_base_data(conscols=None,
                           geocols=None,
                           timecols=None,
                           demand_sector=None,
                           colname="dummy",
                           val=0,
                           extracols_df=None):

        return BaseYearDemandBaseData(conscols=conscols,
                                      geocols=geocols,
                                      timecols=timecols,
                                      demand_sector=demand_sector,
                                      colname=colname,
                                      val=val,
                                      extracols_df=None).get_dataframe()

    if fs.isnone(data):
        # If parameter does not exists
        return data

    specs = filemanager.get_specs(param_name)
    entities = specs.get('entities', [])
    return fill_missing_rows_with_zero__(param_name,
                                         data,
                                         entities,
                                         base_dataframe_all=baseyear_base_data,
                                         **kwargs)


def fill_missing_rows_with_zero_gtprofile(param_name, data: pd.DataFrame, **kwargs):
    if fs.isnone(data):
        return data

    if "ServiceTech" in data.columns:
        entities = ['ServiceTech']
        data['ServiceTech'] = data['ServiceTech'].fillna("EMPTY_SERVICE_TECH")
        df = fill_missing_rows_with_zero__(
            param_name, data, entities, **kwargs)
        df['ServiceTech'] = df.ServiceTech.apply(
            lambda x: None if x == "EMPTY_SERVICE_TECH" else x)
        return df
    else:
        entities = []
        return fill_missing_rows_with_zero__(param_name, data, entities, **kwargs)


def empty_data(param_name,
               data,
               demand_sector,
               energy_service):
    """creates empty dataframe if relevant else returns none data
    """
    if fs.isnone(data):
        if not is_bottomup(demand_sector, energy_service):
            return data
        else:
            return create_empty_dataframe(param_name)
    else:
        return data


def get_entity_values_dataframe_for_ds(param_name, demand_sector):
    ds_es_map = loaders.get_parameter("DS_ES_Map",
                                      demand_sector=demand_sector).query(
        f"DemandSector=='{demand_sector}'")
    dfs = []
    for ds, es in ds_es_map[['DemandSector', 'EnergyService']].values:
        entities_df = get_entity_values_dataframe(param_name, ds, es)
        dfs.append(entities_df)

    return pd.concat(dfs)


def check_extra_rows(data,
                     param_name,
                     demand_sector,
                     energy_service):
    """While applying fill_missing_TechSplitRatio, if there are some invalid
    combinations of STC,ST are given in data. This will capture that and add
    logger message.
    """
    entities_df = get_entity_values_dataframe_for_ds(param_name, demand_sector)
    check_extra_rows_(data,
                      entities_df,
                      param_name,
                      demand_sector,
                      energy_service)


def check_extra_rows_for_es(data,
                            param_name,
                            demand_sector,
                            energy_service):
    entities_df = get_entity_values_dataframe(param_name,
                                              demand_sector,
                                              energy_service)
    check_extra_rows_(data,
                      entities_df,
                      param_name,
                      demand_sector,
                      energy_service)


def check_extra_rows_(data,
                      entities_df,
                      param_name,
                      demand_sector,
                      energy_service):
    """While applying fill_missing_TechSplitRatio, if there are some invalid
    combinations of STC,ST are given in data. This will capture that and add
    logger message.
    """
    if fs.isnone(data) or fs.isnone(entities_df):
        return
    columns = list(entities_df.columns)
    diff = set(pd.MultiIndex.from_frame(data[columns]).unique(
    )) - set(pd.MultiIndex.from_frame(entities_df).unique())
    if len(diff) > 0:
        for items in diff:
            logger.warning(
                f"{param_name} from {demand_sector}, {energy_service} had unexpected combination of {columns} as {items}. Those rows are ignored.")


def get_entity_values_dataframe(param_name, demand_sector, energy_service):
    """get expected dataframe of entities for given parameter
    entity names are defines in parameter name. All entities together
    form a single snapshot of data for all C*G*T* combinations.

    This function implecitly knows what are entities for given parameter
    """
    if param_name in ["EfficiencyLevelSplit", "NumInstances"]:
        # entities -> ['ServiceTech']
        STs = get_STs(demand_sector, energy_service)
        return pd.MultiIndex.from_tuples([(ST,) for ST in STs], names=['ServiceTech']).to_frame().reset_index(drop=True)
    elif param_name == "TechSplitRatio":
        # entities -> ['ServiceTechCategory', 'ServiceTech']
        pairs = get_all_STC_ST_pairs(demand_sector, energy_service)
        return pd.MultiIndex.from_tuples(pairs, names=[
            'ServiceTechCategory', 'ServiceTech']).to_frame().reset_index(drop=True)


def get_granularity(param_name: str, **kwargs) -> tuple:
    """kwargs are optional key value arguments which can have only following names
    demand_sector,energy_service, energy_carrier or service_tech_category
    """
    if param_name in ['BaseYearDemand', 'DemandElasticity', 'ExogenousDemand', 'ResidualDemand']:
        index_ = ['DemandSector', 'EnergyService', 'EnergyCarrier']
        g = loaders.get_parameter("DS_ES_EC_Map",
                                  demand_sector=kwargs['demand_sector'])
    else:
        index_ = ["DemandSector", "EnergyService", "ServiceTechCategory"]
        g = loaders.get_parameter("DS_ES_STC_DemandGranularityMap",
                                  demand_sector=kwargs['demand_sector'])
    names = {"DemandSector": "demand_sector",
             "EnergyService": "energy_service",
             "EnergyCarrier": "energy_carrier",
             "ServiceTechCategory": "service_tech_category"}

    gran_map = g.set_index(index_)

    key = tuple(kwargs[names[i]] for i in index_)
    gran = gran_map.loc[key]
    C = gran['ConsumerGranularity']
    G = gran['GeographicGranularity']
    T = gran['TimeGranularity']

    return C, G, T


def ignore_nonphysicalprimarycarriers(param_name,
                                      data,
                                      demand_sector=None):
    nppc = loaders.get_parameter("NonPhysicalPrimaryCarriers")

    if fs.isnone(nppc) or fs.isnone(data):
        return data

    carriers = nppc.EnergyCarrier.values
    ignore = data.query(" | ".join(f"EnergyCarrier =='{i}'" for i in carriers))
    carriers = ignore.EnergyCarrier.unique()
    if len(carriers) > 0:
        msg = f"In ST_SEC from {demand_sector}, " + \
            f"ignoring rows for following EnergyCarriers {carriers}"
        logger.warning(msg)

        q = " & ".join(f"EnergyCarrier !='{i}'" for i in carriers)
        return data.query(q)
    else:
        return data


def fill_missing_split(param_name=None,
                       data=None,
                       demand_sector=None,
                       energy_service=None):
    """fills missing rows from EfficiencyLevelSplit/TechSplitRatio/NumInstances
    """
    data = empty_data(param_name, data, demand_sector, energy_service)
    if fs.isnone(data):
        return data

    logger.debug(
        f"fill_missing_split for {demand_sector}, {energy_service}")
    entities_df = get_entity_values_dataframe(
        param_name, demand_sector, energy_service)
    names = list(entities_df.columns)
    check_extra_rows(data,
                     param_name,
                     demand_sector,
                     energy_service)
    dfs = []

    for r in range(len(entities_df)):
        items = [entities_df.iloc[r, j] for j in range(len(names))]
        colsdict = dict(zip(names, items))
        query = " & ".join([f"{name}=='{value}'" for name,
                            value in colsdict.items()])
        extracols_df = entities_df.query(query).reset_index(drop=True)
        service_tech_category = colsdict.get("ServiceTechCategory",
                                             ST_to_STC(colsdict['ServiceTech']))
        C, G, T = "CONSUMERALL", "MODELGEOGRAPHY", "YEAR"
        conscols = utilities.get_consumer_columns(C)
        geocols = utilities.get_geographic_columns(G)
        timecols = utilities.get_time_columns(T)

        if len(data) > 0:
            subset = utilities.filter_empty(data.query(query))
            if not fs.is_empty_or_none(subset):
                conscols = utilities.get_consumer_columns_from_dataframe(
                    subset)
                geocols = utilities.get_geographic_columns_from_dataframe(
                    subset)
                timecols = utilities.get_time_columns_from_dataframe(subset)

        val, extracols_df, base = get_base_data(param_name,
                                                demand_sector,
                                                energy_service,
                                                extracols_df,
                                                conscols,
                                                geocols,
                                                timecols)
        if len(extracols_df) == 0:
            logger.warning(
                f"Unable to fill missing values for {param_name} from {demand_sector}, {energy_service} for {items}")
            continue

        indexcols = conscols + geocols + timecols + list(extracols_df.columns)

        if len(data) > 0:
            # subset = utilities.filter_empty(data.query(query))
            d = utilities.override_dataframe_with_check(dataframe1=base,
                                                        dataframe2=subset,
                                                        index_cols=indexcols,
                                                        param_name=param_name,
                                                        demand_sector=demand_sector,
                                                        energy_service=energy_service)
            diff = len(d) - len(subset)
            if diff > 0:
                logger.info(
                    f"Filled {diff} missing rows with {val} for {param_name} from {demand_sector}, {energy_service} in {items}")
        else:
            d = base
            logger.info(
                f"Filled {len(d)} rows with {val} for {param_name} from {demand_sector} , {energy_service} in {items}")

        dfs.append(d)
    return pd.concat(dfs).reset_index(drop=True)


def get_base_data(param_name: str,
                  demand_sector: str,
                  energy_service: str,
                  extracols_df: pd.DataFrame,
                  conscols: list,
                  geocols: list,
                  timecols: list
                  ):
    if param_name == "TechSplitRatio":
        colname = param_name
        service_tech_category = extracols_df['ServiceTechCategory'].iloc[0]
        service_techs = STC_to_STs(service_tech_category)
        val = 1.0 if len(service_techs) == 1 else 0.0
    elif param_name == "NumInstances":
        colname = param_name
        val = 1.0
    else:
        colname = "EfficiencySplitShare"
        service_tech = extracols_df['ServiceTech'].iloc[0]
        efficiency_levels = get_efficiency_levels(demand_sector,
                                                  service_tech)

        val = 1.0 if len(efficiency_levels) == 1 else 0.0
        levels = pd.Series(efficiency_levels, name="EfficiencyLevelName")
        extracols_df = fs.expand_with(
            extracols_df, levels)

    return val, extracols_df, utilities.base_dataframe_all(conscols=conscols,
                                                           geocols=geocols,
                                                           timecols=timecols,
                                                           demand_sector=demand_sector,
                                                           colname=colname,
                                                           val=val,
                                                           extracols_df=extracols_df).reset_index()


def get_efficiency_levels(demand_sector: str,
                          service_tech: str) -> pd.Series:
    ST_SEC = loaders.get_parameter("ST_SEC",
                                   demand_sector=demand_sector)
    st_sec = ST_SEC.query(f"ServiceTech == '{service_tech}'")
    return st_sec['EfficiencyLevelName'].drop_duplicates()


def get_EfficiencyLevelSplit(demand_sector, energy_service):
    """loader function for parameter EfficiencyLevelSplit
    """
    return get_DS_ES_parameter('EfficiencyLevelSplit',
                               demand_sector,
                               energy_service)


def get_ES_Demand(demand_sector,
                  energy_service,
                  service_tech_category):
    """loader function for parameter ES_Demand
    should not be used directly. use loaders.get_parameter instead.
    """
    prefix = f"{service_tech_category}_"
    filepath = find_custom_DS_ES_filepath(demand_sector,
                                          energy_service,
                                          'ES_Demand',
                                          prefix)
    logger.debug(f"Reading {prefix}ES_Demand from file {filepath}")
    return pd.read_csv(filepath)


def extract_STCs(filename):
    return filename.split(constant.ST_SEPARATOR_CHAR)[:-1]


def find_usage_penetrations(folder, STC_combination):
    if os.path.exists(folder):
        return [f for f in os.listdir(folder) if f.endswith("UsagePenetration.csv") and set(extract_STCs(f)) == set(STC_combination)]
    else:
        return []


def get_UsagePenetration(demand_sector,
                         energy_service,
                         STC_combination):
    """loader function for parameter UsagePenetration
    """
    prefix = constant.ST_SEPARATOR_CHAR.join(
        STC_combination) + constant.ST_SEPARATOR_CHAR

    globalpath = os.path.dirname(find_custom_DS_ES_filepath(demand_sector,
                                                            energy_service,
                                                            "UsagePenetration",
                                                            ""))
    scenario = config.get_config_value("scenario")
    scenariopath = globalpath.replace("Default Data",
                                      os.path.join("Scenarios",
                                                   scenario))
    files = {"path": scenariopath,
             "files": find_usage_penetrations(scenariopath, STC_combination)}
    if not files['files']:
        gfiles = find_usage_penetrations(globalpath, STC_combination)
        if not gfiles:
            logger.debug(f"UsagePenetration file for {prefix} not found")
            return
        else:
            files = {"path": globalpath,
                     "files": gfiles}

    if len(files['files']) > 1:
        logger.error(f"UsagePenetration for {prefix} has multiple files")
    else:
        filepath = os.path.join(files['path'], files['files'][0])
        if os.path.exists(filepath):
            logger.debug(f"Reading {prefix} from file {filepath}")
            return pd.read_csv(filepath)


def get_demand_sector_parameter(param_name,
                                demand_sector):
    """loads demand sector parameter which lies inside demand_sector folder
    """
    filepath = find_custom_demand_path(demand_sector, param_name)
    logger.debug(f"Reading {param_name} from file {filepath}")
    cols = list(filemanager.demand_specs()[param_name]['columns'].keys())
    d = pd.read_csv(filepath)
    return d[[c for c in cols if c in d.columns]]


def get_default_file(param_name,
                     demand_sector,
                     energy_service,
                     prefix=""):
    """returns the file for same DS/ES parameter in DS folder
    """
    return find_custom_demand_path(demand_sector,
                                   param_name,
                                   energy_service="",
                                   prefix=prefix)


def get_overridden_file(param_name,
                        demand_sector,
                        energy_service,
                        prefix=""):
    """returns the file for DS/ES parameter in DS/ES folder
    """
    return find_custom_DS_ES_filepath(demand_sector,
                                      energy_service,
                                      param_name,
                                      prefix)


def get_DS_ES_parameter(param_name,
                        demand_sector,
                        energy_service,
                        prefix=""):
    """loads parameter which is inside demand_sector/energy_service folder
    """
    filepath = get_overridden_file(
        param_name, demand_sector, energy_service, prefix)
    specs = filemanager.demand_specs()[param_name]
    cols = list(specs['columns'].keys())
    if specs.get('override', False):
        default_file = get_default_file(
            param_name, demand_sector, energy_service, prefix)
        if os.path.exists(default_file):
            logger.debug(f"Reading {param_name} from file {default_file}")
            default_df = pd.read_csv(default_file)
            default_df = default_df[[
                c for c in cols if c in default_df.columns]]
            if os.path.exists(filepath):
                over_df = pd.read_csv(filepath)
                over_df = over_df[[c for c in cols if c in over_df.columns]]
                check_extra_rows_for_es(over_df,
                                        param_name,
                                        demand_sector,
                                        energy_service)

                entities = specs.get('entities', [])
                if entities:
                    d = override_entitywise(param_name,
                                            demand_sector,
                                            energy_service,
                                            default_df,
                                            over_df,
                                            entities)
                else:
                    d = override(param_name,
                                 demand_sector,
                                 energy_service,
                                 default_df,
                                 over_df)
            else:
                d = default_df
        else:
            logger.debug(f"Reading {param_name} from file {filepath}")
            d = pd.read_csv(filepath)
    else:
        logger.debug(f"Reading {param_name} from file {filepath}")
        d = pd.read_csv(filepath)
    return d[[c for c in cols if c in d.columns]]


def override(param_name,
             demand_sector,
             energy_service,
             default_df,
             over_df):
    """override without entities work on whole dataframe to overide
    """
    indexcols = utilities.get_all_structure_columns(default_df)
    indexcols_ = utilities.get_all_structure_columns(over_df)

    if len(indexcols_) < len(indexcols):
        message = f"{param_name} in {demand_sector},{energy_service} is trying to override with coarser granuarity!"
        logger.error(message)
        raise DemandValidationError(message)

    if len(indexcols_) > len(indexcols):
        base = utilities.base_dataframe(indexcols_,
                                        demand_sector=demand_sector).reset_index()
        del base['dummy']
        default_df = pd.merge(base, default_df, on=indexcols)

    return fs.override_dataframe(default_df,
                                 over_df,
                                 index_cols=indexcols_)


def override_entitywise(param_name,
                        demand_sector,
                        energy_service,
                        default_df,
                        over_df,
                        entities):
    """overriding will work on subframes (created by querying for given entities).
    and then combine it together by concatenating
    """
    try:
        entity_data = pd.concat([default_df[entities],
                                 over_df[entities]]).drop_duplicates()
    except KeyError as ke:
        logger.warning(
            f"For {param_name}, columns {entities} are not present in default or over-ridden data")
        logger.warning(
            f"For {param_name}, {entities} taking overridden data as it is.")
        return over_df

    dfs = []
    for r in range(len(entity_data)):
        item = [entity_data.iloc[r, j] for j in range(len(entities))]
        q = " & ".join([f"{e} == '{value}'" for e,
                       value in zip(entities, item)])
        default = utilities.filter_empty(default_df.query(q))
        default = expand_to_approp_gran(param_name,
                                        entities,
                                        demand_sector,
                                        energy_service,
                                        default)

        over = utilities.filter_empty(over_df.query(q))

        defaultcols = utilities.get_all_structure_columns(
            default)
        overcols = utilities.get_all_structure_columns(
            over)

        if len(overcols) == 0:
            dfs.append(default)
        elif len(overcols) >= len(defaultcols):
            if not entities:
                extracols_df = None
            else:
                extracols_df = pd.DataFrame(
                    dict(zip(entities, [[i] for i in item])))
            base = utilities.base_dataframe(utilities.get_all_structure_columns(over),
                                            demand_sector=demand_sector,
                                            colname=param_name,
                                            val=0,
                                            extracols_df=extracols_df).reset_index()
            if len(default) == 0:
                default = base
            else:
                del base[param_name]
                default = pd.merge(default, base)

            d = fs.override_dataframe(default,
                                      over,
                                      index_cols=overcols+entities)
            dfs.append(d)
        else:
            message = f"{param_name} in {demand_sector},{energy_service} for {item} is trying to override with coarser granuarity!"
            logger.error(message)
            raise DemandValidationError(message)

    return pd.concat(dfs).reset_index()


def get_entity_values(entities, data):
    """get ec, st, stc from data if it is in entities
    """
    energy_carrier = None
    service_tech_category = None
    service_tech = None

    if 'ServiceTechCategory' in entities:
        service_tech_category = data['ServiceTechCategory'].iloc[0]
    elif "ServiceTech" in entities:
        service_tech = data["ServiceTech"].iloc[0]
        service_tech_category = ST_to_STC(service_tech)
    elif 'EnergyCarrier' in entities:
        energy_carrier = data["EnergyCarrier"].iloc[0]

    return energy_carrier, service_tech, service_tech_category


def get_expected_gran_cols(param_name,
                           entities,
                           demand_sector,
                           energy_service,
                           data):
    """gets expected granularity columns for given param_name
    corresponding ServiceTech,ServiceTechCategory or EnergyCarrier
    will be picked up from data passed
    """
    energy_carrier, service_tech, service_tech_category = get_entity_values(entities,
                                                                            data)
    C, G, T = get_granularity(param_name,
                              demand_sector=demand_sector,
                              energy_service=energy_service,
                              energy_carrier=energy_carrier,
                              service_tech_category=service_tech_category)
    C = utilities.get_consumer_columns(C)
    G = utilities.get_geographic_columns(G)
    T = utilities.get_time_columns(T)
    return C, G, T


def expand_to_approp_gran(param_name: str,
                          entities: list,
                          demand_sector: str,
                          energy_service: str,
                          data: pd.DataFrame):
    """If data is coarser than expected granularity then
    expand the data to expected granularity. Same values will
    be used for finer granularity
    """
    if fs.isnone(data) or len(data) == 0:
        return data

    try:
        C, G, T = get_expected_gran_cols(
            param_name, entities, demand_sector, energy_service, data)
    except KeyError as k:
        logger.debug("Unable to find granuarity for " + str(k))
        return data

    cols_data = utilities.get_all_structure_columns(data)
    if set(C+G+T) - set(cols_data):
        extracols_df = pd.DataFrame({e: [data[e].iloc[0]] for e in entities})
        base = utilities.base_dataframe_all(conscols=C,
                                            geocols=G,
                                            timecols=T,
                                            demand_sector=demand_sector,
                                            colname="XXX",
                                            extracols_df=extracols_df)
        base = base.reset_index()
        del base['XXX']
        df = base.merge(data, how="left")
        return df
    else:
        return data


def get_demand_granularity(demand_sector,
                           energy_service,
                           energy_carrier=None,
                           service_tech_category=None):
    params = {"BOTTOMUP": "ES_Demand",
              "GDPELASTICITY": "BaseYearDemand",
              "RESIDUAL": "ResidualDemand",
              "EXOGENOUS": "ExogenousDemand"}

    param_name = params[get_type(demand_sector, energy_service)]

    return get_granularity(param_name,
                           demand_sector=demand_sector,
                           energy_service=energy_service,
                           energy_carrier=energy_carrier,
                           service_tech_category=service_tech_category)


def check_coarser_sum(GTProfile,
                      demand_sector=None,
                      energy_service=None,
                      energy_carrier=None):
    """while applying demand profile to avoid divide by zero error
    it is ncessary to make sure that the denominator is non zero.
    this check is actully assuring the denominator term to be nonzero.
    """
    def valid_gtprofile(subset,
                        ST=None):
        if fs.isnone(subset):
            return True

        service_tech_category = ST_to_STC(ST) if ST else None
        C, G, T = get_demand_granularity(demand_sector,
                                         energy_service,
                                         energy_carrier=energy_carrier,
                                         service_tech_category=service_tech_category)
        grancols = list(constant.TIME_COLUMNS[T] + constant.GEO_COLUMNS[G])
        gtotal = subset.groupby(grancols).sum(
            numeric_only=True)['GTProfile']
        v = len(gtotal[gtotal <= 1e-6]) == 0

        if not v:
            entity = ST or energy_carrier
            logger.error(
                f"In {energy_carrier}_GTProfile for {entity} in {demand_sector},{energy_service} should not sum to zero at granularity {T},{G}")

        return v

    valid = True
    if is_bottomup(demand_sector, energy_service):
        if 'ServiceTech' not in GTProfile.columns:
            GTProfile = GTProfile.copy()
            GTProfile['ServiceTech'] = np.nan

        STs = set(get_STs(demand_sector, energy_service))

        for ST in STs:
            if ST in set(GTProfile.ServiceTech.unique()):
                subset = utilities.filter_empty(
                    GTProfile.query(f"ServiceTech == '{ST}'"))
            elif GTProfile.ServiceTech.isnull().sum() > 0:
                subset = GTProfile[GTProfile.ServiceTech.isna()]
            else:
                subset = None

            valid = valid and valid_gtprofile(subset, ST=ST)
    else:
        valid = valid_gtprofile(GTProfile)

    return valid


def find_custom_DS_ES_filepath(demand_sector,
                               energy_service,
                               name,
                               prefix):
    """find actual location of data in case some data lies in scenario
    """
    return find_custom_demand_path(demand_sector,
                                   name,
                                   energy_service,
                                   prefix)


def find_custom_demand_path(demand_sector,
                            name,
                            energy_service="",
                            prefix=""):
    """find actual location of data in case some data lies in scenario
    """
    return filemanager.find_filepath(name,
                                     demand_sector,
                                     energy_service,
                                     fileprefix=prefix)


def get_GTProfile(demand_sector,
                  energy_service,
                  energy_carrier):
    return get_DS_ES_parameter("GTProfile",
                               demand_sector,
                               energy_service,
                               f"{energy_carrier}_")


def get_all_ds_es_ec():
    nppc = loaders.get_parameter("NonPhysicalPrimaryCarriers")
    dee = get_all_ds_es_ec_()

    if fs.isnone(nppc):
        return dee

    return [(ds, es, ec) for ds, es, ec in dee if ec not in nppc.EnergyCarrier.values]


def get_all_ds_es_ec_():
    """return all possible combinations of ds,es,ec in the model
    """
    DS_ES_EC_Map = get_combined("DS_ES_EC_Map")
    DS_ES_STC_DemandGranularityMap = get_combined(
        "DS_ES_STC_DemandGranularityMap")
    cols = ["DemandSector",
            "EnergyService",
            "EnergyCarrier"]

    if not fs.isnone(DS_ES_EC_Map):
        ds_es_ec = list(zip(*listcols(DS_ES_EC_Map[cols])))
    else:
        ds_es_ec = []

    if fs.isnone(DS_ES_STC_DemandGranularityMap):
        # if no BOTTOMUP services are given
        return ds_es_ec

    for ds, es, stc in zip(*listcols(DS_ES_STC_DemandGranularityMap[cols[:2] + ['ServiceTechCategory']])):
        ECs = fs.flatten([ST_to_ECs(st) for st in STC_to_STs(stc)])
        for ec in ECs:
            if (ds, es, ec) not in ds_es_ec:
                ds_es_ec.append((ds, es, ec))

    return ds_es_ec


def get_RESIDUAL_ECs(DS_ES_Map, DS_ES_EC_Map):
    df = DS_ES_Map.query("InputType == 'RESIDUAL'")[
        ['DemandSector', 'EnergyService']]
    DS_ES = zip(df['DemandSector'], df['EnergyService'])
    if isinstance(DS_ES_EC_Map, type(None)):
        ECs = dict()
    else:
        ECs = {(DS, ES): list(DS_ES_EC_Map.query(f"DemandSector == '{DS}' & EnergyService == '{ES}'")[
            'EnergyCarrier'].values) for DS, ES in DS_ES}

    return ECs


def derive_ES_EC(demand_sector, input_type):
    """return set of ES,EC combinations for given demand_sector and input_type but not_BOTTOMUP
    """
    DS_ES_Map = loaders.get_parameter('DS_ES_Map',
                                      demand_sector=demand_sector)
    DS_ES_EC_Map = loaders.get_parameter('DS_ES_EC_Map',
                                         demand_sector=demand_sector)
    subset = DS_ES_EC_Map.query(f"DemandSector == '{demand_sector}'")
    es_ec = subset[['EnergyService', 'EnergyCarrier']].values
    return [(es, ec) for es, ec in es_ec if len(DS_ES_Map.query(f"DemandSector=='{demand_sector}' & EnergyService=='{es}' & InputType=='{input_type}'")) > 0]


def check_RESIDUAL_EC(DS_ES_Map, DS_ES_EC_Map, DS_ES_STC_DemandGranularityMap):
    """Each EC specified for a <DS, ES> combination,
       whose InputType in DS_ES_Map is RESIDUAL,
       must occur at least once in another
       <DS, ES> combination for the same DS
    """
    def x_in_y(x, y):
        return any([ix in y for ix in x])

    ECS = get_RESIDUAL_ECs(DS_ES_Map, DS_ES_EC_Map)
    items1 = []
    if not fs.isnone(DS_ES_EC_Map):
        for DS, ES in ECS:
            ec = DS_ES_EC_Map.query(
                f'DemandSector == "{DS}" & EnergyService != "{ES}"')['EnergyCarrier']
            if x_in_y(ECS[(DS, ES)], ec.values):
                items1.append(ec)

    DS_ES_STC = DS_ES_STC_DemandGranularityMap
    ST_EC_Map = loaders.get_parameter('ST_EC_Map')
    if len(items1) == 0 and ECS and not fs.isnone(DS_ES_STC_DemandGranularityMap) and ST_EC_Map:
        count = 0
        for key, ECs in ECS.items():
            DS, ES = key
            for EC in ECs:
                STCS = [ST_to_STC(row[0])
                        for row in ST_EC_Map if EC in row[1:]]
                for STC in STCS:
                    count += len(DS_ES_STC.query(
                        f"DemandSector == '{DS}' & EnergyService != '{ES}' & ServiceTechCategory == '{STC}'"))

    return not ECS or len(items1) > 0 or count > 0


def ST_to_STC(ST):
    STC_ST_Map = loaders.get_parameter("STC_ST_Map")
    if not STC_ST_Map:
        return ST
    search = [row[0] for row in STC_ST_Map if ST in row[1:]]
    if search:
        return search[0]
    else:
        return ST


def are_BOTTOMUP(DS_ES_X_Map, DS_ES_Map):
    DS_ES = DS_ES_X_Map.DemandSector, DS_ES_X_Map.EnergyService
    df = fs.combined_key_subset(DS_ES, DS_ES_Map).query(
        "InputType != 'BOTTOMUP'")
    return len(df) == 0


def not_BOTTOMUP(DS_ES_X_Map, DS_ES_Map):
    DS_ES = DS_ES_X_Map.DemandSector, DS_ES_X_Map.EnergyService
    df = fs.combined_key_subset(DS_ES, DS_ES_Map).query(
        "InputType == 'BOTTOMUP'")
    return len(df) == 0


def check_ALL_DS(DS_ES_X_Map):
    """
    ES,entity used with ALL as DS can not be used with any other DS.
    This function checks if this is true.

    entity-> EnergyCarrier for DS_ES_EC_Map
    entity-> ServiceTechCategory for DS_ES_STC_DemandGranularityMap
    """
    if DS_ES_X_Map == "DS_ES_EC_Map":
        entity = "EnergyCarrier"
    else:
        entity = 'ServiceTechCategory'
    X_Map = loaders.load_param(DS_ES_X_Map).to_dict(orient='records')
    ES_EC_with_ALL = [(row['EnergyService'], row[entity])
                      for row in X_Map if row['DemandSector'] == "ALL"]
    ES_EC_without_ALL = [(row['EnergyService'], row[entity]) for row in X_Map if row['DemandSector']
                         != "ALL" and (row['EnergyService'], row[entity]) in ES_EC_with_ALL]
    return len(ES_EC_without_ALL) == 0


def listcols(df):
    return [df[c] for c in df.columns]


def expand_DS_ALL(BOTTOMUP, demand_sector=None):
    """
    Expands Map when DS is ALL
    """
    if BOTTOMUP:
        cond = "=="
        data = loaders.load_param("DS_ES_STC_DemandGranularityMap",
                                  demand_sector=demand_sector)
    else:
        data = loaders.load_param("DS_ES_EC_Map",
                                  demand_sector=demand_sector)
        cond = "!="

    if fs.isnone(data):
        return None

    DS_ES_Map = loaders.get_parameter("DS_ES_Map", demand_sector=demand_sector)
    ESs = [row for row in data.values if row[0] == 'ALL']

    data_ = data.query("DemandSector != 'ALL'")
    data_all = data.query("DemandSector == 'ALL'")
    expanded_rows = []
    colnames = list(data_all.columns)
    for item in data_all.index:
        ES = data_all.loc[item]['EnergyService']
        nonbottomup = DS_ES_Map.query(
            f"EnergyService == '{ES}' & InputType {cond} 'BOTTOMUP'")
        if len(nonbottomup) > 0:
            ds = nonbottomup['DemandSector']
            for eachds in ds:
                values = data_all.loc[item].copy()
                values[colnames.index('DemandSector')] = eachds
                expanded_rows.append(dict(zip(colnames, values)))
    return pd.concat([data_, pd.DataFrame(expanded_rows)]).reset_index(drop=True)


def expand_DS_ES_EC(demand_sector=None):
    return get_combined("DS_ES_EC_Map")


def expand_DS_ES_STC():
    return get_combined("DS_ES_STC_DemandGranularityMap")


def is_valid(DS, EC):
    DS_ES_EC_Map = loaders.load_param("DS_ES_EC_Map")
    DS_ES_STC_DemandGranularityMap = loaders.load_param(
        "DS_ES_STC_DemandGranularityMap")
    ST_EC_Map = loaders.get_parameter("ST_EC_Map")
    ECS = list(DS_ES_EC_Map.query(f"DemandSector == '{DS}'")[
        'EnergyCarrier'].values)
    STS = [row[0] for row in ST_EC_Map if EC in row[1:]]
    DSS = [DS_ES_STC_DemandGranularityMap.query(f"ServiceTech == {ST}")[
        'DemandSector'].values for row in DS_ES_STC_DemandGranularityMap for ST in STS]

    return ECS or DS in DSS


def ST_to_EC(ST):
    return ST_to_ECs(ST)[0]


def ST_to_ECs(ST):
    ST_EC_Map = loaders.get_parameter("ST_EC_Map")
    return fs.flatten([row[1:] for row in ST_EC_Map if row[0] == ST])


def get_service_techs(demand_sector,
                      energy_service,
                      energy_carrier):
    """ServiceTechs for given <demand_sector,energy_service, energy_carrier>
    combination
    """
    DS_ES_STC_DemandGranularityMap = loaders.get_parameter(
        "DS_ES_STC_DemandGranularityMap",
        demand_sector=demand_sector)
    if DS_ES_STC_DemandGranularityMap is None or DS_ES_STC_DemandGranularityMap.size == 0:
        return tuple()
    STCs = DS_ES_STC_DemandGranularityMap.query(f"DemandSector == '{demand_sector}' & EnergyService == '{energy_service}'")[
        'ServiceTechCategory'].values

    ST1 = fs.flatten([STC_to_STs(STC) for STC in STCs])

    ST2 = EC_to_STs(energy_carrier)

    return tuple(set(ST1) & set(ST2))


def get_service_tech_categories(demand_sector,
                                energy_service):
    """ServiceTechCategories for given <demand_sector,energy_service>
    combination
    """
    DS_ES_STC_DemandGranularityMap = loaders.get_parameter(
        "DS_ES_STC_DemandGranularityMap",
        demand_sector=demand_sector)
    STCs = DS_ES_STC_DemandGranularityMap.query(f"DemandSector == '{demand_sector}' & EnergyService == '{energy_service}'")[
        'ServiceTechCategory'].values

    return STCs


def EC_to_ST(energy_carrier):
    return EC_to_STs(energy_carrier)[0]


def EC_to_STs(energy_carrier):
    ST_EC_Map = loaders.get_parameter("ST_EC_Map")
    return [row[0] for row in ST_EC_Map if energy_carrier in row[1:]]


def coarser(x, y, values):
    return values.index(x) <= values.index(y)


def finer(x, y, values):
    return values.index(x) >= values.index(y)


"""
def check_granularity(GRANULARITY):

    DS_ES_EC_Map = loaders.get_parameter("DS_ES_EC_Map")
    DS_ES_Map = loaders.get_parameter("DS_ES_Map")

    def get_Granularity(DS, ES, EC):
        df = DS_ES_EC_Map.query(
            f"(DemandSector == '{DS}') & (EnergyService =='{ES}') & (EnergyCarrier == '{EC}')")
        return df[GRANULARITY].iloc[0] if len(df) != 0 else None

    def get_input_type(DS, ES):
        return DS_ES_Map.query(f"DemandSector == '{DS}' & EnergyService == '{ES}'")['InputType'].iloc[0]

    DS_ES_EC = list(
        zip(*listcols(DS_ES_EC_Map[['DemandSector', 'EnergyService', 'EnergyCarrier']])))
    type_ = {(DS, ES, EC): get_input_type(DS, ES)
             for DS, ES, EC in DS_ES_EC}

    type_RESIDUAL = {item: type_[item]
                     for item in type_ if type_[item] == 'RESIDUAL'}
    t_gran = {item: get_Granularity(*item)
              for item in DS_ES_EC}
    t_gran_RES = {item: get_Granularity(*item)
                  for item in DS_ES_EC if item in type_RESIDUAL}

    if GRANULARITY == "TimeGranularity":
        t_values = [t.upper() for t in constant.TIME_COLUMNS]
    else:
        t_values = [g.upper() for g in constant.GEO_COLUMNS]

    # here condition is list comprehension is filtering out
    # rows with DS==DS_under_consideration and EC=EC_under_consideration
    return all([(t_values.index(t_gran_RES[ritem]) <=
                 t_values.index(t_gran[item]))
                for ritem in t_gran_RES
                for item in t_gran if item[0] == ritem[0] and item[2] == ritem[2]])
"""


class DemandValidationError(Exception):
    pass


@functools.lru_cache(maxsize=None)
def derive_ECs(DS):
    DS_ES_STC_DemandGranularityMap = expand_DS_ES_STC()
    ST_EC_Map = loaders.get_parameter("ST_EC_Map")
    STCs = list(DS_ES_STC_DemandGranularityMap.query(
        f"DemandSector == '{DS}'")['ServiceTechCategory'].values)

    STs = fs.flatten(STC_to_STs(STC) for STC in STCs)

    implicit = fs.flatten([row[1:]
                           for row in ST_EC_Map for ST in STs if row[0] == ST])
    return implicit


def EC_to_STCs(EC):
    return [ST_to_STC(ST) for ST in EC_to_STs(EC)]


def check_time_granularity_DS_Cons1(DS_Cons1_Map, DS_ES_STC_DemandGranularityMap):
    """
    checks if DS_Cons1_Map has time granularity coarser than bottomup demand's
    time granularity for same DS
    """
    if DS_ES_STC_DemandGranularityMap is None or DS_ES_STC_DemandGranularityMap.size == 0:
        return True

    demand_map = DS_ES_STC_DemandGranularityMap[[
        'DemandSector', 'TimeGranularity']]
    cond = True
    t_values = ('YEAR', 'SEASON', 'DAYTYPE', 'DAYSLICE')

    for ds, tgran in demand_map.values:
        if DS_Cons1_Map is None or ds not in DS_Cons1_Map:
            TGRAN = 'YEAR'
        else:
            TGRAN = DS_Cons1_Map[ds][1]
        cond = cond and coarser(TGRAN, tgran, t_values)
    return cond


def check_geo_granularity_DS_Cons1(DS_Cons1_Map, DS_ES_STC_DemandGranularityMap):
    """
    checks if DS_Cons1_Map has geographic granularity finer than bottomup
    demand's geographic granuarity for same ds
    """
    if DS_ES_STC_DemandGranularityMap is None:
        return True

    demand_map = DS_ES_STC_DemandGranularityMap[[
        'DemandSector', 'GeographicGranularity']]

    cond = True
    g_values = tuple(constant.GEO_COLUMNS.keys())

    for ds, ggran in demand_map.values:
        if DS_Cons1_Map is None or ds not in DS_Cons1_Map:
            GGRAN = utilities.get_valid_geographic_levels()[-1].upper()
        else:
            GGRAN = DS_Cons1_Map[ds][0]
        cond = cond and finer(GGRAN, ggran, g_values)
    return cond


def validate_consumertype2(Cons1_Cons2_Map, CONSUMERTYPES1):
    if Cons1_Cons2_Map:
        return fs.x_in_y(x=[row[0] for row in Cons1_Cons2_Map], y=CONSUMERTYPES1)
    else:
        return True


def get_ds_list(name):
    """List of possible DemandSectors for given demand sector parameter.

    Parameters
    ----------
    name: str
       Name of parameter , anyone from nested paramters od demand.
       e.g. BaseYearDemand, DemandElasticity

    Returns
    -------
       list demand sectors which have that parameter
    """
    DS_ES_Map = get_combined("DS_ES_Map")
    if name in ['BaseYearDemand', 'DemandElasticity']:
        return list(DS_ES_Map.query("InputType == 'GDPELASTICITY'")['DemandSector'].values)
    elif name == "ExogenousDemand":
        return list(DS_ES_Map.query("InputType == 'EXOGENOUS'")['DemandSector'].values)
    elif name in ["ResidualDemand"]:
        return list(DS_ES_Map.query("InputType == 'RESIDUAL'")['DemandSector'].values)
    elif name in ['DS_ES_Map', 'DS_ES_EC_Map', 'DS_ES_STC_DemandGranularityMap',
                  'DS_Cons1_Map', 'Cons1_Cons2_Map']:
        return loaders.get_parameter('DS_List')
    else:
        return list(DS_ES_Map.query("InputType == 'BOTTOMUP'")['DemandSector'].values)


def validate_g(ds, es, ec):
    logger.info(f"Validating GTProfile from {ds},{es} for {ec}")
    try:
        data = loaders.get_parameter("GTProfile",
                                     demand_sector=ds,
                                     energy_service=es,
                                     energy_carrier=ec)
        if not fs.isnone(data):
            try:
                v = validate_each_demand_param("GTProfile",
                                               data,
                                               demand_sector=ds,
                                               energy_service=es,
                                               energy_carrier=ec)

                if not v:
                    logger.error(
                        f"Validation failed for GTProfile from {ds},{es} for {ec}")

                    print(
                        f"Validation failed for GTProfile from {ds},{es} for {ec}")
            except Exception as e:
                logger.exception(e)
                logger.error(
                    f"GTProfile for {ds},{es},{ec} has invalid data")
                print(e)
                v = False
            return v
        else:
            logger.debug(f"GTProfile is not given for {ds},{es},{ec}")
            return True
    except utilities.InvalidCGTDataError as cgt:
        logger.exception(cgt)
        logger.error(
            f"GTProfile for {ds},{es},{ec} has invalid entries for CGT combinations")
        return False


def validate_GTProfile():
    return all(execute_in_process_pool(validate_g, get_all_ds_es_ec()))


def finer_than_demand(GTProfile, demand_sector, energy_service, energy_carrier):
    def decuce_ecs(STC):
        STs = STC_to_STs(STC)
        return fs.flatten([ST_to_ECs(ST) for ST in STs])

    def check_gran_gt():
        t_vals = [t.upper() for t in constant.TIME_SLICES]
        g_vals = [g.upper() for g in constant.GEOGRAPHIES]

        gcols = utilities.get_geographic_columns_from_dataframe(
            gtprofile)
        tcols = utilities.get_time_columns_from_dataframe(
            gtprofile)

        v = finer(gcols[-1].upper(), GGRAN, g_vals)
        valid = True
        if not v:
            logger.error(
                f"Geographic granularity of GTProfile for {demand_sector}, {energy_service}, {energy_carrier}, {stc}, {st} is coarser than demand.")
            valid = False
        v = finer(tcols[-1].upper(), TGRAN, t_vals)
        if not v:
            logger.error(
                f"Time granularity of GTProfile for {demand_sector}, {energy_service}, {energy_carrier}, {service_tech} is coarser than demand.")
            valid = False
        return valid

    if is_bottomup(demand_sector, energy_service):
        granmap = loaders.get_parameter("DS_ES_STC_DemandGranularityMap",
                                        demand_sector=demand_sector)
        granmap = granmap.set_index(['DemandSector',
                                     'EnergyService',
                                    'ServiceTechCategory'])
        ds_es_stc = [(ds, es, stc) for ds, es, stc in granmap.index if ds ==
                     demand_sector and es == energy_service and energy_carrier in decuce_ecs(stc)]

        valid = True
        for ds, es, stc in ds_es_stc:
            TGRAN = granmap['TimeGranularity'].loc[(ds, es, stc)]
            GGRAN = granmap['GeographicGranularity'].loc[(ds, es, stc)]
            if 'ServiceTech' in GTProfile.columns:
                service_techs = STC_to_STs(stc)
                for st in service_techs:
                    if st in list(GTProfile.ServiceTech.values):
                        gtprofile = utilities.filter_empty(
                            GTProfile.query(f"ServiceTech=='{st}'"))
                    else:
                        gtprofile = utilities.filter_empty(
                            GTProfile[GTProfile.ServiceTech.isna()])
                        if len(gtprofile) == 0:
                            continue

                    valid = valid and check_gran_gt()
            else:
                gtprofile = GTProfile
                st = ""
                valid = valid and check_gran_gt()

        return valid
    else:
        granmap = loaders.get_parameter("DS_ES_EC_Map",
                                        demand_sector=demand_sector)
        granmap = granmap.set_index(['DemandSector',
                                     'EnergyService',
                                    'EnergyCarrier'])

        st, stc = "", ""
        index_ = (demand_sector, energy_service, energy_carrier)
        TGRAN = granmap['TimeGranularity'][index_]
        GGRAN = granmap['GeographicGranularity'][index_]
        gtprofile = GTProfile
        return check_gran_gt()


def existence_demand_parameter(name):
    ds = get_ds_list(name)
    ds = list(set(ds))
    args = [(name, d) for d in ds]
    valid = execute_in_process_pool(existence_demand_parameter_, args)
    return all(valid)


def existence_demand_parameter_(name, demand_sector, energy_carrier=None):

    try:
        logger.info(f"Validating {name} from {demand_sector}")
        data = loaders.get_parameter(name,
                                     demand_sector=demand_sector)
        if not isinstance(data, pd.DataFrame) and not data:
            specs = filemanager.demand_specs()[name]
            if 'optional' in specs and specs['optional']:
                logger.warning(f"{name} for {demand_sector} is not given")
                valid = True
            else:
                valid = False
        else:
            valid = validate_each_demand_param(name, data,
                                               demand_sector=demand_sector)

        if not valid:
            print(f"Validation failed for  {name} from {demand_sector}")
            logger.error(
                f"Validation failed for  {name} from {demand_sector}")
    except FileNotFoundError as fne:
        logger.error(f"{name} for {demand_sector} is not given")
        valid = False
        logger.exception(fne)
    except utilities.InvalidCGTDataError as cgt:
        logger.error(
            f"{name} for {demand_sector} has invalid entries in CGT fields")
        valid = False
        logger.exception(cgt)
    except Exception as e:
        logger.error(f"{name} for {demand_sector} has invalid data")
        valid = False
        logger.exception(e)

    return valid


def check_basedemand_elasticity_gran():
    """
    cheks if baseyeardemand and demandelasicity have same granularity and is equal to
    granularity specified in DemandGranularityMap
    """
    ds = get_ds_list('BaseYearDemand')
    for d in ds:
        logger.debug(
            f"Checking if granularity is same for BaseYearDemand and DemandElasticity for {d}")
        BaseYearDemand = get_BaseYearDemand(d)
        DemandElasticity = get_DemandElasticity(d)
        indexcols = ["EnergyService", "EnergyCarrier"]
        BaseYearDemand = BaseYearDemand.set_index(indexcols).sort_index()
        DemandElasticity = DemandElasticity.set_index(indexcols).sort_index()
        for item in BaseYearDemand.index.unique():
            q = "EnergyService=='{}' & EnergyCarrier=='{}'".format(
                item[0], item[1])
            byd = utilities.filter_empty(BaseYearDemand.query(q))
            de = utilities.filter_empty(DemandElasticity.query(q))
            bg = utilities.get_geographic_columns_from_dataframe(byd)
            dg = utilities.get_geographic_columns_from_dataframe(de)
            geogran = get_geographic_granularity(d, *item)
            grancols = constant.GEO_COLUMNS[geogran]
            if bg == dg:
                logger.debug(
                    f"Geographic granularity of BaseYearDemand and DemandElasticity is same for {d},{item}")
            if bg != grancols:
                logger.error(
                    f"Geographic granularity of BaseYearDemand for {d},{item} is different than specified in DS_ES_EC_Map.")
                return False
            if dg != grancols:
                logger.error(
                    f"Geographic granularity of DemandElasticity for {d},{item} is diffecrent than specified in DS_ES_EC_Map.")
                return False

    return True


def get_all_ES_Demand(ds, es):
    """returns dictionary of ES_Demand data for each ST.
    it returns dict with key as ST and ES_Demand as value
    """
    DS_ES_STC_DemandGranularityMap = get_combined(
        "DS_ES_STC_DemandGranularityMap")
    STCs = DS_ES_STC_DemandGranularityMap.query(f"DemandSector == '{ds}' & EnergyService == '{es}'")[
        'ServiceTechCategory']

    return {s: loaders.get_parameter('ES_Demand',
                                     demand_sector=ds,
                                     energy_service=es,
                                     service_tech_category=s) for s in STCs}


def read_header(filepath):
    df = pd.read_csv(filepath, nrows=1)
    return list(df.columns)


def check_ES_Demand_columns():
    ds_es = get_bottomup_ds_es()
    valid = True
    for ds, es in ds_es:
        valid = valid and _check_ES_Demand_columns(ds, es)
    return valid


def get_structural_columns(ds):
    return constant.TIME_COLUMNS[utilities.get_valid_time_levels()[-1]] +\
        constant.GEO_COLUMNS[utilities.get_valid_geographic_levels()[-1]] +\
        get_cons_columns(ds)


def _check_ES_Demand_columns(ds, es):
    """checks if ES_Demand file has correct column names specified
    """
    DS_ES_STC_DemandGranularityMap = loaders.get_parameter(
        "DS_ES_STC_DemandGranularityMap", demand_sector=ds)
    STCs = DS_ES_STC_DemandGranularityMap.query(f"DemandSector == '{ds}' & EnergyService == '{es}'")[
        'ServiceTechCategory']

    filepaths = {s: find_custom_DS_ES_filepath(ds,
                                               es,
                                               'ES_Demand',
                                               f"{s}_") for s in STCs}
    valid = True
    for STC, path in filepaths.items():
        columns = read_header(path)
        structural = get_structural_columns(ds)
        other_cols = [c for c in columns if c not in structural]
        unexpected_cols = [c for c in other_cols if STC not in c]
        if unexpected_cols:
            logger.warning(
                f"Found unexpected columns {unexpected_cols}, in {STC}_ES_Demand file")
        stc_cols = [c for c in other_cols if STC in c]
        combinations = [set(c.split(constant.ST_SEPARATOR_CHAR))
                        for c in stc_cols]
        if any([combinations.count(c) > 1 for c in combinations]):
            logger.error(
                f"It is not allowed for two columns to have the exact same combinations of STCs in {STC}_ES_Demand")
            valid = False
        stcs = get_corresponding_stcs(ds, es, STC)
        expected = fs.flatten([[set(x) for x in itertools.combinations(
            stcs, n)] for n in range(1, len(stcs)+1)])
        unexpected = [comb for comb in combinations if comb not in expected]
        if unexpected:
            logger.error(
                f"Found unexpected combination of STs in {STC}_ES_Demand")
            logger.error("Unexpected combination of STs in column {}".format(
                [constant.ST_SEPARATOR_CHAR.join(c) for c in unexpected]))
            valid = False

    return valid


def get_all_UsagePenetration(ds, es):
    """returns all penetration data as dictionary with key as st, value as
    dictionary of ST combinations and actual penetration data.
    {"ST":{(ST1,ST2): penetration data for ST1 and ST2}
    """
    DS_ES_STC_DemandGranularityMap = get_combined(
        "DS_ES_STC_DemandGranularityMap")
    STCs = DS_ES_STC_DemandGranularityMap.query(f"DemandSector == '{ds}' & EnergyService == '{es}'")[
        'ServiceTechCategory']

    d = {}
    for s in STCs:
        es_demand = loaders.get_parameter('ES_Demand',
                                          demand_sector=ds,
                                          energy_service=es,
                                          service_tech_category=s)
        combs = [tuple(name.split(constant.ST_SEPARATOR_CHAR))
                 for name in es_demand.columns if s in name]

        d[s] = {tuple(c): loaders.get_parameter('UsagePenetration',
                                                demand_sector=ds,
                                                energy_service=es,
                                                STC_combination=c) for c in combs}
    return d


def get_data(name, ds, es):
    if name in ['EfficiencyLevelSplit', 'NumInstances', 'TechSplitRatio']:
        return {(ds, es): loaders.get_parameter(name,
                                                demand_sector=ds,
                                                energy_service=es)}
    elif name == "ES_Demand":
        return get_all_ES_Demand(ds, es)
    elif name == "UsagePenetration":
        return get_all_UsagePenetration(ds, es)
    else:
        logger.error(f"Unknown parameter {name}")


def validate_each_demand_param_(name, item, data, ds, es, stc):
    """encapsulation over validate_each_demand_param to catch exception
    """
    logger.info(f"Validating {name} from {ds},{es} for {stc}")
    if fs.isnone(data):
        return False
    try:
        v = validate_each_demand_param(name,
                                       data,
                                       demand_sector=ds,
                                       energy_service=es,
                                       service_tech_category=stc)

        if not v:
            logger.error(
                f"Validation failed for {name} from {ds},{es} for {stc}")

            print(
                f"Validation failed for {name} from {ds},{es} for {stc}")
    except Exception as e:
        logger.exception(e)
        logger.error(
            f"{name} for {ds},{es},{item} has invalid data")
        print(e)
        v = False
    return v


def existence_demand_energy_service_parameter(name):
    """checks existence and basic data validation of
    EfficiencyLevelSplit,NumInstances,ES_Demand,UsagePenetration
    """
    ds_es = get_bottomup_ds_es()
    args = []
    v = True
    for ds, es in ds_es:
        try:
            data_ = get_data(name, ds, es)

            for st, data in data_.items():
                if not isinstance(data, dict):
                    data = {st: data}
                for item in data:
                    args.append((name,
                                 item,
                                 data[item],
                                 ds,
                                 es,
                                 st))

        except FileNotFoundError as fne:
            logger.error(f"{name} for {ds},{es} is not given")
            logger.exception(fne)
            v = False
        except utilities.InvalidCGTDataError as cgt:
            logger.error(
                f"{name} for {ds},{es} has invalid entries in CGT fields")
            logger.exception(cgt)
            v = False

    valid = execute_in_process_pool(validate_each_demand_param_, args)
    # valid = [validate_each_demand_param_(*item) for item in args]
    return all(valid) and v


def validate_each_demand_param(name, data, **kwargs):
    """Validates individual parameter according to specs given in yml file.
    """
    if fs.isnone(data):
        return False
    specs = filemanager.demand_specs()[name]

    if specs.get("optional", False) and isinstance(data, type(None)):
        return True

    return loaders.validate_param(name,
                                  specs,
                                  data,
                                  "rumi.io.demand",
                                  **kwargs)


def subset(data, indexnames, items):
    if isinstance(items, (str, int)) or items is None:
        items = (items,)

    q = " & ".join([f"{name} == '{item}'" for name,
                    item in zip(indexnames, items)])
    return data.query(q)


def check_EfficiencyLevelSplit_granularity():
    return _check_DS_ES_granularity("EfficiencyLevelSplit")


def check_NumInstances_granularity():
    return _check_DS_ES_granularity("NumInstances")


def check_TechSplitRatio_granularity():
    return _check_DS_ES_granularity("TechSplitRatio")


def get_bottomup_ds_es():
    DS_ES_Map = get_combined("DS_ES_Map")
    ds_es = DS_ES_Map.query("InputType == 'BOTTOMUP'")[
        ['DemandSector', 'EnergyService']].copy().values
    return ds_es


def get_nonbottomup_ds_es():
    DS_ES_Map = get_combined("DS_ES_Map")
    ds_es = DS_ES_Map.query("InputType != 'BOTTOMUP'")[
        ['DemandSector', 'EnergyService']].copy().values
    return ds_es


def is_bottomup(ds, es):
    DS_ES_Map = get_combined("DS_ES_Map")
    row = DS_ES_Map.query(f"DemandSector == '{ds}' & EnergyService == '{es}'")

    return len(row) > 0 and row['InputType'].values[0] == 'BOTTOMUP'


def check_demand_granularity_(param_name,
                              data,
                              ds,
                              CSTAR=True,
                              GSTAR=True,
                              TSTAR=True,
                              check_function=utilities.check_granularity_per_entity):
    data = data.set_index(['EnergyService', 'EnergyCarrier'])
    data.sort_index(inplace=True)
    logger.debug(f"Checking granularity of {param_name} for {ds}")
    for item in data.index.unique():
        d = subset(data, data.index.names, item)
        d = utilities.filter_empty(d)

        entity = (ds,) + item

        ConsumerGranularity = None
        GeographicGranularity, TimeGranularity = None, None
        es, ec = item
        C, G, T = get_granularity(param_name,
                                  demand_sector=ds,
                                  energy_service=es,
                                  energy_carrier=ec)
        if CSTAR:
            ConsumerGranularity = C
        if GSTAR:
            GeographicGranularity = G
        if TSTAR:
            TimeGranularity = T

        v = check_function(d,
                           entity,
                           GeographicGranularity,
                           TimeGranularity,
                           ConsumerGranularity)
        if not v:
            logger.error(
                f"Granularity check failed for {param_name} for {entity}")
        return v


def check_demand_granularity(param_name,
                             CSTAR=False,
                             GSTAR=False,
                             TSTAR=False,
                             check_function=utilities.check_granularity_per_entity):
    """
    Checks whether given data follows granularity as specified in granularity
    map. data file directly inside demand sector folder is tested using this
    function.
    """

    if not any([CSTAR, GSTAR, TSTAR]):
        raise Exception(
            "check_demand_granularity function must have valid CSTAR/GSTAR/TSTAR argument")

    dslist = get_ds_list(param_name)
    if not dslist:
        return True

    valid = True
    for ds in dslist:
        data = loaders.get_parameter(param_name, demand_sector=ds)
        v = check_demand_granularity_(param_name,
                                      data,
                                      ds,
                                      CSTAR=CSTAR,
                                      GSTAR=GSTAR,
                                      TSTAR=TSTAR,
                                      check_function=check_function)
        valid = valid and v
    return valid


def get_corresponding_sts(demand_sector,
                          energy_service,
                          service_tech):
    """Returns STs with same ds, es and EC corresponding to this ds,es,st combination
    """
    STs = get_STs(demand_sector, energy_service)
    ECs = set(ST_to_ECs(service_tech))
    return [st for st in STs if set(ST_to_ECs(st)) & ECs]


def get_corresponding_stcs(demand_sector,
                           energy_service,
                           service_tech_category):
    """Returns STCs with same ds, es and EC corresponding to this ds,es,st combination
    """
    DS_ES_STC_DemandGranularityMap = loaders.get_parameter(
        "DS_ES_STC_DemandGranularityMap",
        demand_sector=demand_sector)
    STCs = DS_ES_STC_DemandGranularityMap.query(f"DemandSector == '{demand_sector}' & EnergyService == '{energy_service}'")[
        'ServiceTechCategory']
    return list(STCs.values)


def coarsest(gran_map):
    """returns coarsest granuarities from given granalarity map dataframe
    """
    c = min(gran_map.to_dict(orient='records'),
            key=lambda x: len(constant.CONSUMER_COLUMNS[x['ConsumerGranularity']]))['ConsumerGranularity']
    g = min(gran_map['GeographicGranularity'].values,
            key=lambda x: len(constant.GEO_COLUMNS[x]))
    t = min(gran_map['TimeGranularity'].values,
            key=lambda x: len(constant.TIME_COLUMNS[x]))
    return c, g, t


def check_ST_SEC_granularity(ST_SEC, demand_sector):
    st_sec = ST_SEC.set_index(['ServiceTech', 'EnergyService'])

    valid = True
    for ST, ES in st_sec.index.unique():
        subset = ST_SEC.query(
            f"ServiceTech == '{ST}' & EnergyService == '{ES}'")
        subset = utilities.filter_empty(subset)
        C, G, T = get_granularity("ST_SEC",
                                  demand_sector=demand_sector,
                                  energy_service=ES,
                                  service_tech_category=ST_to_STC(ST))
        v = coarser_granularity_per_entity("ST_SEC",
                                           subset,
                                           ST,
                                           G,
                                           'YEAR',
                                           C)
        valid = valid and v
        if not v:
            logger.error(
                f"Granularity check failed for ST_SEC in {demand_sector}, for {ST}, {ES}")

    return valid


def _check_DS_ES_granularity(param_name):
    """
    Checks whether EfficiencyLevelSplit/NumInstances follows granularity as specified in granularity map.
    """

    granularity = get_gran_map(param_name)
    ds_es = get_bottomup_ds_es()
    valid = True
    for ds, es in ds_es:
        data_ = loaders.get_parameter(param_name,
                                      demand_sector=ds,
                                      energy_service=es)
        data = data_.set_index('ServiceTech')
        logger.debug(f"Checking granularity of {param_name} for {ds},{es}")
        for ST in data.index.unique():
            d = data.loc[ST]
            d = utilities.filter_empty(d)
            stc = ST_to_STC(ST)
            g = granularity.loc[(ds, es, stc)]
            ConsumerGranularity = g['ConsumerGranularity']
            GeographicGranularity = g['GeographicGranularity']
            TimeGranularity = g['TimeGranularity']

            v = coarser_granularity_per_entity(param_name,
                                               d,
                                               ST,
                                               GeographicGranularity,
                                               TimeGranularity,
                                               ConsumerGranularity)
            valid = valid and v
            if not v:
                logger.error(
                    f"Granularity check failed for {param_name} for {ST}")

    return valid


def coarser_granularity_per_entity(param_name,
                                   d,
                                   entity,
                                   GeographicGranularity,
                                   TimeGranularity,
                                   ConsumerGranularity=None):
    """check if the entity specified is coarses than specified
    """
    geo_columns, time_columns, cons_columns = [], [], []

    if GeographicGranularity:
        geo_columns = utilities.get_geographic_columns(GeographicGranularity)
    if TimeGranularity:
        time_columns = utilities.get_time_columns(TimeGranularity)
    if ConsumerGranularity:
        cons_columns = constant.CONSUMER_COLUMNS[ConsumerGranularity]

    dataset_columns = utilities.get_all_structure_columns(d)

    diff = set(dataset_columns) - \
        set(geo_columns + time_columns + cons_columns)

    valid = True

    if diff:
        c, r = d[list(diff)].shape
        empty = d[list(diff)].isnull().sum().sum() == c*r

        if not empty:
            line1 = f"In {param_name} granularity is finer than expected for {entity}!"
            line2 = f"It has these columns extra, {diff}"
            logger.error("\n".join([line1, line2]))

        valid = valid and empty

    return valid


def _check_ES_Demand_granularity(param_name):
    """
    Checks whether ES_Demand follows granularity as
    specified in granularity map.
    """

    ds_es = get_bottomup_ds_es()
    valid = True
    for ds, es in ds_es:
        data_ = get_data(param_name, ds, es)
        for STC, data in data_.items():
            d_ = data
            if not isinstance(data, dict):
                d_ = {STC: data}

            for item, df in d_.items():
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                d = utilities.filter_empty(df)
                C, G, T = get_granularity(param_name,
                                          demand_sector=ds,
                                          energy_service=es,
                                          service_tech_category=STC)

                ConsumerGranularity, GeographicGranularity, TimeGranularity = C, G, T
                v = utilities.check_granularity_per_entity(d,
                                                           item,
                                                           GeographicGranularity,
                                                           TimeGranularity,
                                                           ConsumerGranularity)
                valid = valid and v
                if not v:
                    logger.error(
                        f"Granularity check failed for {param_name} for {STC}, {item}")

    return valid


def _check_UsagePenetration_granularity(param_name):
    """
    Checks whether UsagePenetration follows granularity as
    specified in granularity map.
    """

    granularity = get_gran_map(param_name)
    ds_es = get_bottomup_ds_es()
    valid = True
    for ds, es in ds_es:
        data_ = get_data(param_name, ds, es)
        for STC, data in data_.items():
            d_ = data
            if not isinstance(data, dict):
                d_ = {STC: data}

            for comb, df in d_.items():
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                d = utilities.filter_empty(df)
                g = granularity.loc[(ds, es, STC)]
                ConsumerGranularity = g['ConsumerGranularity']
                GeographicGranularity = g['GeographicGranularity']
                TimeGranularity = g['TimeGranularity']
                v = coarser_granularity_per_entity(param_name,
                                                   d,
                                                   STC,
                                                   GeographicGranularity,
                                                   TimeGranularity,
                                                   ConsumerGranularity)

                valid = valid and v
                if not v:
                    logger.error(
                        f"Granularity check failed for {param_name} for {STC}, {comb}")

    return valid


def check_ES_Demand_granularity():
    return _check_ES_Demand_granularity("ES_Demand")


def check_UsagePenetration_granularity():
    return _check_UsagePenetration_granularity("UsagePenetration")


def check_numconsumers_granularity():
    """
    Checks whether NumConsumers data follows granularity as specified in
    granularity map.
    """

    granularity = get_combined("DS_Cons1_Map")
    param_name = 'NumConsumers'
    dslist = get_ds_list(param_name)
    valid = True
    for ds in dslist:
        data = loaders.get_parameter(param_name, demand_sector=ds)
        if fs.isnone(data):
            continue
        g = granularity[ds]
        conscols = get_cons_columns(ds)
        if conscols:
            ConsumerGranularity = conscols[-1].upper()
        else:
            ConsumerGranularity = "CONSUMERALL"
        GeographicGranularity = g[0]
        TimeGranularity = g[1]

        v = utilities.check_granularity_per_entity(data,
                                                   (ds, "NumConsumers"),
                                                   GeographicGranularity,
                                                   TimeGranularity,
                                                   ConsumerGranularity)
        valid = valid and v
        if not v:
            logger.error(
                f"Granularity check failed for {param_name} for {ds}")

    return valid


def is_nppc(EC):
    nppc = loaders.get_parameter("NonPhysicalPrimaryCarriers")

    if fs.isnone(nppc):
        return False
    return EC in nppc.EnergyCarrier.values


def get_STs(demand_sector, energy_service):
    """returns all ServiceTechs in given demand_sector and energy_service
    """
    return get_STs_(f"DemandSector == '{demand_sector}' & EnergyService == '{energy_service}'")


def get_STs_from_ds(demand_sector):
    """returns all ServiceTechs in given demand_sector
    """
    return get_STs_(f"DemandSector == '{demand_sector}'")


def get_STs_(query):
    DS_ES_STC_DemandGranularityMap = get_combined(
        "DS_ES_STC_DemandGranularityMap")
    STCs = DS_ES_STC_DemandGranularityMap.query(query)[
        'ServiceTechCategory']
    STs = fs.flatten([STC_to_STs(STC) for STC in STCs])
    return list([ST for ST in set(STs) if not is_nppc(ST_to_EC(ST))])


def get_STCs(ds, es):
    """generate all ServiceTechCategories for given demand_sector and energy_service
    """
    DS_ES_STC_DemandGranularityMap = loaders.get_parameter(
        "DS_ES_STC_DemandGranularityMap", demand_sector=ds)
    ds_es_stc = DS_ES_STC_DemandGranularityMap.query(
        f"DemandSector == '{ds}' & EnergyService == '{es}'")
    return [STC for STC in ds_es_stc['ServiceTechCategory']]


def count(itr):
    c = 0
    while True:
        try:
            next(itr)
            c += 1
        except StopIteration as s:
            return c


def list_dir_usagepenetration(folderpath):
    files = []
    if os.path.exists(folderpath):
        files = [f for f in os.listdir(
            folderpath) if f.endswith("UsagePenetration.csv")]
    d = {}
    for f in files:
        d.setdefault(tuple(set(extract_STCs(f))), []).append(
            os.path.join(folderpath, f))

    return d


def check_total_UsagePenetration():
    """checks if UsagePenetration for all combinations of STCs for given
    demand_sector and energy_service totals less than or equal to 1
    """
    precision = 1e-6
    ds_es = get_bottomup_ds_es()
    valid = True
    scenario = config.get_config_value("scenario")

    for ds, es in ds_es:
        STCs = get_STCs(ds, es)
        globalpath = os.path.dirname(find_custom_DS_ES_filepath(ds,
                                                                es,
                                                                "UsagePenetration",
                                                                ""))
        scenariopath = globalpath.replace("Default Data",
                                          os.path.join("Scenarios",
                                                       scenario))
        files = list_dir_usagepenetration(globalpath)
        sfiles = list_dir_usagepenetration(scenariopath)
        files.update(sfiles)
        p = []
        for comb, filepaths in files.items():
            if len(filepaths) > 1:
                logger.error(
                    f"Duplicate UsagePenetration files found for {comb}")
                valid = False
            else:
                df = pd.read_csv(filepaths[0])
                p.append(df.set_index(utilities.get_all_structure_columns(df)))

        if not p:
            continue

        v = (functools.reduce(
            lambda x, y: x+y, [item['UsagePenetration'] for item in p], 0) > 1 + precision).sum() == 0
        if not v:
            print(f"UsagePenetration for {ds}, {es} sums more than 1!")
            logger.error(f"UsagePenetration for {ds},{es} sums more than 1!")

        valid = valid and v

    return valid


def assure_ST_ES(param_name, data: pd.DataFrame, demand_sector):
    """ check for <ST, ES> should be valid combination
    <ES,ST> -> from STC_ES_Map and STC_ST_Map
    """
    ST_ES_EC_df = data.loc[:, ['ServiceTech',
                               'EnergyService', 'EnergyCarrier']]
    ST_ES_EC_df['ServiceTechCategory'] = ST_ES_EC_df.ServiceTech.apply(
        ST_to_STC)

    es_stc = set((es, st, stc) for st, stc in ST_ES_EC_df[['ServiceTech', 'ServiceTechCategory']].drop_duplicates().values
                 for es in STC_to_ESs(stc, demand_sector))

    es_stc_data = set(ST_ES_EC_df[['EnergyService', 'ServiceTech', 'ServiceTechCategory']].itertuples(
        index=False, name=None))

    if es_stc_data == es_stc:
        return True

    extra = es_stc_data - es_stc
    less = es_stc - es_stc_data
    if less:
        missing_rows = [(es, st) for es, st, stc in less]
        logger.error(
            f"In {param_name} parameter from {demand_sector}, entries for following EnergyService,ServiceTech are missing , {missing_rows}")
    if extra:
        extra_rows = [(es, st) for es, st, stc in extra]
        logger.error(
            f"In {param_name} parameter from {demand_sector}, entries for following EnergyService,ServiceTech are invalid, {extra_rows}")
    return False


def assure_ST_EC(param_name, data: pd.DataFrame, demand_sector):
    """ check for <ST, EC> should be valid combination
    <EC,ST> -> from ST_EC_Map
    """
    ST_ES_EC_df = data.loc[:, ['ServiceTech',
                               'EnergyService', 'EnergyCarrier']]

    st_ec = set((st, ec) for st in ST_ES_EC_df['ServiceTech'].unique()
                for ec in ST_to_ECs(st))

    st_ec_data = set(ST_ES_EC_df[['ServiceTech', 'EnergyCarrier']].itertuples(
        index=False, name=None))

    if st_ec_data == st_ec:
        return True

    extra = st_ec_data - st_ec
    less = st_ec - st_ec_data
    if less:
        logger.error(
            f"In {param_name} parameter from {demand_sector}, entries for following ServiceTech,EnergyCarrier are missing , {less}")
    if extra:
        logger.error(
            f"In {param_name} parameter from {demand_sector}, entries for following ServiceTech,EnergyCarrier are invalid, {extra}")
    return False


def assure_one_across_entity(data,
                             columnname,
                             param_name,
                             precision=1e-5,
                             entity="ServiceTechCategory",
                             demand_sector=None,
                             energy_service=None):
    """Assures that values in given columnname sums to one across all CGT in
    given entity category
    """
    valid = True
    for category in (data[entity]).unique():
        subset = data.query(f"{entity}=='{category}'")
        subset = utilities.filter_empty(subset)
        cols = utilities.get_all_structure_columns(subset)
        groupcols = cols + [entity]
        groupedcol = subset.groupby(groupcols).sum(
            numeric_only=True)[columnname]
        diff = pd.Series([1]*len(groupedcol),
                         index=groupedcol.index) - groupedcol
        check = diff.apply(abs) <= precision
        v = np.all(check)
        if not v:
            d = groupedcol[groupedcol == 0]
            if len(d) == len(check[check == False]):
                v = True
                logger.debug(
                    f"In {param_name} from {demand_sector}/{energy_service} following rows have {columnname} as zero")
                for row in d.reset_index().values:
                    logger.debug(row)
            else:
                logger.error(
                    f"{param_name} from {demand_sector}/{energy_service} Data does not sum to 1 for {columnname} across all CGT in {entity}, {category}")
        valid = v and valid

    return valid


def assure_one_across_servicetechcategory(data,
                                          columnname,
                                          param_name,
                                          precision=1e-5,
                                          demand_sector=None,
                                          energy_service=None
                                          ):
    """Assures that values in given columnname sums to one across all CGT in
    all ServiceTechCategories
    """
    return assure_one_across_entity(data,
                                    columnname,
                                    param_name,
                                    precision,
                                    entity="ServiceTechCategory",
                                    demand_sector=demand_sector,
                                    energy_service=energy_service)


def assure_one_across_servicetech(data,
                                  columnname,
                                  param_name,
                                  precision=1e-5,
                                  demand_sector=None,
                                  energy_service=None):
    """Assures that values in given columnname sums to one across all CGT in
    all ServiceTechs
    """
    return assure_one_across_entity(data,
                                    columnname,
                                    param_name,
                                    precision,
                                    entity="ServiceTech",
                                    demand_sector=demand_sector,
                                    energy_service=energy_service)


def assure_coarsest_gran_DS_ES_Param(param_name):
    ds_es = get_bottomup_ds_es()
    dss = set([ds for ds, es in ds_es])
    valid = True
    for ds in dss:
        valid = assure_coarsest_gran_DS_ES_Param_(param_name, ds)

    return valid


def assure_coarsest_gran_DS_ES_Param_(param_name,
                                      demand_sector):
    """ Note that default file of any overridden parameter,
    the input should be given at the coarsest consumer type, geographic and temporal
    granularities among these respective granularities specified across all energy
    services in DS_ES_STC_DemandGranularityMap for the specified STC in the specified
    demand sector.
    """
    def error(granname, GRAN):
        logger.error(
            f"For {param_name} in {demand_sector} default input for {stc}, {granname} is finer than expected {GRAN}")

    def get_gran_columns(C, G, T):
        C_ = utilities.get_consumer_columns(C)
        G_ = utilities.get_geographic_columns(G)
        T_ = utilities.get_time_columns(T)
        return C_, G_, T_

    def get_cols_from_dataframe(df):
        c = utilities.get_consumer_columns_from_dataframe(df)
        g = utilities.get_geographic_columns_from_dataframe(df)
        t = utilities.get_time_columns_from_dataframe(df)
        return c, g, t

    specs = filemanager.get_specs(param_name)
    entities = specs.get('entities')
    filepath = get_default_file(param_name,
                                demand_sector,
                                "", "")

    if not os.path.exists(filepath):
        return True

    data = pd.read_csv(filepath)
    gran_map = loaders.get_parameter("DS_ES_STC_DemandGranularityMap",
                                     demand_sector=demand_sector)

    valid = True
    logger.debug(
        f"Checking if {param_name} default data is given at appropriate granularity")
    for items in data.set_index(entities).index.unique():
        df = utilities.filter_empty(subset(data, entities, items))
        ec, st, stc = get_entity_values(entities, df)
        q = f"DemandSector=='{demand_sector}' & ServiceTechCategory=='{stc}'"
        granm = gran_map.query(q)
        if len(granm) == 0:
            logger.warning(
                f"Invalid ServiceTechCategory in {param_name} from {demand_sector}, {stc}")
            continue

        C, G, T = coarsest(granm)
        C_, G_, T_ = get_gran_columns(C, G, T)
        c, g, t = get_cols_from_dataframe(df)

        v = len(C_) >= len(c)
        if not v:
            error("ConsumerGranularity", C)
            valid = False

        v = len(G_) >= len(g)
        if not v:
            error("GeographicGranularity", G)
            valid = False

        v = len(T_) >= len(t)
        if not v:
            error("TimeGranularity", T)
            valid = False

    return valid


def add_demand_sector_filters(demand_sector):
    """This adds filters to all the demand paramters which has demand_sector in it.
    it changes the yaml datastructure. This will work with assumption that
    yaml files are read once and cached forever.

    This function has to be called before reading any file to make sure filters
    are applied
    """
    dataframe_filter = f"DemandSector == '{demand_sector}'"
    yaml = filemanager.demand_specs()
    params = [k for k in yaml if 'DemandSector' in yaml[k].get('columns', [])]
    other = ['DS_List']
    for param_name in params + other:
        yaml[param_name].setdefault('filterqueries', [])

    for param_name in params:
        yaml[param_name]['filterqueries'].append(dataframe_filter)

    yaml['DS_List']['filterqueries'].append(f" item == '{demand_sector}'")


def assure_same_efficiencylevelnames(ST_SEC, demand_sector):
    valid = True
    for ST in ST_SEC.ServiceTech.unique():
        subset = ST_SEC.query(f"ServiceTech == '{ST}'")
        ESs = subset.EnergyService.unique()
        levels = {}
        l = []
        for e in ESs:
            l.append(subset[subset.EnergyService == e]
                     ['EfficiencyLevelName'].unique())

        v = []
        for i, item in enumerate(l[1:], start=1):
            v.append(set(item) == set(l[i-1]))

        v = all(v)
        if not v:
            logger.error(
                f"For ST_SEC in {demand_sector}, EfficiencyLevelNames for {ST} are not same across these EnergyServices {ESs}")

        valid = v and valid

    return valid


def STC_to_STs(STC):
    STC_ST_Map = loaders.get_parameter("STC_ST_Map")
    if not STC_ST_Map:
        return [STC]
    STC_ST = [row[1:] for row in STC_ST_Map if row[0] == STC]
    if STC_ST:
        return STC_ST[0]
    else:
        return [STC]


def STC_to_ESs(STC, demand_sector):
    """Find all EnergyServices for given STC in given demand_sector
    """
    STC_ES_Map = loaders.get_parameter("STC_ES_Map")
    DS_ES_STC_DemandGranularityMap = loaders.get_parameter(
        "DS_ES_STC_DemandGranularityMap",
        demand_sector=demand_sector)
    q = f"DemandSector == '{demand_sector}' & ServiceTechCategory == '{STC}'"
    subset = DS_ES_STC_DemandGranularityMap.query(q)
    return list(subset.EnergyService.values)


def STC_to_ESs_(STC):
    """Find all EnergyServices for given STC in given demand_sector
    """
    STC_ES_Map = loaders.get_parameter("STC_ES_Map")
    return [x for row in STC_ES_Map for x in row[1:2:] if row[0] == STC]


def check_ST_EC_consistency(data):
    """check if given dataframe has ST-EC pairs consitent with
    other model specifications"""

    x = data.ServiceTech.apply(ST_to_ECs)
    e = data.EnergyCarrier
    return all([e.iloc[i] in x.iloc[i] for i in range(len(x))])


if __name__ == "__main__":
    pass
