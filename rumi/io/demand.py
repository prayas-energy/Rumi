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
import csv
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
from rumi.io.common import balancing_area, balancing_time
from rumi.io.utilities import check_consumer_validity
from rumi.io.utilities import check_geographic_validity
from rumi.io.utilities import check_time_validity
from rumi.io.multiprocessutils import execute_in_process_pool
logger = logging.getLogger(__name__)


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
    DS_Cons1_Map = loaders.get_parameter("DS_Cons1_Map")
    type1 = DS_Cons1_Map[ds][-1]
    Cons1_Cons2_Map = loaders.get_parameter("Cons1_Cons2_Map")
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


def get_consumer_granularity(ds, specified_gran):
    """Converts CONSUMERALL to actual granularity

    Parameters
    -----------
    demand_specs: str
       Demand sector

    Returns
    -------
    one of CONSUMERTYPE1,CONSUMERTYPE2
    """
    if specified_gran != "CONSUMERALL":
        return specified_gran
    if get_consumer_levels(ds) == 1:
        return "CONSUMERTYPE1"
    else:
        return "CONSUMERTYPE1"


def get_geographic_granularity(demand_sector,
                               energy_service,
                               energy_carrier):
    DS_ES_EC_DemandGranularity_Map = loaders.get_parameter(
        "DS_ES_EC_DemandGranularity_Map")
    granularity_map = DS_ES_EC_DemandGranularity_Map.set_index(['DemandSector',
                                                                'EnergyService',
                                                                'EnergyCarrier'])
    return granularity_map.loc[(demand_sector,
                                energy_service,
                                energy_carrier)]['GeographicGranularity']


def get_type(demand_sector, energy_service):
    """find type of service BOTTOMUP,EXTRANEOUS,GDPELASTICITY or RESIDUAL
    """
    DS_ES_Map = loaders.get_parameter('DS_ES_Map')
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


def get_ExtraneousDemand(demand_sector):
    """loader function for parameter ExtraneousDemand
    """

    extraneous = get_demand_sector_parameter('ExtraneousDemand',
                                             demand_sector)
    return extraneous


def get_ST_Efficiency(demand_sector):
    """ST_Efficiency loader function
    """
    return get_demand_sector_parameter("ST_Efficiency",
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


def get_EfficiencyLevelSplit(demand_sector, energy_service):
    """loader function for parameter EfficiencyLevelSplit
    """
    return get_DS_ES_parameter('EfficiencyLevelSplit',
                               demand_sector,
                               energy_service)


def get_ES_Demand(demand_sector,
                  energy_service,
                  service_tech):
    """loader function for parameter ES_Demand
    should not be used directly. use loaders.get_parameter instead.
    """
    prefix = f"{service_tech}_"
    filepath = find_custom_DS_ES_filepath(demand_sector,
                                          energy_service,
                                          'ES_Demand',
                                          prefix)
    logger.debug(f"Reading {prefix}ES_Demand from file {filepath}")
    return pd.read_csv(filepath)


def get_Penetration(demand_sector,
                    energy_service,
                    ST_combination):
    """loader function for parameter Penetration
    """
    for item in itertools.permutations(ST_combination):
        prefix = constant.ST_SEPARATOR_CHAR.join(
            item) + constant.ST_SEPARATOR_CHAR
        filepath = find_custom_DS_ES_filepath(demand_sector,
                                              energy_service,
                                              'Penetration',
                                              prefix)
        logger.debug(f"Searching for file {filepath}")
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


def get_DS_ES_parameter(param_name,
                        demand_sector,
                        energy_service):
    """loads parameter which is inside demand_sector/energy_service folder
    """
    filepath = find_custom_DS_ES_filepath(demand_sector,
                                          energy_service,
                                          param_name,
                                          "")
    logger.debug(f"Reading {param_name} from file {filepath}")
    cols = list(filemanager.demand_specs()[param_name]['columns'].keys())
    d = pd.read_csv(filepath)
    return d[[c for c in cols if c in d.columns]]


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


def get_mapped_items(DS_ES_EC_Map):
    """returns list of ECS from DS_ES_EC_Map
    """
    return fs.flatten(fs.drop_columns(DS_ES_EC_Map, 2))


def get_RESIDUAL_ECs(DS_ES_Map, DS_ES_EC_Map):
    df = DS_ES_Map.query("InputType == 'RESIDUAL'")[
        ['DemandSector', 'EnergyService']]
    DS_ES = zip(df['DemandSector'], df['EnergyService'])
    ECs = {(DS, ES): row[2:]
           for DS, ES in DS_ES
           for row in DS_ES_EC_Map if DS == row[0] and ES == row[1]}
    return ECs


def derive_ES_EC(demand_sector, input_type):
    """return set of ES,EC combinations for given demand_sector and input_type but not_BOTTOMUP
    """
    DS_ES_Map = loaders.get_parameter('DS_ES_Map')
    DS_ES_EC_Map = loaders.get_parameter('DS_ES_EC_Map')
    es_ec = fs.concat(*[[(row[1], ec) for ec in row[2:]]
                        for row in DS_ES_EC_Map if row[0] == demand_sector])
    return [(es, ec) for es, ec in es_ec if len(DS_ES_Map.query(f"DemandSector=='{demand_sector}' & EnergyService=='{es}' & InputType=='{input_type}'")) > 0]


def check_RESIDUAL_EC(DS_ES_Map, DS_ES_EC_Map):
    """Each EC specified for a <DS, ES> combination,
       whose InputType in DS_ES_Map is RESIDUAL,
       must occur at least once in another
       <DS, ES> combination for the same DS
    """
    def x_in_y(x, y):
        return any([ix in y for ix in x])

    ECS = get_RESIDUAL_ECs(DS_ES_Map, DS_ES_EC_Map)
    items1 = [row
              for row in DS_ES_EC_Map
              for DS, ES in ECS if row[0] == DS and row[1] != ES and x_in_y(ECS[(DS, ES)], row[2:])]
    if len(items1) == 0 and ECS:
        DS_ES_ST = expand_DS_ES_ST()
        ST_Info = loaders.get_parameter('ST_Info')
        items2 = []
        for ECs in ECS.values():
            for EC in ECs:
                STS = ST_Info.query(f"EnergyCarrier == '{EC}'")[
                    'ServiceTech']
                items2.extend([row for row in DS_ES_ST for DS, ES in ECS if row[0]
                               == DS and row[1] != ES and x_in_y(STS, row[2:])])

    return not ECS or len(items1) > 0 or len(items2) > 0


def are_BOTTOMUP(DS_ES_X_Map, DS_ES_Map):
    DS_ES = fs.transpose(fs.take_columns(DS_ES_X_Map, 2))
    df = fs.combined_key_subset(DS_ES, DS_ES_Map).query(
        "InputType != 'BOTTOMUP'")
    return len(df) == 0


def not_BOTTOMUP(DS_ES_X_Map, DS_ES_Map):
    DS_ES = fs.transpose(fs.take_columns(DS_ES_X_Map, 2))
    df = fs.combined_key_subset(DS_ES, DS_ES_Map).query(
        "InputType == 'BOTTOMUP'")
    return len(df) == 0


def check_ALL_DS(DS_ES_X_Map):
    """
    ES used with ALL as DS can not be used with any other DS.
    This function checks if this is true.
    """
    ES_with_ALL = [row[1] for row in DS_ES_X_Map if row[0] == "ALL"]
    ES_without_ALL = [ES for ES in ES_with_ALL
                      for row in DS_ES_X_Map if row[0] != "ALL"]
    return len(ES_without_ALL) == 0


def listcols(df):
    return [df[c] for c in df.columns]


def check_ALL_ES(DS_ES_EC_DemandGranularity_Map):
    """function for validation
    """
    DS_EC_ALL = DS_ES_EC_DemandGranularity_Map.query(
        "EnergyService == 'ALL'")[['DemandSector', 'EnergyCarrier']]
    DS_EC_NOALL = DS_ES_EC_DemandGranularity_Map.query(
        "EnergyService != 'ALL'")[['DemandSector', 'EnergyCarrier']]
    ALL = set(zip(*listcols(DS_EC_ALL)))
    NOALL = set(zip(*listcols(DS_EC_NOALL)))
    return not ALL & NOALL


def expand_DS_ALL(BOTTOMUP):
    """
    Expands Map when DS is ALL
    """
    if BOTTOMUP:
        cond = "=="
        data = loaders.load_param("DS_ES_ST_Map")
    else:
        data = loaders.load_param("DS_ES_EC_Map")
        cond = "!="

    DS_ES_Map = loaders.load_param("DS_ES_Map")
    ESs = [row for row in data if row[0] == 'ALL']

    for row in ESs:
        ES = row[1]
        data.remove(row)
        nonbottomup = DS_ES_Map.query(
            f"EnergyService == '{ES}' & InputType {cond} 'BOTTOMUP'")
        if len(nonbottomup) > 0:
            ds = nonbottomup['DemandSector']
            for eachds in ds:
                newrow = row.copy()
                newrow[0] = eachds
                data.append(newrow)
    return data


def expand_DS_ES_EC():
    return expand_DS_ALL(BOTTOMUP=False)


def expand_DS_ES_ST():
    return expand_DS_ALL(BOTTOMUP=True)


def is_valid(DS, EC):
    DS_ES_EC_Map = loaders.load_param("DS_ES_EC_Map")
    DS_ES_ST_Map = loaders.load_param("DS_ES_ST_Map")
    ST_Info = loaders.get_parameter("ST_Info")
    ECS = [row for row in DS_ES_EC_Map if row[0] == DS and row[1] == EC]
    STS = ST_Info.query(f"EnergyCarrier == '{EC}'")['ServiceTech']
    DSS = [row[0] for row in DS_ES_ST_Map for ST in STS if row[2] == ST]

    return ECS or DS in DSS


@functools.lru_cache()
def expand_DS_ES_EC_DemandGranularity_Map():
    DS_ES_EC_DemandGranularity_Map = loaders.load_param(
        "DS_ES_EC_DemandGranularity_Map")
    DS_ES_Map = loaders.get_parameter("DS_ES_Map")

    data = DS_ES_EC_DemandGranularity_Map.to_dict(orient="records")
    DSs = [d for d in data if d['EnergyService'] == 'ALL']
    for DS in DSs:
        data.remove(DS)
        DemandSector = DS['DemandSector']
        ALL_DS_ES = DS_ES_Map.query(f"DemandSector == '{DemandSector}'")[
            ['DemandSector', 'EnergyService']].to_dict(orient="records")

        for item in ALL_DS_ES:
            d = DS.copy()
            d.update(item)
            if is_valid(d['DemandSector'], d['EnergyCarrier']):
                data.append(d)
    return pd.DataFrame(data)


def ST_to_EC(ST):
    ST_Info = loaders.get_parameter("ST_Info")
    return ST_Info.query(f"ServiceTech == '{ST}'")['EnergyCarrier'].iloc[0]


def get_service_techs(demand_sector,
                      energy_service,
                      energy_carrier):
    """ServiceTechs for given <demand_sector,energy_service, energy_carrier>
    combination
    """
    DS_ES_ST_Map = loaders.get_parameter("DS_ES_ST_Map")
    ST1 = fs.flatten([row[2:] for row in DS_ES_ST_Map if row[0]
                      == demand_sector and row[1] == energy_service])
    ST2 = EC_to_ST(energy_carrier)

    return tuple(set(ST1) & set(ST2))


def EC_to_ST(energy_carrier):
    ST_Info = loaders.get_parameter("ST_Info")
    return ST_Info.query(f"EnergyCarrier == '{energy_carrier}'")[
        'ServiceTech'].values


def derive_DS_ES_EC():
    DS_ES_EC_Map = expand_DS_ES_EC()
    DS_ES_ST_Map = expand_DS_ES_ST()

    explicit = []
    for row in DS_ES_EC_Map:
        explicit.extend([(row[0], row[1], EC) for EC in row[2:]])

    implicit = []
    for row in DS_ES_ST_Map:
        implicit.extend([(row[0], row[1], ST_to_EC(ST)) for ST in row[2:]])

    return explicit + implicit


def check_DS_ES_EC_validity():
    """
    checks if DS_ES_EC_DemandGranularity_Map has valid combinations of
    <DS,ES,EC>
    """
    gmap = expand_DS_ES_EC_DemandGranularity_Map()
    a = list(
        zip(*listcols(gmap[['DemandSector', 'EnergyService', 'EnergyCarrier']])))
    b = derive_DS_ES_EC()
    return fs.one_to_one(a, b)


def coarser(x, y, values):
    return values.index(x) <= values.index(y)


def finer(x, y, values):
    return values.index(x) >= values.index(y)


def check_granularity(GRANULARITY):

    DS_ES_EC_DemandGranularity_Map = expand_DS_ES_EC_DemandGranularity_Map()
    DS_ES_Map = loaders.get_parameter("DS_ES_Map")

    def get_Granularity(DS, ES, EC):
        df = DS_ES_EC_DemandGranularity_Map.query(
            f"(DemandSector == '{DS}') & (EnergyService =='{ES}') & (EnergyCarrier == '{EC}')")
        return df[GRANULARITY].iloc[0] if len(df) != 0 else None

    def get_input_type(DS, ES):
        return DS_ES_Map.query(f"DemandSector == '{DS}' & EnergyService == '{ES}'")['InputType'].iloc[0]

    DS_ES_EC = list(
        zip(*listcols(DS_ES_EC_DemandGranularity_Map[['DemandSector', 'EnergyService', 'EnergyCarrier']])))
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


def check_ST_ES(DS_ES_ST_Map):
    STS = get_mapped_items(DS_ES_ST_Map)
    repeating_STS = [ST for ST in STS if STS.count(ST) > 1]
    cond = True
    for ST in repeating_STS:
        cond = cond and len(set([row[1]
                                 for row in DS_ES_ST_Map if ST in row[2:]])) == 1
    return cond


class DemandValidationError(Exception):
    pass


@functools.lru_cache(maxsize=None)
def derive_ECs(DS):
    DS_ES_EC_Map = expand_DS_ES_EC()
    DS_ES_ST_Map = expand_DS_ES_ST()
    ST_Info = loaders.get_parameter("ST_Info")
    explicit = fs.flatten([row[2:] for row in DS_ES_EC_Map if row[0] == DS])
    STs = fs.flatten([row[2:] for row in DS_ES_ST_Map if row[0] == DS])

    implicit = [ST_Info.query(f"ServiceTech == '{ST}'")['EnergyCarrier'].iloc[0]
                for ST in STs if len(ST_Info.query(f"ServiceTech == '{ST}'")) != 0]
    return explicit + implicit


def check_time_granularity_DS_Cons1():
    """
    checks if DS_Cons1_Map has time granularity coarser than balancing time
    """

    DS_Cons1_Map = loaders.get_parameter("DS_Cons1_Map")
    cond = True
    t_values = ('YEAR', 'SEASON', 'DAYTYPE', 'DAYSLICE')

    for row in DS_Cons1_Map:
        DS, GGRAN, TGRAN = row[:3]
        ECs = derive_ECs(DS)
        cond = cond and all(
            [coarser(TGRAN, balancing_time(EC), t_values) for EC in ECs])
    return cond


def check_geo_granularity_DS_Cons1():
    """
    checks if DS_Cons1_Map has geographic granularity finer than balancing area
    """
    DS_Cons1_Map = loaders.get_parameter("DS_Cons1_Map")
    cond = True
    g_values = tuple(constant.GEO_COLUMNS.keys())

    for row in DS_Cons1_Map:
        DS, GGRAN, TGRAN = row[:3]
        ECs = derive_ECs(DS)
        cond = cond and all(
            [finer(GGRAN, balancing_area(EC), g_values) for EC in ECs])
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

    if name in ['BaseYearDemand', 'DemandElasticity']:
        DS_ES_Map = loaders.get_parameter("DS_ES_Map")
        return DS_ES_Map.query("InputType == 'GDPELASTICITY'")['DemandSector'].values
    elif name == "ExtraneousDemand":
        DS_ES_Map = loaders.get_parameter("DS_ES_Map")
        return DS_ES_Map.query("InputType == 'EXTRANEOUS'")['DemandSector'].values
    elif name in ["ResidualDemand"]:
        DS_ES_Map = loaders.get_parameter("DS_ES_Map")
        return DS_ES_Map.query("InputType == 'RESIDUAL'")['DemandSector'].values
    else:
        DS_ES_Map = loaders.get_parameter("DS_ES_Map")
        return DS_ES_Map.query("InputType == 'BOTTOMUP'")['DemandSector'].values


def existence_demand_parameter(name):
    ds = get_ds_list(name)
    ds = list(set(ds))
    args = [(name, d) for d in ds]
    valid = execute_in_process_pool(existence_demand_parameter_, args)
    return all(valid)


def existence_demand_parameter_(name, demand_sector):

    try:
        logger.info(f"Validating {name} from {demand_sector}")
        data = loaders.get_parameter(name,
                                     demand_sector=demand_sector)
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
                    f"Geographic granularity of BaseYearDemand for {d},{item} is diffecrent than specified in DS_ES_EC_DemandGranularity_Map.")
                return False
            if dg != grancols:
                logger.error(
                    f"Geographic granularity of DemandElasticity for {d},{item} is diffecrent than specified in DS_ES_EC_DemandGranularity_Map.")
                return False

    return True


def get_all_ES_Demand(ds, es):
    """returns dictionary of ES_Demand data for each ST.
    it returns dict with key as ST and ES_Demand as value
    """
    DS_ES_ST_Map = loaders.get_parameter("DS_ES_ST_Map")
    STs = [row[2:]
           for row in DS_ES_ST_Map if row[0] == ds and row[1] == es][0]
    return {s: loaders.get_parameter('ES_Demand',
                                     demand_sector=ds,
                                     energy_service=es,
                                     service_tech=s) for s in STs}


def read_header(filepath):
    with open(filepath) as f:
        csvf = csv.reader(f)
        return next(csvf)


def check_ES_Demand_columns():
    ds_es = get_bottomup_ds_es()
    valid = True
    for ds, es in ds_es:
        valid = valid and _check_ES_Demand_columns(ds, es)
    return valid


def get_structural_columns(ds):
    return constant.TIME_COLUMNS[utilities.get_valid_time_levels()[-1]] + \
        constant.GEO_COLUMNS[utilities.get_valid_geographic_levels()[-1]] + \
        get_cons_columns(ds)


def _check_ES_Demand_columns(ds, es):
    """checks if ES_Demand file has correct column names specified
    """
    DS_ES_ST_Map = loaders.get_parameter("DS_ES_ST_Map")
    STs = [row[2:]
           for row in DS_ES_ST_Map if row[0] == ds and row[1] == es][0]
    filepaths = {s: find_custom_DS_ES_filepath(ds,
                                               es,
                                               'ES_Demand',
                                               f"{s}_") for s in STs}
    valid = True
    for ST, path in filepaths.items():
        columns = read_header(path)
        structural = get_structural_columns(ds)
        other_cols = [c for c in columns if c not in structural]
        unexpected_cols = [c for c in other_cols if ST not in c]
        if unexpected_cols:
            logger.warning(
                f"Found unexpected columns {unexpected_cols} in {ST}_ES_Demand file")
        st_cols = [c for c in other_cols if ST in c]
        combinations = [set(c.split(constant.ST_SEPARATOR_CHAR))
                        for c in st_cols]
        if any([combinations.count(c) > 1 for c in combinations]):
            logger.error(
                "It is not allowed for two columns to have the exact same combinations of STs in {ST}_ES_Demand")
            valid = False
        sts = get_corresponding_sts(ds, es, ST)
        expected = fs.flatten([[set(x) for x in itertools.combinations(
            sts, n)] for n in range(1, len(sts)+1)])
        unexpected = [comb for comb in combinations if comb not in expected]
        if unexpected:
            logger.error(
                f"Found unexpected combination of STs in {ST}_ES_Demand")
            logger.error("Unexpected combination of STs in column {}".format(
                [constant.ST_SEPARATOR_CHAR.join(c) for c in unexpected]))
            valid = False

    return valid


def get_all_Penetration(ds, es):
    """returns all penetration data as dictionary with key as st, value as
    dictionary of ST combinations and actual penetration data.
    {"ST":{(ST1,ST2): penetration data for ST1 and ST2}
    """
    DS_ES_ST_Map = loaders.get_parameter("DS_ES_ST_Map")
    STs = [row[2:]
           for row in DS_ES_ST_Map if row[0] == ds and row[1] == es][0]
    d = {}
    for s in STs:
        es_demand = loaders.get_parameter('ES_Demand',
                                          demand_sector=ds,
                                          energy_service=es,
                                          service_tech=s)
        combs = [tuple(name.split(constant.ST_SEPARATOR_CHAR))
                 for name in es_demand.columns if s in name]
        d[s] = {tuple(c): loaders.get_parameter('Penetration',
                                                demand_sector=ds,
                                                energy_service=es,
                                                ST_combination=c) for c in combs}
    return d


def get_data(name, ds, es):
    if name in ['EfficiencyLevelSplit', 'NumInstances']:
        return {(ds, es): loaders.get_parameter(name,
                                                demand_sector=ds,
                                                energy_service=es)}
    elif name == "ES_Demand":
        return get_all_ES_Demand(ds, es)
    elif name == "Penetration":
        return get_all_Penetration(ds, es)
    else:
        logger.error(f"Unknown parameter {name}")


def validate_each_demand_param_(name, item, data, ds, es, st):
    """encapsulation over validate_each_demand_param to catch exception
    """
    logger.info(f"Validating {name} from {ds},{es} for {st}")
    try:
        v = validate_each_demand_param(name,
                                       data,
                                       demand_sector=ds,
                                       energy_service=es,
                                       service_tech=st)

        if not v:
            logger.error(
                f"Validaton failed for {name} from {ds},{es} for {st}")

            print(
                f"Validaton failed for {name} from {ds},{es} for {st}")
    except Exception as e:
        logger.exception(e)
        logger.error(
            f"{name} for {ds},{es},{item} has invalid data")
        print(e)
        v = False
    return v


def existence_demand_energy_service_parameter(name):
    """checks existence and basic data validation of
    EfficiencyLevelSplit,NumInstances,ES_Demand,Penetration
    """
    ds_es = get_bottomup_ds_es()
    args = []
    for ds, es in ds_es:
        try:
            data_ = get_data(name, ds, es)

        except FileNotFoundError as fne:
            logger.error(f"{name} for {ds},{es} is not given")
            logger.exception(fne)
            return False

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

    valid = execute_in_process_pool(validate_each_demand_param_, args)
    #valid = [validate_each_demand_param_(*item) for item in args]
    return all(valid)


def validate_each_demand_param(name, data, **kwargs):
    """Validates individual parameter according to specs given in yml file.
    """
    specs = filemanager.demand_specs()[name]

    if specs.get("optional", False) and isinstance(data, type(None)):
        return True
    
    return loaders.validate_param(name,
                                  specs,
                                  data,
                                  "rumi.io.demand",
                                  **kwargs)


def subset(data, indexnames, items):
    if isinstance(items, str):
        items = (items,)

    q = " & ".join([f"{name} == '{item}'" for name,
                    item in zip(indexnames, items)])
    return data.query(q)


def check_efficiency_levels(data, param_name, *args):
    ST_Info = loaders.get_parameter("ST_Info")
    st_info = ST_Info.set_index('ServiceTech')
    valid = True
    els = data.set_index('ServiceTech')
    for service_tech in els.index.unique():
        df = els.loc[service_tech]
        levels = len(df['EfficiencyLevelName'].unique())
        n = st_info.loc[service_tech]['NumEfficiencyLevels']
        v = n == levels
        valid = valid and v
        if not v:
            logger.error(
                f"For {param_name} in {args}, efficiency levels do not match for {service_tech}")
    return valid


def check_EfficiencyLevelSplit_granularity():
    return _check_DS_ES_granularity("EfficiencyLevelSplit")


def check_NumInstances_granularity():
    return _check_DS_ES_granularity("NumInstances")


def get_bottomup_ds_es():
    DS_ES_Map = loaders.get_parameter("DS_ES_Map")
    ds_es = DS_ES_Map.query("InputType == 'BOTTOMUP'")[
        ['DemandSector', 'EnergyService']].values
    return ds_es


def get_nonbottomup_ds_es():
    DS_ES_Map = loaders.get_parameter("DS_ES_Map")
    ds_es = DS_ES_Map.query("InputType != 'BOTTOMUP'")[
        ['DemandSector', 'EnergyService']].values
    return ds_es


def check_granularity_per_entity(d,
                                 entity,
                                 GeographicGranularity,
                                 TimeGranularity,
                                 ConsumerGranularity=None):
    """checks granuarity only. i.e. only columns are checked.
    contents of columns are not validated here.
    """

    geo_columns, time_columns, cons_columns = [], [], []

    if GeographicGranularity:
        geo_columns = common.get_geographic_columns(GeographicGranularity)
        dataset_columns = [c for c in d.columns if c in constant.GEOGRAPHIES]
    if TimeGranularity:
        time_columns = common.get_time_columns(TimeGranularity)
        dataset_columns.extend(
            [c for c in d.columns if c in constant.TIME_SLICES])
    if ConsumerGranularity:
        cons_columns = constant.CONSUMER_COLUMNS[ConsumerGranularity]
        dataset_columns.extend(
            [c for c in d.columns if c in constant.CONSUMER_TYPES])

    diff1 = set(geo_columns + time_columns +
                cons_columns) - set(dataset_columns)
    diff2 = set(dataset_columns) - \
        set(geo_columns + time_columns + cons_columns)

    valid = True

    if diff2:
        c, r = d[list(diff2)].shape
        empty = d[list(diff2)].isnull().sum().sum() == c*r

        if not empty:
            logger.debug(f"Granularity is finer than expected for {entity}!")

        return valid

    if diff1:
        logger.error(f"{diff1} not found in data for {entity}")
        valid = False
    else:
        allcols = geo_columns+time_columns + cons_columns
        nonempty = d[allcols].isnull().sum().sum()

        valid = nonempty == 0
        if not valid:
            logger.error(f"one of columns {allcols} is empty for {entity}.")

    return valid


def check_demand_granularity(param_name,
                             CSTAR=None,
                             GSTAR=None,
                             TSTAR=None,
                             check_function=check_granularity_per_entity):
    """
    Checks whether given data follows granularity as specified in granularity
    map. data file directly inside demand sector folder is tested using this
    function.
    """

    if CSTAR == None and GSTAR == None and TSTAR == None:
        raise Exception(
            "check_granularity function must have valid GSTAR/TSTAR argument")

    granularity_map = loaders.get_parameter("DS_ES_EC_DemandGranularity_Map")
    granularity = granularity_map.set_index(['DemandSector',
                                             'EnergyService',
                                             'EnergyCarrier'])
    dslist = get_ds_list(param_name)
    valid = True
    for ds in dslist:
        data = loaders.get_parameter(param_name, demand_sector=ds)
        data = data.set_index(['EnergyService', 'EnergyCarrier'])
        data.sort_index(inplace=True)
        logger.debug(f"Checking granularity of {param_name} for {ds}")
        for item in data.index.unique():
            d = subset(data, data.index.names, item)
            d = utilities.filter_empty(d)

            entity = (ds,) + item
            g = granularity.loc[entity]
            ConsumerGranularity = None
            GeographicGranularity, TimeGranularity = None, None
            if CSTAR:
                ConsumerGranularity = get_consumer_granularity(ds,
                                                               g['ConsumerGranularity'])
            if GSTAR:
                GeographicGranularity = g['GeographicGranularity']
            if TSTAR:
                TimeGranularity = g['TimeGranularity']

            v = check_function(d,
                               entity,
                               GeographicGranularity,
                               TimeGranularity,
                               ConsumerGranularity)
            valid = valid and v
            if not v:
                logger.error(
                    f"Granularity check failed for {param_name} for {entity}")

    return valid


def get_corresponding_sts(demand_sector,
                          energy_service,
                          service_tech):
    DS_ES_ST_Map = loaders.get_parameter('DS_ES_ST_Map')
    ST_Info = loaders.get_parameter('ST_Info')

    STs = fs.flatten([row[2:] for row in DS_ES_ST_Map if row[0] ==
                      demand_sector and row[1] == energy_service and service_tech in row])

    ST_Info = ST_Info.set_index('ServiceTech')
    EC = ST_Info.loc[service_tech]['EnergyCarrier']
    return [s for s in STs if ST_Info.loc[s]['EnergyCarrier'] == EC]


def coarsest(gran_map, ds):
    c = min(gran_map.to_dict(orient='records'),
            key=lambda x: len(constant.CONSUMER_COLUMNS[get_consumer_granularity(ds, x['ConsumerGranularity'])]))['ConsumerGranularity']
    g = min(gran_map['GeographicGranularity'].values,
            key=lambda x: len(constant.GEO_COLUMNS[x]))
    t = min(gran_map['TimeGranularity'].values,
            key=lambda x: len(constant.TIME_COLUMNS[x]))
    return c, g, t


def _check_DS_ES_granularity(param_name):
    """
    Checks whether EfficiencyLevelSplit/NumInstances follows granularity as specified in granularity map.
    """

    granularity_map = loaders.get_parameter("DS_ST_Granularity_Map")
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
            sts = get_corresponding_sts(ds, es, ST)
            g = granularity_map.query(
                f"DemandSector == '{ds}' & ServiceTech in {sts}")
            ConsumerGranularity, GeographicGranularity, TimeGranularity = coarsest(
                g, ds)
            v = utilities.check_granularity_per_entity(d,
                                                       ST,
                                                       GeographicGranularity,
                                                       TimeGranularity,
                                                       ConsumerGranularity)
            valid = valid and v
            if not v:
                logger.error(
                    f"Granularity check failed for {param_name} for {ST}")

    return valid


def _check_ES_Demand_granularity(param_name):
    """
    Checks whether ES_Demand follows granularity as
    specified in granularity map.
    """

    granularity_map = loaders.get_parameter("DS_ES_EC_DemandGranularity_Map")
    granularity = granularity_map.set_index(['DemandSector',
                                             'EnergyService',
                                             'EnergyCarrier'])
    ds_es = get_bottomup_ds_es()
    valid = True
    for ds, es in ds_es:
        data_ = get_data(param_name, ds, es)
        for ST, data in data_.items():
            d_ = data
            if not isinstance(data, dict):
                d_ = {ST: data}

            for item, df in d_.items():
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                d = utilities.filter_empty(df)
                g = granularity.loc[(ds, es, ST_to_EC(ST))]
                ConsumerGranularity = get_consumer_granularity(ds,
                                                               g['ConsumerGranularity'])
                GeographicGranularity = g['GeographicGranularity']
                TimeGranularity = g['TimeGranularity']
                v = utilities.check_granularity_per_entity(d,
                                                           item,
                                                           GeographicGranularity,
                                                           TimeGranularity,
                                                           ConsumerGranularity)
                valid = valid and v
                if not v:
                    logger.error(
                        f"Granularity check failed for {param_name} for {ST}, {item}")

    return valid


def _check_Penetration_granularity(param_name):
    """
    Checks whether Penetration follows granularity as
    specified in granularity map.
    """

    granularity_map = loaders.get_parameter("DS_ST_Granularity_Map")
    ds_es = get_bottomup_ds_es()
    valid = True
    for ds, es in ds_es:
        data_ = get_data(param_name, ds, es)
        for ST, data in data_.items():
            d_ = data
            if not isinstance(data, dict):
                d_ = {ST: data}

            for comb, df in d_.items():
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                d = utilities.filter_empty(df)
                g = granularity_map.query(
                    f"DemandSector == '{ds}' & ServiceTech in {comb}")
                ConsumerGranularity, GeographicGranularity, TimeGranularity = coarsest(
                    g, ds)
                v = utilities.check_granularity_per_entity(d,
                                                           ST,
                                                           GeographicGranularity,
                                                           TimeGranularity,
                                                           ConsumerGranularity)

                valid = valid and v
                if not v:
                    logger.error(
                        f"Granularity check failed for {param_name} for {ST}, {comb}")

    return valid


def check_ES_Demand_granularity():
    return _check_ES_Demand_granularity("ES_Demand")


def check_Penetration_granularity():
    return _check_Penetration_granularity("Penetration")


def check_numconsumers_granularity():
    """
    Checks whether NumConsumers data follows granularity as specified in
    granularity map.
    """

    granularity = loaders.get_parameter("DS_Cons1_Map")
    param_name = 'NumConsumers'
    dslist = get_ds_list(param_name)
    valid = True
    for ds in dslist:
        data = loaders.get_parameter(param_name, demand_sector=ds)
        d = utilities.filter_empty(data)
        g = granularity[ds]
        ConsumerGranularity = get_cons_columns(ds)[-1].upper()
        GeographicGranularity = g[0]
        TimeGranularity = g[1]

        v = check_granularity_per_entity(data,
                                         (ds, "NumConsumers"),
                                         GeographicGranularity,
                                         TimeGranularity,
                                         ConsumerGranularity)
        valid = valid and v
        if not v:
            logger.error(
                f"Granularity check failed for {param_name} for {ds}")

    return valid


def save_output(compute_demand):
    """decorator function to be applied to demand computation function.
    it will save the results to required destination and return same.
    also if results are alredy present then it will read the results and
    return, instaed of computing it again.
    """
    def get_columns(d):
        """This is just to reorder columns"""
        tgc = [c for c in constant.TIME_SLICES +
               constant.GEOGRAPHIES + constant.CONSUMER_TYPES if c in d.columns]
        other = [c for c in d.columns if c not in tgc]
        return tgc + other

    @functools.wraps(compute_demand)
    def wrapper(*args):
        output_path = filemanager.get_output_path("Demand")
        filename = "_".join(args+('Demand',))
        path = os.path.join(output_path, ".".join([filename, "csv"]))

        if os.path.exists(path):
            result = pd.read_csv(path)
        else:
            result = compute_demand(*args)
            result = result[get_columns(result)]
            result.to_csv(path, index=False)
        return result

    return wrapper


def check_ST_granularity():
    """ checks if granuarity is coarser than corresponding granularty in
    DS_ES_EC_DemandGranularity_Map for derived DS,ES,EC from DS,ST

    Returns
    -------
    True or False
    """
    DS_ES_EC_DemandGranularity_Map = loaders.get_parameter(
        "DS_ES_EC_DemandGranularity_Map")
    DS_ST_Granularity_Map = loaders.get_parameter("DS_ST_Granularity_Map")

    DS_ST_Granularity_Map = DS_ST_Granularity_Map.set_index(
        ["DemandSector", "ServiceTech"])
    for item in DS_ST_Granularity_Map.index:
        consgran = DS_ST_Granularity_Map.loc[item]['ConsumerGranularity']
        geogran = DS_ST_Granularity_Map.loc[item]['GeographicGranularity']
        timegran = DS_ST_Granularity_Map.loc[item]['TimeGranularity']
        ds, st = item
        ec = ST_to_EC(st)
        ds_es_ec = DS_ES_EC_DemandGranularity_Map.query(
            f"DemandSector == '{ds}' & EnergyCarrier == '{ec}'")

        def _check_gran(gran, grantype='TimeGranularity',
                        grandata=list(constant.TIME_COLUMNS.keys())):
            gran_ = min(
                ds_es_ec[grantype].values, key=grandata.index)
            if not coarser(gran, gran_, grandata):
                logger.error(
                    f"In DS_ST_Granularity_Map, for <{ds}, {st}> {grantype} should be coarser than or equal to {gran_}")
                return False
            return True

        c = _check_gran(consgran, 'ConsumerGranularity',
                        list(constant.CONSUMER_COLUMNS.keys()))
        g = _check_gran(geogran, 'GeographicGranularity',
                        list(constant.GEO_COLUMNS.keys()))
        t = _check_gran(timegran, 'TimeGranularity',
                        list(constant.TIME_COLUMNS.keys()))
        if not all([c, g, t]):
            return False

    return True


def check_total_penetration():
    """checks if penetrations for each ST together with which it can
    appear totals less than or equal to 1
    """
    precision = 1e-6
    ds_es = get_bottomup_ds_es()
    valid = True
    for ds, es in ds_es:
        DS_ES_ST_Map = loaders.get_parameter("DS_ES_ST_Map")
        STs = [row[2:]
               for row in DS_ES_ST_Map if row[0] == ds and row[1] == es][0]

        for s in STs:
            es_demand = loaders.get_parameter('ES_Demand',
                                              demand_sector=ds,
                                              energy_service=es,
                                              service_tech=s)
            combs = [tuple(name.split(constant.ST_SEPARATOR_CHAR))
                     for name in es_demand.columns if s in name]
            p = [loaders.get_parameter('Penetration',
                                       demand_sector=ds,
                                       energy_service=es,
                                       ST_combination=c) for c in combs]
            indexcols = utilities.get_all_structure_columns(p[0])
            p = [item.set_index(indexcols) for item in p]
            v = (functools.reduce(
                lambda x, y: x+y, [item['Penetration'] for item in p], 0) > 1 + precision).sum() == 0
            if not v:
                print(f"Penetration for {combs} sums more than 1!")
                logger.error(f"Penetration for {combs} sums more than 1!")
                
            valid = valid and v
                
    return valid


if __name__ == "__main__":
    pass
