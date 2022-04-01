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
""" Compute demand for various types of DemandSector and EnergyService combinations.

"""
import sys
import os
import shutil
import functools
import logging
import pandas as pd
import numpy as np
import click
import time
import csv
from rumi.io import config
from rumi.io import loaders
from rumi.io import constant
from rumi.io import demand as demandio
from rumi.io import functionstore as fs
from rumi.io import common
from rumi.io import filemanager
from rumi.io.logger import init_logger, get_event
from rumi.processing import utilities
from rumi.io.utilities import groupby_time
import rumi.io.utilities as ioutils
from rumi.processing.utilities import get_geographic_columns_from_dataframe
from rumi.io.common import get_geographic_columns
from rumi.io.common import get_time_columns
from rumi.io.multiprocessutils import execute_in_process_pool
logger = logging.getLogger(__name__)


@demandio.save_output
def compute_demand(demand_sector, energy_service, energy_carrier):
    """Compute demand for given set of demand_sector, energy_service, energy_carrier.

    :param: demand_sector: str -> Demand sector name
    :param: energy_service: str -> Energy serive name
    :param: energy_carrier: str -> Energy carrier name
    :returns: computed demand as a DataFrame

    """
    logger.info(
        f"Computing demand for {demand_sector},{energy_service},{energy_carrier}")
    type_ = demandio.get_type(demand_sector, energy_service)
    if type_ == "EXTRANEOUS":
        r = compute_extraneous_demand(demand_sector,
                                      energy_service,
                                      energy_carrier)
    elif type_ == "BOTTOMUP":
        r = compute_bottomup_demand(demand_sector,
                                    energy_service,
                                    energy_carrier)
    elif type_ == "RESIDUAL":
        r = compute_residual_demand(demand_sector,
                                    energy_service,
                                    energy_carrier)
    else:
        r = compute_gdpelasticity_demand(demand_sector,
                                         energy_service,
                                         energy_carrier)
    r = utilities.seasonwise_timeslices(r, 'EnergyDemand')
    return ioutils.order_rows(r)


def get_consumer_granularity(demand_sector,
                             energy_service,
                             energy_carrier):
    DS_ES_EC_DemandGranularity_Map = loaders.get_parameter(
        "DS_ES_EC_DemandGranularity_Map")
    granularity_map = DS_ES_EC_DemandGranularity_Map.set_index(['DemandSector',
                                                                'EnergyService',
                                                                'EnergyCarrier'])
    return granularity_map.loc[(demand_sector,
                                energy_service,
                                energy_carrier)]['ConsumerGranularity']


def get_time_granularity(demand_sector,
                         energy_service,
                         energy_carrier):
    DS_ES_EC_DemandGranularity_Map = loaders.get_parameter(
        "DS_ES_EC_DemandGranularity_Map")
    granularity_map = DS_ES_EC_DemandGranularity_Map.set_index(['DemandSector',
                                                                'EnergyService',
                                                                'EnergyCarrier'])
    return granularity_map.loc[(demand_sector,
                                energy_service,
                                energy_carrier)]['TimeGranularity']


def get_cons_columns(demand_sector,
                     energy_service,
                     energy_carrier):
    consgran = get_consumer_granularity(demand_sector,
                                        energy_service,
                                        energy_carrier)
    if consgran == 'CONSUMERALL':
        return demandio.get_cons_columns(demand_sector)
    return constant.CONSUMER_COLUMNS[consgran]


def get_query(energy_service, energy_carrier):
    return f"(EnergyService == '{energy_service}') & (EnergyCarrier == '{energy_carrier}')"


def compute_extraneous_demand(demand_sector,
                              energy_service,
                              energy_carrier):
    balancing_area = demandio.balancing_area(energy_carrier)
    balancing_time = demandio.balancing_time(energy_carrier)
    consumer_cols = get_cons_columns(
        demand_sector, energy_service, energy_carrier)
    groupcols = consumer_cols + list(get_geographic_columns(balancing_area))
    df = loaders.get_parameter("ExtraneousDemand", demand_sector=demand_sector)
    q = get_query(energy_service, energy_carrier)
    df = ioutils.filter_empty(df.query(q))
    demand = groupby_time(df, groupcols, balancing_time,
                          'EnergyDemand')
    return demand.drop(['EnergyService', 'EnergyCarrier'], axis=1)


def compute_gdp_rate(GDP):
    """computes gdp rate from actual gdp values
    """

    geocols = get_geographic_columns_from_dataframe(GDP)
    gdp = GDP.set_index(geocols)
    gdp = gdp.sort_index()
    index = gdp.index.unique()
    d = []
    for item in index:
        gdp_ = gdp.loc[item]
        gdp_ = gdp_.sort_values('Year')
        gdp_values = gdp_['GDP'].values
        year = gdp_['Year'].values[1:]
        gdp_rate = [gdp_ratio(gdp_values[i],  gdp_values[i-1])
                    for i, g in enumerate(gdp_values[1:], start=1)]
        df = pd.DataFrame({"Year": year,
                           "GDP_RATE": gdp_rate},
                          index=gdp_.index[1:])
        d.append(df)
    return pd.concat(d).reset_index()


def gdp_ratio(current, previous):
    return (current - previous)/previous


def get_demand_elasticity(demand_sector,
                          energy_service,
                          energy_carrier):
    q = get_query(energy_service, energy_carrier)
    df = loaders.get_parameter("DemandElasticity", demand_sector=demand_sector)
    return ioutils.filter_empty(df.query(q))


def get_base_year_demand(demand_sector,
                         energy_service,
                         energy_carrier):
    q = get_query(energy_service, energy_carrier)
    df = loaders.get_parameter("BaseYearDemand", demand_sector=demand_sector)
    return ioutils.filter_empty(df.query(q))


def compute_demand_growth_rate(demand_sector,
                               energy_service,
                               energy_carrier):
    """Multiplication of gdpelasticity and gdp_rate
    """
    geogran = demandio.get_geographic_granularity(demand_sector,
                                                  energy_service,
                                                  energy_carrier)

    elasticity = get_demand_elasticity(demand_sector,
                                       energy_service,
                                       energy_carrier)
    GDP = loaders.get_parameter("GDP")
    g_geocols = get_geographic_columns_from_dataframe(GDP)

    conscols = get_cons_columns(demand_sector, energy_service, energy_carrier)
    if conscols:  #
        elasticity = elasticity.set_index(conscols)
        gre = []
        for con in elasticity.index.unique():
            elasticity_ = elasticity.loc[con].reset_index()
            df = growth_rate_elasticity(GDP,
                                        elasticity_,
                                        conscols)
            n = len(df)
            for i, c in enumerate(conscols):
                df[c] = [con[i]]*n if len(conscols) > 1 else [con]*n
            gre.append(df)
        return pd.concat(gre)
    else:
        return growth_rate_elasticity(GDP, elasticity)


def growth_rate_elasticity(GDP, elasticity, conscols=[]):
    g_geocols = get_geographic_columns_from_dataframe(GDP)
    e_geocols = get_geographic_columns_from_dataframe(elasticity)
    other_cols = [c for c in e_geocols if c not in g_geocols]

    if other_cols:
        return growth_rate_elasticity_finer(GDP, elasticity,
                                            g_geocols,
                                            other_cols)
    else:
        return growth_rate_elasticity_coarser(GDP, elasticity, e_geocols)


def growth_rate_elasticity_finer(GDP, elasticity, g_geocols, indexcols):
    elasticity = elasticity.set_index(indexcols)
    elasticity.sort_index(level=1)
    gdp_rate = compute_gdp_rate(GDP)

    d = []

    for geo in elasticity.index.unique():
        # compute for every extra geographic granularity
        e_ = elasticity.loc[geo]
        e = e_.set_index(g_geocols + ['Year'])
        key = e.index.droplevel('Year')
        ukey = key.unique()
        if isinstance(ukey[0], tuple):
            ukey = key[0]
        q = " & ".join([f"({name} == '{ukey[i]}')" for i,
                        name in enumerate(key.names)])

        gr = gdp_rate.query(q)
        gr = gr.set_index(g_geocols + ['Year'])
        r = (gr['GDP_RATE']*e['Elasticity'])
        r.rename('GROWTH_RATE', inplace=True)
        df = r.reset_index()
        n = len(df)

        if isinstance(geo, tuple):
            df = pd.DataFrame([dict(zip(indexcols, geo))]*n).join(df)
        else:
            df = pd.DataFrame([dict(zip(indexcols, (geo,)))]*n).join(df)
        d.append(df)
    return pd.concat(d)


def growth_rate_elasticity_coarser(GDP, elasticity, e_geocols):
    gdp = GDP.groupby(e_geocols + ['Year']).sum().reset_index()
    gdp_rate = compute_gdp_rate(gdp)

    gdp_rate = gdp_rate.set_index(e_geocols + ['Year'])

    e = elasticity.set_index(e_geocols + ['Year'])
    r = e['Elasticity']*gdp_rate['GDP_RATE']
    r.rename('GROWTH_RATE', inplace=True)
    return r.reset_index()


def gdp_energy_demand(year, growth_rate, base_demand):
    """computes demand based on gdpgrowth and BaseYearDemand.

    it uses follwing recursion
    Demand[y] = DemandGrowthRate[y] * Demand[y-1]
    this actually resolves to
    Demand[y] = product(DemandGrowthRate[all x such x<=y]) * BaseYearDemand
    """
    bd = base_demand
    r = (growth_rate.query(f"Year <= {year}")['GROWTH_RATE']+1).prod()*bd
    return r


def compute_gdpelasticity_demand(demand_sector,
                                 energy_service,
                                 energy_carrier):
    """computes demand based on gdpgrowth and BaseYearDemand.

    it uses follwing recursion
    Demand[y] = DemandGrowthRate[y] * Demand[y-1]
    this actually resolves to
    Demand[y] = product(DemandGrowthRate[all x such x <= y]) * BaseYearDemand
    """
    demand_growth_rate = compute_demand_growth_rate(demand_sector,
                                                    energy_service,
                                                    energy_carrier)

    base_year_demand = get_base_year_demand(demand_sector,
                                            energy_service,
                                            energy_carrier)

    geogran = demandio.get_geographic_granularity(demand_sector,
                                                  energy_service,
                                                  energy_carrier)
    geocols = get_geographic_columns(geogran)
    geocols_dgr = get_geographic_columns_from_dataframe(demand_growth_rate)
    conscols = get_cons_columns(demand_sector, energy_service, energy_carrier)

    timegran = get_time_granularity(demand_sector,
                                    energy_service,
                                    energy_carrier)
    timecols = get_time_columns(timegran)
    indexcols = conscols + geocols + timecols

    base_year_demand = base_year_demand.set_index(conscols+geocols + timecols)

    gr = demand_growth_rate.set_index(conscols + geocols_dgr)
    gr = gr.sort_index(level=1)
    years = demand_growth_rate['Year'].unique()

    demand_dfs = []

    for item in base_year_demand.index.unique():
        index = base_year_demand.index
        key = tuple([item[index.names.index(c)]
                     for c in conscols + geocols_dgr])
        g = gr.loc[key]
        bd = base_year_demand.loc[item]
        base_demand = bd['BaseYearDemand']
        d = []
        demand = [gdp_energy_demand(y, g, base_demand) for y in years]
        indexnames = [n for n in indexcols if n != 'Year']
        year_index = indexcols.index('Year')
        indexvalues = tuple([x for i, x in enumerate(item) if i != year_index])
        index = pd.MultiIndex.from_tuples([indexvalues for y in years],
                                          names=indexnames)
        df = pd.DataFrame({"EnergyDemand": demand,
                           "Year": years},
                          index=index)

        demand_dfs.append(df)

    return pd.concat(demand_dfs).reset_index()


def filepath_demand_servicetech(demand_sector,
                                energy_service,
                                service_tech,
                                energy_carrier):
    output_path = filemanager.get_output_path("Demand")
    args = (demand_sector,
            energy_service,
            service_tech,
            energy_carrier)
    filename = "_".join(args+('Demand',))
    path = os.path.join(output_path, ".".join([filename, "csv"]))

    return path


class BottomupDemand:
    """ class to compute demand for <DS,ES,EC> which are of BOTTOMUP type
    """

    def save_st_output(demand=True):
        def save_st_output_(method):
            """decorator function to save results of energy demand per ST
            """

            @functools.wraps(method)
            def wrapper(instance, service_tech):
                if demand:
                    path = filepath_demand_servicetech(instance.demand_sector,
                                                       instance.energy_service,
                                                       service_tech,
                                                       instance.energy_carrier)
                    r = method(instance, service_tech)
                    result = r.rename(
                        'EnergyDemand', inplace=True).reset_index()
                    result = ioutils.order_rows(
                        utilities.seasonwise_timeslices(result, 'EnergyDemand'))

                    cols = ioutils.get_ordered_cols(result)
                    result[cols].to_csv(path, index=False)
                    return r
                else:
                    path = os.path.join(filemanager.get_output_path(
                        "Demand"), ".".join([f"TotalNumInstances_{service_tech}", "csv"]))
                    result = method(instance, service_tech)
                    if os.path.exists(path):
                        result = pd.concat([pd.read_csv(path), result])
                    cols = ioutils.get_ordered_cols(result)
                    cols.remove('DemandSector')
                    cols.insert(0, 'DemandSector')
                    result[cols].to_csv(path, index=False)
                    return result

            return wrapper

        return save_st_output_

    def __init__(self, demand_sector, energy_service, energy_carrier):
        self.demand_sector = demand_sector
        self.energy_service = energy_service
        self.energy_carrier = energy_carrier
        self.find_index_cols()

        self.__get_NumConsumers()
        self.__get_ST_Efficiency()
        self.__get_EfficiencyLevelSplit()
        self.__get_NumInstances()

    def compute_demand(self):
        service_techs = demandio.get_service_techs(self.demand_sector,
                                                   self.energy_service,
                                                   self.energy_carrier)
        total_demand = []

        for service_tech in service_techs:
            logger.debug(f"Computing for ServiceTech, {service_tech}")
            esu_demand = self.compute_esu_demand(service_tech)
            total_demand.append(esu_demand)

        d = sum_series(total_demand)
        d.rename('EnergyDemand', inplace=True)
        return d

    def save_tot_num_instances(self):
        service_techs = demandio.get_service_techs(self.demand_sector,
                                                   self.energy_service,
                                                   self.energy_carrier)
        total_demand = []

        for service_tech in service_techs:
            logger.debug(
                f"Computing NumAppliances for ServiceTech, {service_tech}")
            self.save_tot_num_instances_for_service_tech(service_tech)

    @save_st_output(demand=False)
    def save_tot_num_instances_for_service_tech(self, service_tech):
        ds = self.demand_sector
        es = self.energy_service

        ES_Demand = loaders.get_parameter('ES_Demand',
                                          demand_sector=ds,
                                          energy_service=es,
                                          service_tech=service_tech)
        ES_Demand = ES_Demand.set_index(self.demandindex)

        combinations = {name: tuple(name.split(constant.ST_SEPARATOR_CHAR))
                        for name in ES_Demand.columns if service_tech in name}

        appliances = []
        q = f"ServiceTech == '{service_tech}'"
        sts = demandio.get_corresponding_sts(self.demand_sector,
                                             self.energy_service,
                                             service_tech)

        indexcols = self.get_index_cols(sts)
        num_consumers = self.NumConsumers['NumConsumers']
        num_instances = self.NumInstances.set_index(
            indexcols).query(q)['NumInstances']

        ST_efficiency = self.ST_Efficiency.query(q)

        for name, comb in combinations.items():
            logger.debug(f"ServiceTech combination, {name}")
            p = loaders.get_parameter('Penetration',
                                      demand_sector=ds,
                                      energy_service=es,
                                      ST_combination=comb)
            p = p.set_index(self.get_index_cols(comb))

            d = num_consumers * p['Penetration'] * num_instances
            appliances.append(d)
        tot_num_instances = sum_series(appliances)

        efficiency_level_split = self.EfficiencyLevelSplit.set_index(
            indexcols).query(q)

        n = []
        for level in efficiency_level_split['EfficiencyLevelName'].unique():
            q = f'EfficiencyLevelName == "{level}"'
            split_share = efficiency_level_split.query(q)['SplitShare']
            df = tot_num_instances*split_share
            df = df.rename("TotalNumInstances").reset_index()
            df['EfficiencyLevelName'] = level
            n.append(df)

        r = pd.concat(n)
        r['DemandSector'] = self.demand_sector
        return r

    @save_st_output(demand=True)
    def compute_esu_demand(self, service_tech):
        """
        Compute ESU demand for given service_tech
        """
        ds = self.demand_sector
        es = self.energy_service

        ES_Demand = loaders.get_parameter('ES_Demand',
                                          demand_sector=ds,
                                          energy_service=es,
                                          service_tech=service_tech)
        ES_Demand = ES_Demand.set_index(self.demandindex)

        combinations = {name: tuple(name.split(constant.ST_SEPARATOR_CHAR))
                        for name in ES_Demand.columns if service_tech in name}
        esu_demand = []
        logger.debug(f"Total Combinations, {combinations}")
        sts = demandio.get_corresponding_sts(self.demand_sector,
                                             self.energy_service,
                                             service_tech)
        indexcols = self.get_index_cols(sts)

        for name, comb in combinations.items():
            logger.debug(f"ServiceTech combination, {name}")

            p = loaders.get_parameter('Penetration',
                                      demand_sector=ds,
                                      energy_service=es,
                                      ST_combination=comb)
            p = p.set_index(self.get_index_cols(comb))

            e = self.compute_esu_demand_per_comb(service_tech,
                                                 p['Penetration'],
                                                 ES_Demand[name],
                                                 indexcols)
            esu_demand.append(e)

        return sum_series(esu_demand)

    def compute_esu_demand_per_comb(self,
                                    service_tech,
                                    penetration,
                                    es_demand,
                                    indexcols):
        """compute ESU demand for particular combination with other
        service_tech
        """
        q = f"ServiceTech == '{service_tech}'"
        num_consumers = self.NumConsumers['NumConsumers']
        num_instances = self.NumInstances.set_index(
            indexcols).query(q)['NumInstances']
        efficiency_level_split = self.EfficiencyLevelSplit.set_index(
            indexcols).query(q)
        ST_efficiency = self.ST_Efficiency.query(q)

        esu_demand = []
        for level in ST_efficiency['EfficiencyLevelName'].unique():
            q = f'EfficiencyLevelName == "{level}"'
            efficiency = ST_efficiency.query(q)['Efficiency']
            split_share = efficiency_level_split.query(q)['SplitShare']

            d = num_consumers * penetration * num_instances * \
                es_demand * split_share * efficiency

            esu_demand.append(d)

        return sum_series(esu_demand)

    def __get_NumInstances(self):
        ds = self.demand_sector
        es = self.energy_service
        self.NumInstances = loaders.get_parameter('NumInstances',
                                                  demand_sector=ds,
                                                  energy_service=es)
        # self.NumInstances = NumInstances.set_index(self.indexcols)

    def __get_EfficiencyLevelSplit(self):
        ds = self.demand_sector
        es = self.energy_service
        self.EfficiencyLevelSplit = loaders.get_parameter('EfficiencyLevelSplit',
                                                          demand_sector=ds,
                                                          energy_service=es)
        # self.EfficiencyLevelSplit = efsplit.set_index(self.indexcols)

    def __get_ST_Efficiency(self):
        ST_Efficiency = loaders.get_parameter('ST_Efficiency',
                                              demand_sector=self.demand_sector)
        self.ST_Efficiency = ST_Efficiency.set_index('Year')
        # always given at Year level

    def get_index_cols(self, sts):
        ST_Granularity_Map = loaders.get_parameter('DS_ST_Granularity_Map')
        st_gran = ST_Granularity_Map.set_index(['DemandSector', 'ServiceTech'])

        g = ST_Granularity_Map.query(
            f"DemandSector == '{self.demand_sector}'  & ServiceTech in {sts}")
        consgran, ggran, tgran = demandio.coarsest(g, self.demand_sector)
        t = constant.TIME_COLUMNS[tgran]
        g = constant.GEO_COLUMNS[ggran]
        c = constant.CONSUMER_COLUMNS[consgran]
        return c + g + t

    def find_index_cols(self):
        geogran = demandio.get_geographic_granularity(self.demand_sector,
                                                      self.energy_service,
                                                      self.energy_carrier)
        geocols = get_geographic_columns(geogran)
        timegran = get_time_granularity(self.demand_sector,
                                        self.energy_service,
                                        self.energy_carrier)
        timecols = get_time_columns(timegran)
        self.conscols = get_cons_columns(self.demand_sector,
                                         self.energy_service,
                                         self.energy_carrier)

        self.demandindex = self.conscols + geocols + timecols

    def __get_NumConsumers(self):
        NumConsumers = loaders.get_parameter('NumConsumers',
                                             demand_sector=self.demand_sector)
        indexcols = self.conscols + self.get_index_cols_num_consumers()
        self.NumConsumers = NumConsumers.set_index(indexcols)

    def get_index_cols_num_consumers(self):
        demand_sector = self.demand_sector
        DS_Cons1_Map = loaders.get_parameter('DS_Cons1_Map')
        geogran = DS_Cons1_Map[demand_sector][0]
        timegran = DS_Cons1_Map[demand_sector][1]
        return get_geographic_columns(geogran) + get_time_columns(timegran)


def compute_bottomup_demand(demand_sector,
                            energy_service,
                            energy_carrier):
    bottomup = BottomupDemand(demand_sector,
                              energy_service,
                              energy_carrier)

    r = bottomup.compute_demand().reset_index()
    return r


def sum_series(items):
    """
    vector adds multiple pd.Series given as list like collection
    """
    s = items[0]
    for item in items[1:]:
        s = s + item
    return s


def get_non_bottomup_tuples(demand_sector,  energy_service, energy_carrier):
    DS_ES_EC_Map = loaders.get_parameter('DS_ES_EC_Map')
    return [row[:3] for row in DS_ES_EC_Map if row[0] ==
            demand_sector and row[2] == energy_carrier and row[1] != energy_service]


def get_bottomup_tuples(demand_sector, energy_carrier):
    DS_ES_ST_Map = loaders.get_parameter('DS_ES_ST_Map')
    DS_ES_ST = [(row[0], row[1], item)
                for row in DS_ES_ST_Map for item in row[2:] if row[0] == demand_sector]

    ST_Info = loaders.get_parameter('ST_Info')
    for ds, es, st in DS_ES_ST:
        ecs = ST_Info.query(
            f"EnergyCarrier == '{energy_carrier}' & ServiceTech == '{st}'")
        if len(ecs) > 0:
            yield ds, es, energy_carrier


def aggregate_demand(demand, aggcols):
    """
    Aggregates non residual demand to coarser level for
    corresponding residual demand.
    """
    # can be optimized if needed, aggregate only if required.
    return demand.reset_index().groupby(aggcols)['EnergyDemand'].sum()


def compute_residual_demand(demand_sector,
                            energy_service,
                            energy_carrier):
    def get_index_cols():
        dataset_cols = residual_demand.columns
        all_indexcols = constant.TIME_SLICES + \
            constant.GEOGRAPHIES + constant.CONSUMER_TYPES

        return [c for c in all_indexcols if c in dataset_cols]

    ResidualDemand = loaders.get_parameter("ResidualDemand",
                                           demand_sector=demand_sector)

    q = f"EnergyCarrier == '{energy_carrier}' & EnergyService == '{energy_service}'"
    residual_demand = ResidualDemand.query(q)
    indexcols = get_index_cols()
    residual_share = residual_demand.set_index(indexcols)['ResidualShare']

    non_bottomup = get_non_bottomup_tuples(demand_sector,
                                           energy_service,
                                           energy_carrier)

    bottomup = set(get_bottomup_tuples(demand_sector,
                                       energy_carrier))

    s = sum_series([aggregate_demand(compute_demand(*item),
                                     indexcols)
                    for item in non_bottomup+list(bottomup)])

    rs = residual_share*s
    rs.rename('EnergyDemand', inplace=True)
    return rs.reset_index()[indexcols + ['EnergyDemand']]


def coarsest(ds_es_ec, take_cons_cols):
    if take_cons_cols:
        c = min(ds_es_ec.to_dict(orient='records'),
                key=lambda x: len(get_cons_columns(x['DemandSector'], x['EnergyService'], x['EnergyCarrier'])))
        conscols = get_cons_columns(
            c['DemandSector'], c['EnergyService'], c['EnergyCarrier'])
    else:
        conscols = []
    g = min(ds_es_ec['GeographicGranularity'].values,
            key=lambda x: len(constant.GEO_COLUMNS[x]))
    t = min(ds_es_ec['TimeGranularity'].values,
            key=lambda x: len(constant.TIME_COLUMNS[x]))

    return constant.TIME_COLUMNS[t] + constant.GEO_COLUMNS[g] + conscols


def get_group_cols_for_ec(ec):
    g = get_geographic_columns(demandio.balancing_area(ec))
    t = get_time_columns(demandio.balancing_time(ec))
    return t+g


def finest_balancing_area_time(ds_es_ec):
    EC = ds_es_ec['EnergyCarrier'].unique()
    g = [get_geographic_columns(demandio.balancing_area(ec)) for ec in EC]
    t = [get_time_columns(demandio.balancing_time(ec)) for ec in EC]

    g = max(g, key=len)
    t = max(t, key=len)
    return t+g


def compute_demand_balancing(demand_sector, energy_service, energy_carrier):
    """aggregates the demand over balancing area and balancing time
    """
    cols = get_group_cols_for_ec(energy_carrier)
    return compute_demand_with_coarseness(demand_sector,
                                          energy_service,
                                          energy_carrier,
                                          cols)


def compute_demand_with_coarseness(demand_sector,
                                   energy_service,
                                   energy_carrier,
                                   columns):
    """aggregates the demand over the granularity given by columns
    """
    demand = compute_demand(demand_sector, energy_service, energy_carrier)
    return demand.groupby(columns).sum()


def compute_demand_with_coarseness_service_techs(demand_sector,
                                                 energy_service,
                                                 energy_carrier,
                                                 columns):
    """aggregates the demand with service_tech, over the granularity given by columns

    Parameters
    -----------
    demand_sector: str
    energy_service: str
    energy_carrier: str
    columns: list

    Returns
    -------
       a dataframe
    """
    demand = compute_demand(demand_sector, energy_service, energy_carrier)
    service_techs_files = find_service_tech_demand_filepath(demand_sector,
                                                            energy_service,
                                                            energy_carrier)
    if not service_techs_files:
        return ioutils.order_rows(demand.groupby(columns).sum().reset_index())
    else:
        d = []
        for st in service_techs_files:
            df = pd.read_csv(service_techs_files[st]).groupby(
                columns).sum().reset_index()
            df['ServiceTech'] = st
            d.append(ioutils.order_rows(df))
        return pd.concat(d)


def find_service_tech_demand_filepath(demand_sector,
                                      energy_service,
                                      energy_carrier):
    service_techs = demandio.get_service_techs(demand_sector,
                                               energy_service,
                                               energy_carrier)
    return {st: filepath_demand_servicetech(demand_sector,
                                            energy_service,
                                            st,
                                            energy_carrier) for st in service_techs}


def save_coarsest(entity_names=('EnergyService', 'EnergyCarrier')):
    """Aggregates Demand for EnergyService,EnergyCarrier over all DemandSectors
    """
    ds_es_ec_columns = ['DemandSector', 'EnergyService', 'EnergyCarrier']
    DS_ES_EC_DemandGranularity_Map = loaders.get_parameter(
        "DS_ES_EC_DemandGranularity_Map")
    entities = DS_ES_EC_DemandGranularity_Map.set_index(
        entity_names).index

    for items in entities:
        if isinstance(items, str):
            items = (items,)

        logging.debug(
            f"Writing results for " + ",".join(items) + " across all demand sectors")
        filepath = os.path.join(
            filemanager.get_output_path("Demand"), "_".join(items + ('Demand.csv',)))
        if os.path.exists(filepath):
            logger.debug(f"{filepath} exists, skipping")
            continue

        q = " & ".join([f"{name} == '{item}'" for name,
                        item in zip(entity_names, items)])
        ds_es_ec_map = DS_ES_EC_DemandGranularity_Map.query(q)
        ds_es_ec = ds_es_ec_map.set_index(ds_es_ec_columns).index
        cols = coarsest(ds_es_ec_map, "DemandSector" in entity_names)

        target_cols = ["EnergyDemand",
                       'SeasonEnergyDemand'] if 'Season' in cols else ['EnergyDemand']

        with open(filepath, "w", newline='') as f:
            csvf = csv.DictWriter(f, dialect='excel',
                                  fieldnames=cols + target_cols)
            csvf.writeheader()
            d = []
            for ds, es, ec in ds_es_ec:
                d.append(compute_demand_with_coarseness(ds, es, ec, cols))
            df = utilities.seasonwise_timeslices(sum_series(d).reset_index(),
                                                 'EnergyDemand')
            df = ioutils.order_rows(df)
            csvf.writerows(df.to_dict(orient='records'))


def check_energy_service_service_tech_count(ds_es_ec_map, ds_es_ec):
    """checks if combination of DS-EC-ES-ST has at least two items in it
    """
    if len(ds_es_ec_map) > 1:  # either multiple energy services
        return True
    else:
        ds, es, ec = next(iter(ds_es_ec))  # this will have only one item in it
        # or check multiple service_techs in next line
        return len(find_service_tech_demand_filepath(ds, es, ec)) > 1


def save_coarsest_with_ST():
    """Aggregates Demand for EnergyService,EnergyCarrier over all DemandSectors
    """
    entity_names = ['DemandSector', 'EnergyCarrier']
    ds_es_ec_columns = ['DemandSector', 'EnergyService', 'EnergyCarrier']
    DS_ES_EC_DemandGranularity_Map = loaders.get_parameter(
        "DS_ES_EC_DemandGranularity_Map")
    entities = DS_ES_EC_DemandGranularity_Map.set_index(
        entity_names).index
    print(f"Writing results across all demand sectors with ServiceTech and EnergyService")

    for items in entities:
        if isinstance(items, str):
            items = (items,)

        logging.debug(
            f"Writing results for " + ",".join(items) +
            " across all demand sectors with ServiceTech and EnergyService")
        filepath = os.path.join(
            filemanager.get_output_path("Demand"), "_".join(items + ('EnergyService_ServiceTech_Demand.csv',)))
        if os.path.exists(filepath):
            logger.debug(f"{filepath} exists, skipping")
            continue

        q = " & ".join([f"{name} == '{item}'" for name,
                        item in zip(entity_names, items)])
        ds_es_ec_map = DS_ES_EC_DemandGranularity_Map.query(q)
        ds_es_ec = ds_es_ec_map.set_index(ds_es_ec_columns).index
        cols = coarsest(ds_es_ec_map, "DemandSector" in entity_names)

        if not check_energy_service_service_tech_count(
                ds_es_ec_map.set_index(ds_es_ec_columns),
                ds_es_ec):
            continue

        target_cols = ['EnergyService',
                       'ServiceTech',
                       "EnergyDemand"]
        if 'Season' in cols:
            target_cols += ['SeasonEnergyDemand']

        with open(filepath, "w", newline='') as f:
            csvf = csv.DictWriter(f, dialect='excel',
                                  fieldnames=cols + target_cols)
            csvf.writeheader()
            d = []
            for ds, es, ec in ds_es_ec:
                df = compute_demand_with_coarseness_service_techs(
                    ds, es, ec, cols)
                df['EnergyService'] = es
                csvf.writerows(df.to_dict(orient='records'))


def complete_run():
    """runs validation and complete demand processing flow.
    """
    if config.get_config_value("validation") == 'True':
        validate()

    compute_demand_all()
    path = save_by_balancing_area_time()
    save_coarsest(entity_names=['DemandSector', 'EnergyCarrier'])
    save_coarsest(entity_names=['EnergyCarrier'])
    save_coarsest(entity_names=['EnergyService', 'EnergyCarrier'])
    save_coarsest_with_ST()
    copy_supply_parameter(path)


def validate():
    """Validate Common and Demand parameters
    returns True or False
    """
    valid = loaders.validate_params("Common")
    if not valid:
        raise Exception("Validation failed for Common")
    valid = valid and loaders.validate_params("Demand")
    if not valid:
        raise Exception("Validation failed for Demand")


def get_residual_services(ds_es_ec):
    return [(ds, es, ec) for ds, es, ec in ds_es_ec if demandio.get_type(ds, es) == "RESIDUAL"]


def get_non_residual_services(ds_es_ec):
    return [(ds, es, ec) for ds, es, ec in ds_es_ec if demandio.get_type(ds, es) != "RESIDUAL"]


def save_total_num_instances(ds_es_ec):
    bottomups = [(ds, es, ec) for ds, es,
                 ec in ds_es_ec if demandio.get_type(ds, es) == "BOTTOMUP"]
    for ds, es, ec in bottomups:
        b = BottomupDemand(ds, es, ec)
        b.save_tot_num_instances()


def compute_demand_all():
    """Computes demand for all combinations of demand_sector, energy_service
    and energy_carrier
    """
    DS_ES_EC_DemandGranularity_Map = loaders.get_parameter(
        "DS_ES_EC_DemandGranularity_Map")

    ds_es_ec = DS_ES_EC_DemandGranularity_Map.set_index(
        ['DemandSector', 'EnergyService', 'EnergyCarrier']).index

    path_all = os.path.join(
        filemanager.get_output_path("Demand"), "EndUseDemandEnergy.csv")

    fine_cols = ['EnergyCarrier'] + \
        finest_balancing_area_time(DS_ES_EC_DemandGranularity_Map) + \
        ['EndUseDemandEnergy', 'TotalEnergyDemand']

    print("Computing demand")
    logger.info("Computing demand")
    nonres_ds_es_ec = get_non_residual_services(ds_es_ec)
    logger.info("Computing non RESIDUAL demand")
    execute_in_process_pool(compute_demand, nonres_ds_es_ec)

    res_ds_es_ec = get_residual_services(ds_es_ec)
    logger.info("Computing RESIDUAL demand")
    execute_in_process_pool(compute_demand, res_ds_es_ec)

    logger.info("Computing total num instances")
    save_total_num_instances(ds_es_ec)


def check_valid_ds_es_ec(demand_sector,
                         energy_service,
                         energy_carrier):
    valid = True
    DS_ES_EC_DemandGranularity_Map = loaders.get_parameter(
        "DS_ES_EC_DemandGranularity_Map")
    ds_es_ec = DS_ES_EC_DemandGranularity_Map.set_index(
        ['DemandSector', 'EnergyService', 'EnergyCarrier']).index

    if demand_sector not in DS_ES_EC_DemandGranularity_Map.DemandSector.unique():
        print(f"Invalid value for demand_sector parameter, {demand_sector}")
        valid = False
    if energy_service not in DS_ES_EC_DemandGranularity_Map.EnergyService.unique():
        print(f"Invalid value for energy_service parameter, {energy_service}")
        valid = False
    if energy_carrier not in DS_ES_EC_DemandGranularity_Map.EnergyCarrier.unique():
        print(f"Invalid value for energy_carrier parameter, {energy_carrier}")
        valid = False
    if (demand_sector, energy_service, energy_carrier) not in ds_es_ec:
        print("Invalid combination of demand_sector, energy_service, energy_carrier")
        valid = False

    return valid


def save_by_balancing_area_time():
    """daves demand at balancing area and balancing time in EndUseDemandEnergy.csv
    """
    DS_ES_EC_DemandGranularity_Map = loaders.get_parameter(
        "DS_ES_EC_DemandGranularity_Map")

    ds_es_ec = DS_ES_EC_DemandGranularity_Map.set_index(
        ['DemandSector', 'EnergyService', 'EnergyCarrier']).index

    path_all = os.path.join(
        filemanager.get_output_path("Demand"), "EndUseDemandEnergy.csv")

    fine_cols = ['EnergyCarrier'] + \
        finest_balancing_area_time(DS_ES_EC_DemandGranularity_Map) + \
        ['EndUseDemandEnergy', 'TotalEnergyDemand']

    print("Writing results by balancing area and balancing time")
    logger.info("Writing results by balancing area and balancing time")
    with open(path_all, "w", newline='') as f:
        csvf = csv.DictWriter(f, dialect='excel', fieldnames=fine_cols)
        csvf.writeheader()
        for EC in DS_ES_EC_DemandGranularity_Map['EnergyCarrier'].unique():
            ds_es_EC = [(ds, es, ec)
                        for ds, es, ec in ds_es_ec if ec == EC]
            d = []
            for ds, es, ec in ds_es_EC:
                d.append(compute_demand_balancing(ds,
                                                  es,
                                                  EC))
            s = sum_series(d)
            df = s.reset_index()
            df = utilities.seasonwise_timeslices(df,
                                                 'EnergyDemand')
            if 'SeasonEnergyDemand' in df.columns:
                df = df.rename(
                    columns={'SeasonEnergyDemand': 'TotalEnergyDemand'})
            else:
                df['TotalEnergyDemand'] = df['EnergyDemand']
            df = df.rename(columns={'EnergyDemand': 'EndUseDemandEnergy'})

            df['EnergyCarrier'] = [EC]*len(df)
            df = ioutils.order_rows(df)
            csvf.writerows(df.to_dict(orient='records'))

    return path_all


def copy_supply_parameter(srcpath):
    """Copies EndUseDemandEnergy to supply paramters data
    """
    supplypath = filemanager.find_filepath('EndUseDemandEnergy')
    scenario = config.get_config_value("scenario")
    scenario_ = os.path.join("Scenarios", scenario)
    path = supplypath.replace("Global Data", scenario_)
    if os.path.exists(path):
        warning = f"Results of previous run are present at {path}. Results will be overwritten, press n to cancel, enter to continue (y/n)"
        check = input(warning)
        if check and check.strip() and check.strip().lower()[0] == "n":
            return
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copy(srcpath, path)


def clean_output():
    """Deletes files from output directory.
    """
    outputpath = filemanager.get_output_path("Demand")
    if os.listdir(outputpath):
        warning = f"Results of previous run are present at {outputpath}. Results will be overwritten, press n to cancel, enter to continue (y/n)"
        check = input(warning)
        if check and check.strip() and check.strip().lower()[0] == "n":
            sys.exit(1)
        for p in [os.path.join(outputpath, f)
                  for f in os.listdir(outputpath)]:

            os.unlink(p)


def rumi_demand(model_instance_path: str,
                scenario: str,
                output: str,
                demand_sector: str,
                energy_service: str,
                energy_carrier: str,
                logger_level: str,
                numthreads: int,
                validation: bool):
    """Computes demand for given demand_sector,energy_service.energy_carrier. if none
    is given , it computes demand for all combinations.


    :param: model_instance_path : str
              Path where model instance is stored
    :param: scenario : str
              Name of Scenario
    :param: output : str
              Custom output folder
    :param: demand_sector : str
              Name of demand sector
    :param: energy_service : str
              Name of energy service
    :param: energy_carrier : str
              Name of energy carrier
    :param: logger_level : str
              Level for logging,one of INFO,WARN,DEBUG,ERROR
    :param: numthreads : int
              Number of threads/processes
    :param: validation : bool
              Whether to do validation

    """
    if not loaders.sanity_check_cmd_args("Demand",
                                         model_instance_path,
                                         scenario,
                                         logger_level,
                                         numthreads,
                                         cmd='rumi_demand'):
        return

    global logger
    config.initialize_config(model_instance_path, scenario)
    if output:
        config.set_config("output", output)
    config.set_config("numthreads", str(numthreads))
    config.set_config("validation", str(validation))

    if not all([demand_sector, energy_service, energy_carrier]):
        clean_output()

    init_logger("Demand", logger_level)
    logger = logging.getLogger("rumi.processing.demand")

    try:
        ds_es_ec = [demand_sector, energy_service, energy_carrier]
        if all(ds_es_ec):
            if not check_valid_ds_es_ec(demand_sector,
                                        energy_service,
                                        energy_carrier):
                pass
            else:
                print(compute_demand(demand_sector,
                                     energy_service,
                                     energy_carrier))
        elif any(ds_es_ec):  # any but not all!
            print(
                "Partial inputs given for demand_sector, energy_service and energy_carrier")
            print(
                "demand_sector, energy_service and energy_carrier , either all should be given or none.")
            return
        else:
            complete_run()
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        time.sleep(1)
        get_event().set()


@click.command()
@click.option("-m", "--model_instance_path",
              type=click.Path(exists=True),
              help="Path of the model instance root folder")
@click.option("-s", "--scenario",
              help="Name of the scenario within specified model")
@click.option("-o", "--output",
              help="Path of the output folder",
              default=None)
@click.option("-D", "--demand_sector",
              help="Name of the demand sector",
              default=None)
@click.option("-E", "--energy_service",
              help="Name of the energy service",
              default=None)
@click.option("-C", "--energy_carrier",
              help="Name of the energy carrier",
              default=None)
@click.option("-l", "--logger_level",
              help="Level for logging: one of INFO, WARN, DEBUG or ERROR (default: INFO)",
              default="INFO")
@click.option("-t", "--numthreads",
              help="Number of threads/processes (default: 2)",
              default=2)
@click.option("--validation/--no-validation",
              help="Enable/disable validation (default: Enabled)",
              default=True)
def _main(model_instance_path,
          scenario: str,
          output: str,
          demand_sector: str,
          energy_service: str,
          energy_carrier: str,
          logger_level: str,
          numthreads: int,
          validation: bool):
    """Command line interface for processing demand inputs.

    -m/--model_instance_path and -s/--scenario are mandatory arguments, while the
    others are optional.

    if demand_sector, energy_service, energy_carrier options are provided, demand
    will be computed only for given demand_sector, energy_service, energy_carrier
    combinations. If these parameters are not given (none of them),
    then demand will be processed for all demand_sector, energy_service and
    energy_carrier combinations.
    """
    rumi_demand(model_instance_path=model_instance_path,
                scenario=scenario,
                output=output,
                demand_sector=demand_sector,
                energy_service=energy_service,
                energy_carrier=energy_carrier,
                logger_level=logger_level,
                numthreads=numthreads,
                validation=validation)


def main():
    if len(sys.argv) == 1:
        print("To see valid options  run the command with --help")
        print("rumi_demand --help")
    else:
        _main()


if __name__ == "__main__":
    main()
