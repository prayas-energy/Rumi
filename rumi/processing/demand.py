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
import operator
from pathlib import Path
from pathlib import Path
from rumi.io import config
from rumi.io import loaders
from rumi.io import constant
from rumi.io import demand as demandio
from rumi.io import functionstore as fs
from rumi.io import common
from rumi.io import filemanager
from rumi.io.logger import init_logger, get_event, get_queue
from rumi.processing import utilities
from rumi.processing import emission
from rumi.io.utilities import groupby_time
import rumi.io.utilities as ioutils
from rumi.processing.utilities import get_geographic_columns_from_dataframe
from rumi.io.common import get_geographic_columns
from rumi.io.common import get_time_columns
from rumi.io.multiprocessutils import execute_in_process_pool
logger = logging.getLogger(__name__)
print = functools.partial(print, flush=True)


def create_base(root, *args):
    """Creates all folders in root as chain of subfolders.
    This function handles if directory is already present.
    But we try to call this function from multiple process/threads
    simulteniously when there are no existing folders then it fails.
    To handle this we create all output folders beforehand from main
    process/thread.
    """
    def mkdir(root, subfolder):
        p = root / subfolder
        if not p.exists():
            p.mkdir()
        return p

    p = functools.reduce(mkdir, args, root)
    return p


def get_output_filepath(entity_names, entity_values, postfix="", output_type="Demand"):
    """Returns demand output filepath
    """
    if entity_names is None:
        entity_names = []
    if entity_values is None:
        entity_values = []
    data = dict(zip(entity_names, entity_values))

    def get_base_folder():
        root = Path(filemanager.get_output_path("Demand"))
        if len(entity_names) == 1 and entity_names[0] == "ServiceTech":
            return create_base(root, "TotalNumInstances")
        elif len(entity_names) == 1 and entity_names[0] == "EnergyCarrier":
            return create_base(root, *entity_names)
        elif "DemandSector" in entity_names and "EnergyService" in entity_names:
            return create_base(root, "DemandSector",
                               data['DemandSector'], data['EnergyService'])
        elif "EnergyService" in entity_names:
            return create_base(root, "EnergyService", data['EnergyService'])
        elif "DemandSector" in entity_names:
            return create_base(root, "DemandSector", data['DemandSector'])
        else:
            return root

    def get_filename():
        if output_type in ["NonEnergyEmissions", "EndUseEmissions-DemandMetDomesticEC"]:
            return f"{output_type}.csv"
        elif not entity_names and not entity_values:
            return "EndUseDemandEnergy.csv"
        elif len(entity_names) == 1 and entity_names[0] == "ServiceTech":
            return "_".join(("TotalNumInstances", entity_values[0] + ".csv"))
        elif postfix:
            return "_".join(entity_values + (postfix, f"{output_type}.csv"))
        else:
            return "_".join(entity_values + (f"{output_type}.csv",))

    return os.path.join(get_base_folder(), get_filename())


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
        names = "DemandSector", "EnergyService", "EnergyCarrier"
        path = get_output_filepath(names, args)

        if os.path.exists(path):
            result = pd.read_csv(path)
        else:
            result = compute_demand(*args)
            result = result[get_columns(result)]
            result.to_csv(path, index=False)
        return result

    return wrapper


def apply_demand_profile_without_reset_index(demand, GTProfile):
    dindex = ioutils.get_all_structure_columns(demand)
    profile_index = ioutils.get_all_structure_columns(GTProfile)
    # common_ is always coarser than profile_index
    common_ = [c for c in dindex if c in profile_index]
    demand = demand.set_index(dindex)
    gtprofile = GTProfile.set_index(profile_index)
    total = ioutils.groupby(GTProfile, common_, "GTProfile")
    d = demand['EnergyDemand']*gtprofile['GTProfile']/total
    d.name = "EnergyDemand"
    return d


def apply_demand_profile_(demand, GTProfile):
    d = apply_demand_profile_without_reset_index(demand, GTProfile)
    return d.reset_index()


def apply_demand_profile(demand, demand_sector, energy_service, energy_carrier):
    """Apply GTProfile if it exists for given demand_sector and energy_carrier
    """
    GTProfile = loaders.get_parameter("GTProfile",
                                      demand_sector=demand_sector,
                                      energy_service=energy_service,
                                      energy_carrier=energy_carrier)
    if isinstance(GTProfile, pd.DataFrame):
        logger.debug(
            f"Applying demand profile for {demand_sector},{energy_service},{energy_carrier}")
        r = apply_demand_profile_(demand, GTProfile)
        return r

    else:
        return demand


def demand_profile(f):

    @functools.wraps(f)
    def wrapper(demand_sector, energy_service, energy_carrier):
        r = f(demand_sector, energy_service, energy_carrier)
        return apply_demand_profile(r, demand_sector, energy_service, energy_carrier)

    return wrapper


@save_output
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
    if type_ == "EXOGENOUS":
        r = compute_exogenous_demand(demand_sector,
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
    DS_ES_EC_combined_map = demandio.get_combined_granularity_map()
    q = " & ".join([f"DemandSector == '{demand_sector}'",
                    f"EnergyService == '{energy_service}'",
                    f"EnergyCarrier == '{energy_carrier}'"])
    cgran = DS_ES_EC_combined_map.query(q).iloc[0, :]['ConsumerGranularity']
    return cgran


def get_time_granularity(demand_sector,
                         energy_service,
                         energy_carrier):
    DS_ES_EC_Map = loaders.get_parameter(
        "DS_ES_EC_Map",
        demand_sector=demand_sector)
    granularity_map = DS_ES_EC_Map.set_index(['DemandSector',
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
    return constant.CONSUMER_COLUMNS[consgran]


def get_query(energy_service, energy_carrier):
    return f"(EnergyService == '{energy_service}') & (EnergyCarrier == '{energy_carrier}')"


@demand_profile
def compute_exogenous_demand(demand_sector,
                             energy_service,
                             energy_carrier):
    df = loaders.get_parameter("ExogenousDemand", demand_sector=demand_sector)
    q = get_query(energy_service, energy_carrier)
    df = ioutils.filter_empty(df.query(q))
    return df.drop(['EnergyService', 'EnergyCarrier'], axis=1)


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
    gdp = GDP.groupby(e_geocols + ['Year']
                      ).sum(numeric_only=True).reset_index()
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


@demand_profile
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
    return get_output_filepath(("DemandSector", "EnergyService", "ServiceTech", "EnergyCarrier"), (demand_sector, energy_service, service_tech, energy_carrier))


def save_st_output(demand=True):
    def save_st_output_(method):
        """decorator function to save results of energy demand per ST
        """

        def save_demand(method, instance, service_tech):
            ds, es, ec = instance.demand_sector, instance.energy_service, instance.energy_carrier
            names = ("DemandSector", 'EnergyService',
                     'ServiceTech', 'EnergyCarrier')
            values = instance.demand_sector, instance.energy_service, service_tech, instance.energy_carrier
            path = get_output_filepath(names, values)
            r = method(instance, service_tech)
            result = r.copy()
            result = result.rename(
                'EnergyDemand', inplace=True)
            GTProfile = loaders.get_parameter("GTProfile",
                                              demand_sector=ds,
                                              energy_service=es,
                                              energy_carrier=ec)
            if isinstance(GTProfile, pd.DataFrame):
                if 'ServiceTech' in GTProfile.columns:
                    if service_tech in GTProfile.ServiceTech.values:
                        gtprofile = ioutils.filter_empty(
                            GTProfile.query(f"ServiceTech=='{service_tech}'"))
                    else:
                        gtprofile = ioutils.filter_empty(
                            GTProfile[GTProfile.ServiceTech.isna()])
                else:
                    gtprofile = GTProfile
                logger.debug(
                    f"Applying demand profile for {ds},{es},{ec},{service_tech}")
                result = apply_demand_profile_without_reset_index(result.reset_index(),
                                                                  gtprofile)

            return_result = result.copy()
            result = ioutils.order_rows(
                utilities.seasonwise_timeslices(result.reset_index(), 'EnergyDemand'))

            cols = ioutils.get_ordered_cols(result)
            result[cols].to_csv(path, index=False)
            return return_result

        @functools.wraps(method)
        def wrapper(instance, service_tech):
            if demand:
                return save_demand(method, instance, service_tech)
            else:
                path = get_output_filepath(
                    ('ServiceTech',), (service_tech,))
                result = method(instance, service_tech)
                if os.path.exists(path):
                    result = pd.concat([pd.read_csv(path), result])
                cols = ioutils.get_ordered_cols(result)
                cols.remove('DemandSector')
                cols.remove('EnergyService')
                cols.insert(0, 'EnergyService')
                cols.insert(0, 'DemandSector')
                result[cols].to_csv(path, index=False)
                return result

        return wrapper

    return save_st_output_


class BottomupDemand:
    """ class to compute demand for <DS,ES,EC> which are of BOTTOMUP type
    """

    def __init__(self, demand_sector, energy_service, energy_carrier):
        self.demand_sector = demand_sector
        self.energy_service = energy_service
        self.energy_carrier = energy_carrier

        self.__get_NumConsumers()
        self.__load_parameters()

    def compute_quantity(self):
        service_tech_categories = demandio.get_service_tech_categories(self.demand_sector,
                                                                       self.energy_service)
        total_demand = []

        for service_tech_category in service_tech_categories:
            logger.debug(
                f"Computing for ServiceTechCategory, {service_tech_category}")
            esu_demand = self.compute_esu_quantity(service_tech_category)
            total_demand.append(esu_demand)

        d = sum_series(total_demand)
        d.rename('EnergyDemand', inplace=True)
        return d

    def save_tot_num_instances(self):
        service_tech_categories = demandio.get_service_tech_categories(self.demand_sector,
                                                                       self.energy_service)

        for service_tech_category in service_tech_categories:
            logger.debug(
                f"Computing NumAppliances for ServiceTech, {service_tech_category}")
            self.save_tot_num_instances_for_service_tech_category(
                service_tech_category)

    def get_indexed(self, df, query):
        """sets appropriate index and returns indexed dataframe
        """
        d = ioutils.filter_empty(df.query(query))
        cols = ioutils.get_all_structure_columns(d)
        if cols:
            return d.set_index(cols)

    def save_tot_num_instances_for_service_tech_category(self, service_tech_category):
        @save_st_output(demand=False)
        def compute_num_instances_per_service_tech(self, service_tech):
            tsr = self.get_tech_split_ratio(service_tech_category,
                                            service_tech)

            q = f"ServiceTech == '{service_tech}'"
            num_instances = self.get_indexed(self.NumInstances, q)[
                'NumInstances']
            efficiency_level_split = self.get_indexed(
                self.EfficiencyLevelSplit, q)

            n = []
            for level in efficiency_level_split['EfficiencyLevelName'].unique():
                q = f'EfficiencyLevelName == "{level}"'
                split_share = efficiency_level_split.query(
                    q)['EfficiencySplitShare']
                df = tot_num_instances * split_share * tsr * num_instances
                df = df.rename("TotalNumInstances").reset_index()
                df['EfficiencyLevelName'] = level
                n.append(ioutils.order_rows(df))

            r = pd.concat(n)
            r['DemandSector'] = self.demand_sector
            r['EnergyService'] = self.energy_service
            return r

        ds = self.demand_sector
        es = self.energy_service

        ES_Demand = loaders.get_parameter('ES_Demand',
                                          demand_sector=ds,
                                          energy_service=es,
                                          service_tech_category=service_tech_category)
        demandindex = self.find_demand_index_cols(service_tech_category)
        ES_Demand = ES_Demand.set_index(demandindex)

        combinations = {name: tuple(name.split(constant.ST_SEPARATOR_CHAR))
                        for name in ES_Demand.columns if service_tech_category in name}

        num_consumers = self.get_NumConsumers(service_tech_category)

        appliances = []
        for name, comb in combinations.items():
            logger.debug(f"ServiceTechCategory combination, {name}")
            p = loaders.get_parameter('UsagePenetration',
                                      demand_sector=ds,
                                      energy_service=es,
                                      STC_combination=comb)
            p = p.set_index(ioutils.get_all_structure_columns(p))

            d = num_consumers * p['UsagePenetration']
            appliances.append(d)
        tot_num_instances = sum_series(appliances)

        service_techs = demandio.STC_to_STs(service_tech_category)
        STs = [
            ST for ST in service_techs if self.energy_carrier in demandio.ST_to_ECs(ST)]
        if STs:
            for ST in STs:
                compute_num_instances_per_service_tech(self, ST)

    def get_tech_split_ratio(self,
                             service_tech_category,
                             service_tech):
        qs = f"ServiceTech == '{service_tech}'"
        qc = f"ServiceTechCategory == '{service_tech_category}'"
        tech_split_ratio = self.get_indexed(self.TechSplitRatio,
                                            " & ".join([qs, qc]))
        return tech_split_ratio['TechSplitRatio']

    @save_st_output(demand=True)
    def compute_esu_quantity_ST(self, service_tech):
        return self.compute_esu_quantity_ST_(service_tech)

    def compute_esu_quantity_ST_(self, service_tech):
        """Little hack for backward compatibility.
        First argument self is not to make this as method. it has to
        be passed explicitly just like calling any function
        """
        ds = self.demand_sector
        es = self.energy_service
        service_tech_category = demandio.ST_to_STC(service_tech)

        ES_Demand = loaders.get_parameter('ES_Demand',
                                          demand_sector=ds,
                                          energy_service=es,
                                          service_tech_category=service_tech_category)
        demandindex = self.find_demand_index_cols(service_tech_category)
        ES_Demand = ES_Demand.set_index(demandindex)

        combinations = {name: tuple(name.split(constant.ST_SEPARATOR_CHAR))
                        for name in ES_Demand.columns if service_tech_category in name}
        esu_demand = []
        logger.debug(f"Total Combinations, {combinations}")
        stcs = demandio.get_corresponding_stcs(self.demand_sector,
                                               self.energy_service,
                                               service_tech_category)

        for name, comb in combinations.items():
            logger.debug(f"Computing  {name},{comb}")

            u = loaders.get_parameter('UsagePenetration',
                                      demand_sector=ds,
                                      energy_service=es,
                                      STC_combination=comb)
            u = u.set_index(ioutils.get_all_structure_columns(u))
            e = self.compute_quantity_per_comb(service_tech_category,
                                               service_tech,
                                               u['UsagePenetration'],
                                               ES_Demand[name])
            esu_demand.append(e)

        return sum_series(esu_demand)

    def compute_esu_quantity(self, service_tech_category):
        """
        Compute ESU demand for given service_tech_category
        """
        service_techs = demandio.STC_to_STs(service_tech_category)
        STs = [
            ST for ST in service_techs if self.energy_carrier in demandio.ST_to_ECs(ST)]
        if STs:
            return sum_series([self.compute_esu_quantity_ST(ST) for ST in STs])
        else:
            return 0

    def compute_quantity_per_comb(self,
                                  service_tech_category,
                                  service_tech,
                                  usagepenetration,
                                  es_demand):
        """compute ESU demand for particular combination with other
        service_tech_category
        """
        logger.debug(f"Computing demand for ServiceTech, {service_tech}")
        q = f"ServiceTech == '{service_tech}'"
        num_consumers = self.get_NumConsumers(service_tech_category)
        num_instances = self.get_indexed(self.NumInstances, q)['NumInstances']
        efficiency_level_split = self.get_indexed(self.EfficiencyLevelSplit, q)

        q_ = " & ".join([q,
                         f"EnergyService == '{self.energy_service}'",
                         f"EnergyCarrier == '{self.energy_carrier}'"])
        ST_efficiency = self.get_indexed(self.ST_SEC, q_)
        tsr = self.get_tech_split_ratio(service_tech_category,
                                        service_tech)
        esu_demand = []

        for level in ST_efficiency['EfficiencyLevelName'].unique():
            q = f'EfficiencyLevelName == "{level}"'
            efficiency = ST_efficiency.query(
                q)['SpecificEnergyConsumption']
            split_share = efficiency_level_split.query(
                q)['EfficiencySplitShare']
            combined_es_demand = num_consumers * num_instances *\
                es_demand * usagepenetration * tsr  # multiply by emission factor
            # for nonenergy part from ST_EmissionDetails
            d = combined_es_demand * split_share * efficiency
            esu_demand.append(d)

        return sum_series(esu_demand)

    def __load_parameters(self):
        ds = self.demand_sector
        es = self.energy_service
        self.TechSplitRatio = loaders.get_parameter("TechSplitRatio",
                                                    demand_sector=ds,
                                                    energy_service=es)
        self.NumInstances = loaders.get_parameter('NumInstances',
                                                  demand_sector=ds,
                                                  energy_service=es)
        self.EfficiencyLevelSplit = loaders.get_parameter('EfficiencyLevelSplit',
                                                          demand_sector=ds,
                                                          energy_service=es)
        ST_SEC = loaders.get_parameter('ST_SEC',
                                       demand_sector=self.demand_sector)
        self.ST_SEC = ST_SEC  # ST_SEC.set_index('Year')

    def find_demand_index_cols(self, service_tech_category):
        """Index for ES_Demand
        """
        C, G, T = demandio.get_granularity("ES_Demand",
                                           demand_sector=self.demand_sector,
                                           energy_service=self.energy_service,
                                           service_tech_category=service_tech_category)
        conscols = ioutils.get_consumer_columns(C)
        geocols = get_geographic_columns(G)
        timecols = get_time_columns(T)
        return conscols + geocols + timecols

    def __get_NumConsumers(self):
        NumConsumers = loaders.get_parameter('NumConsumers',
                                             demand_sector=self.demand_sector)
        # indexcols = self.get_index_cols_num_consumers()
        self.NumConsumers = NumConsumers

    def get_NumConsumers(self, service_tech_category):
        DS_Cons1_Map = loaders.get_parameter("DS_Cons1_Map",
                                             demand_sector=self.demand_sector)

        if DS_Cons1_Map is None or self.demand_sector not in DS_Cons1_Map:
            return 1
        elif not fs.isnone(self.NumConsumers):
            def get_geo_cols(cols): return [
                c for c in cols if c in constant.GEOGRAPHIES]
            dgeocols = get_geo_cols(
                self.find_demand_index_cols(service_tech_category))
            indexcols = self.get_index_cols_num_consumers()
            geocols = get_geo_cols(indexcols)
            if len(dgeocols) < len(geocols):
                cols = [c for c in indexcols if c not in geocols] + dgeocols
                num_consumers = self.NumConsumers.groupby(
                    cols)['NumConsumers'].sum()
            else:
                num_consumers = self.NumConsumers.set_index(indexcols)[
                    'NumConsumers']

            return num_consumers
        else:
            return 1

    def get_index_cols_num_consumers(self):
        demand_sector = self.demand_sector
        DS_Cons1_Map = loaders.get_parameter('DS_Cons1_Map',
                                             demand_sector=self.demand_sector)
        geogran = DS_Cons1_Map[demand_sector][0]
        timegran = DS_Cons1_Map[demand_sector][1]
        if DS_Cons1_Map is None or demand_sector not in DS_Cons1_Map:
            conscols = []
        else:
            conscols = demandio.get_cons_columns(demand_sector)
        return conscols + get_geographic_columns(geogran) + get_time_columns(timegran)


class NonEnergyEmissions(BottomupDemand):

    def __init__(self, demand_sector, energy_service):
        super().__init__(demand_sector, energy_service, None)

    def compute_quantity(self):
        service_tech_categories = demandio.get_service_tech_categories(self.demand_sector,
                                                                       self.energy_service)
        total_demand = []

        for service_tech_category in service_tech_categories:
            logger.debug(
                f"Computing NonEnergyEmissions for ServiceTechCategory, {service_tech_category}")
            esu_demand = self.compute_esu_quantity(service_tech_category)
            if isinstance(esu_demand, pd.Series) and len(esu_demand) > 0:
                total_demand.append(esu_demand)

        if total_demand:
            d = pd.concat(total_demand)
            return d.reset_index()
        else:
            return 0

    def compute_esu_quantity_ST(self, service_tech):
        d = self.compute_esu_quantity_ST_(service_tech)
        return d

    def compute_esu_quantity(self, service_tech_category):
        service_techs = demandio.STC_to_STs(service_tech_category)
        if service_techs:
            emissions = [self.compute_esu_quantity_ST(ST)
                         for ST in service_techs]
            es = [e for e in emissions if isinstance(
                e, pd.Series) and len(e) > 0]
            if es:
                d = pd.concat(es)
                return d
            else:
                return 0
        else:
            return 0

    def compute_quantity_per_comb(self,
                                  service_tech_category,
                                  service_tech,
                                  usagepenetration,
                                  es_demand):
        """compute NonEnergyEmissions for particular combination with other
        service_tech_category
        """
        logger.debug(
            f"Computing NonEnergyEmissions for ServiceTech, {service_tech}")
        q = f"ServiceTech == '{service_tech}'"
        num_consumers = self.get_NumConsumers(service_tech_category)
        num_instances = self.get_indexed(self.NumInstances, q)
        if num_instances is not None:
            num_instances = num_instances['NumInstances']

        tsr = self.get_tech_split_ratio(service_tech_category,
                                        service_tech)
        esu_demand = []
        ST_EmissionDetails = loaders.get_parameter("ST_EmissionDetails",
                                                   demand_sector=self.demand_sector)
        if ST_EmissionDetails is not None:
            q = " & ".join([q, f"EnergyService == '{self.energy_service}'"])
            st_emissions = ST_EmissionDetails.query(q)
            st_emissions = st_emissions.set_index(['Year'])
            nonenergy_ems_factor = st_emissions[st_emissions.EnergyCarrier.isnull(
            )]
        if ST_EmissionDetails is not None and len(nonenergy_ems_factor) > 0 and num_instances is not None:
            combined_es_demand = num_consumers * num_instances * es_demand * \
                usagepenetration * tsr
            d = combined_es_demand * nonenergy_ems_factor['DomEmissionFactor']
            d.rename("NonEnergyEmissions", inplace=True)
            emissions = d.reset_index()
            emissions['ServiceTechCategory'] = service_tech_category
            emissions['ServiceTech'] = service_tech
            emissions.set_index(
                [c for c in emissions.columns if c != 'NonEnergyEmissions'], inplace=True)
            return emissions['NonEnergyEmissions']
        else:
            return 0


# @demand_profile
def compute_bottomup_demand(demand_sector,
                            energy_service,
                            energy_carrier):
    bottomup = BottomupDemand(demand_sector,
                              energy_service,
                              energy_carrier)

    r = bottomup.compute_quantity().reset_index()
    return r


def sum_series(items):
    """
    vector adds multiple pd.Series given as list like collection
    """
    indices = [set(s.index.names) for s in items if isinstance(s, pd.Series)]
    if not indices:
        return 0
    aggcols = ioutils.order_columns(functools.reduce(lambda x, y: x & y,
                                                     indices,
                                                     indices[0]))
    if any(len(aggcols) < len(s.index.names) for s in items if isinstance(s, pd.Series)):
        items = [s if isinstance(s, (float, int)) else ioutils.groupby(
            s.reset_index(), aggcols, s.name) for s in items]

    s = items[0]
    for item in items[1:]:
        s = s + item
    return s


def get_non_bottomup_tuples(demand_sector,  energy_service, energy_carrier):
    DS_ES_EC_Map = loaders.get_parameter('DS_ES_EC_Map',
                                         demand_sector=demand_sector)[['DemandSector',
                                                                       'EnergyService',
                                                                       'EnergyCarrier']]

    return [row[:3] for row in DS_ES_EC_Map.values if row[0] ==
            demand_sector and row[2] == energy_carrier and row[1] != energy_service]


def get_bottomup_tuples(demand_sector, energy_carrier):
    DS_ES_STC_DemandGranularityMap = loaders.get_parameter(
        'DS_ES_STC_DemandGranularityMap',
        demand_sector=demand_sector)
    DS_ES_STC = DS_ES_STC_DemandGranularityMap[['DemandSector',
                                                'EnergyService',
                                               'ServiceTechCategory']]

    for ds, es, stc in DS_ES_STC.values:
        sts = demandio.STC_to_STs(stc)
        ecs = set(fs.flatten([demandio.ST_to_ECs(st) for st in sts]))
        if energy_carrier in ecs:
            yield ds, es, energy_carrier


def aggregate_demand(demand, aggcols):
    """
    Aggregates non residual demand to coarser level for
    corresponding residual demand.
    """
    # can be optimized if needed, aggregate only if required.
    # return demand.reset_index().groupby(aggcols)['EnergyDemand'].sum()
    return ioutils.groupby(demand, aggcols, 'EnergyDemand')


@demand_profile
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
    try:
        s = sum_series([aggregate_demand(compute_demand(*item),
                                         indexcols)
                        for item in non_bottomup+list(bottomup) if (last := item)])
    except KeyError as k:
        logger.error(
            f"Demand for {last} is coarser than ResidualDemand for {demand_sector}")
        raise demandio.DemandValidationError(
            f"ResidualDemand computation failed because at least one non-residual demand in {demand_sector} is having coarser granuarity than residual demand")
    except Exception as e:
        logger.exception(e)
        logger.error(f"Last item {last}")
        raise e

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


def finest_balancing_columns(ds_es_ec):
    g = [ioutils.get_geographic_columns(demandio.balancing_area(ec))
         for ds, es, ec in ds_es_ec]
    t = [ioutils.get_time_columns(demandio.balancing_time(ec))
         for ds, es, ec in ds_es_ec]

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
                                          cols,
                                          True)


def compute_demand_with_coarseness(demand_sector,
                                   energy_service,
                                   energy_carrier,
                                   columns,
                                   flag=False):
    """aggregates the demand over the granularity given by columns if passed columns are of coarser granuarity.
    if passed columns are of finer granularity than demand , then demand is returned just by aggrigating over
    the existing granularity of demand
    """
    demand = compute_demand(demand_sector, energy_service, energy_carrier)
    if flag:
        # only for saving in balacing time and area granularities
        G = [c for c in columns if c in constant.GEOGRAPHIES]
        T = [c for c in columns if c in constant.TIME_SLICES]
        g = ioutils.get_geographic_columns_from_dataframe(demand)
        t = ioutils.get_time_columns_from_dataframe(demand)

        if len(g) < len(G):
            message = f"""Failed to aggregate demand for {demand_sector},{energy_service},{energy_carrier} on {G}
Instead aggregating on {g}"""
            logger.debug(message)
            G = g
        if len(t) < len(T):
            message = f"""Failed to aggregate demand for {demand_sector},{energy_service},{energy_carrier} on {T}
Instead aggregating on {t}"""
            logger.debug(message)
            T = t

        columns = T + G

    return ioutils.groupby(demand, columns, 'EnergyDemand')


@functools.lru_cache()
def get_data_demand_with_servicetechs(demand_sector,
                                      energy_service,
                                      energy_carrier):
    _ = compute_demand(demand_sector, energy_service, energy_carrier)
    # this will make sure that demand is computed before hand
    service_techs_files = find_service_tech_demand_filepath(demand_sector,
                                                            energy_service,
                                                            energy_carrier)

    d = []
    for st in service_techs_files:
        df = pd.read_csv(service_techs_files[st])
        df = utilities.seasonwise_timeslices(df, 'EnergyDemand')
        df['ServiceTech'] = st
        df['ServiceTechCategory'] = df.ServiceTech.apply(
            demandio.ST_to_STC)
        d.append(ioutils.order_rows(df))

    return d


def write_demand_with_service_techs(demand_sector,
                                    energy_service,
                                    energy_carrier,
                                    csvwriter: csv.DictWriter):
    d = get_data_demand_with_servicetechs(demand_sector,
                                          energy_service,
                                          energy_carrier)

    for data in d:
        data['EnergyService'] = energy_service
        csvwriter.writerows(data.to_dict(orient="records"))


def get_demand_with_service_techs(demand_sector,
                                  energy_service,
                                  energy_carrier):
    """
    -----------
    demand_sector: str
    energy_service: str
    energy_carrier: str

    Returns
    -------
       a dataframe
    """
    d = get_data_demand_with_servicetechs(demand_sector,
                                          energy_service,
                                          energy_carrier)
    if d:
        return pd.concat(d)
    else:
        return pd.DataFrame()


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


def pick_coarsest_finest(ds_es_ec,
                         take_cons_cols=True,
                         coarsest=True,
                         data_loader=compute_demand):
    """get coarsest/finest granularity-columns among given demand outputs
    """

    allcols = []
    for ds, es, ec in ds_es_ec:
        # There is assumption that demand has been computed already and saved to
        # file . This compute_demand call will just read that file
        d = data_loader(ds, es, ec)
        allcols.append(set(ioutils.get_all_structure_columns(d)))

    if take_cons_cols:
        start = allcols[0]
    else:
        start = set(constant.TIME_SLICES + constant.GEOGRAPHIES)

    if coarsest:
        def func(x, y): return x & y
    else:
        def func(x, y): return x | y

    return ioutils.order_columns(functools.reduce(func,
                                                  allcols,
                                                  start))


def get_finest(ds_es_ec, take_cons_cols=True, data_loader=compute_demand):
    """get finest granularity-columns among given demand outputs
    """
    return pick_coarsest_finest(ds_es_ec,
                                take_cons_cols=take_cons_cols,
                                coarsest=False,
                                data_loader=data_loader)


def get_coarsest(ds_es_ec, take_cons_cols=True, data_loader=compute_demand):
    """get coarsest granularity-columns among given demand outputs
    """
    return pick_coarsest_finest(ds_es_ec,
                                take_cons_cols=take_cons_cols,
                                coarsest=True,
                                data_loader=data_loader)


def save_coarsest(entity_names=('EnergyService', 'EnergyCarrier')):
    """Aggregates Demand for EnergyService,EnergyCarrier over all DemandSectors
    """
    ds_es_ec_columns = ['DemandSector', 'EnergyService', 'EnergyCarrier']
    granularity_Map = demandio.get_combined_granularity_map()
    entities = granularity_Map.set_index(
        entity_names).index

    for items in entities.unique():
        if isinstance(items, str):
            items = (items,)

        logging.debug(
            f"Writing results for " + ",".join(items) + " across all demand sectors")
        filepath = get_output_filepath(entity_names, items)
        if os.path.exists(filepath):
            logger.debug(f"{filepath} exists, skipping")
            continue

        q = " & ".join([f"{name} == '{item}'" for name,
                        item in zip(entity_names, items)])
        ds_es_ec_map = granularity_Map.query(q)

        ds_es_ec = ds_es_ec_map.set_index(ds_es_ec_columns).index
        # cols = coarsest(ds_es_ec_map, "DemandSector" in entity_names)
        cols = get_coarsest(ds_es_ec)
        target_cols = ["EnergyDemand",
                       'SeasonEnergyDemand'] if 'Season' in cols else ['EnergyDemand']

        with open(filepath, "w", newline='') as f:
            csvf = csv.DictWriter(f, dialect='excel',
                                  fieldnames=cols + target_cols)
            csvf.writeheader()
            d = []
            for ds, es, ec in ds_es_ec:
                d_ = compute_demand_with_coarseness(ds, es, ec, cols)
                d.append(d_)
                # print(d_)

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
    DS_ES_EC_DemandGranularityMap = demandio.get_combined_granularity_map()
    entities = DS_ES_EC_DemandGranularityMap.set_index(
        entity_names).index
    print(f"Writing results with ServiceTech and EnergyService")

    for items in entities.unique():
        if isinstance(items, str):
            items = (items,)

        q = " & ".join([f"{name} == '{item}'" for name,
                        item in zip(entity_names, items)])
        ds_es_ec_map = DS_ES_EC_DemandGranularityMap.query(q)

        ds_es_ec = ds_es_ec_map.set_index(ds_es_ec_columns).index
        if not check_energy_service_service_tech_count(
                ds_es_ec_map.set_index(ds_es_ec_columns),
                ds_es_ec):
            continue

        cols = get_finest(ds_es_ec,
                          take_cons_cols="DemandSector" in entity_names,
                          data_loader=get_demand_with_service_techs)

        logging.debug(
            f"Writing results for " + ",".join(items) +
            " across all energy services with ServiceTech and EnergyService")
        filepath = get_output_filepath(entity_names, items, postfix="ES_ST")

        if os.path.exists(filepath):
            logger.debug(f"{filepath} exists, skipping")
            continue

        target_cols = ['EnergyService',
                       'ServiceTech',
                       'ServiceTechCategory',
                       "EnergyDemand"]
        if 'Season' in cols:
            target_cols += ['SeasonEnergyDemand']

        with open(filepath, "w", newline='') as f:
            csvf = csv.DictWriter(f, dialect='excel',
                                  fieldnames=cols + target_cols)
            csvf.writeheader()
            for ds, es, ec in ds_es_ec:
                write_demand_with_service_techs(ds, es, ec, csvf)


def complete_run():
    """runs validation and complete demand processing flow.
    """
    validation_flag = config.get_config_value("validation") == 'True'
    create_output_dirs()
    if (validation_flag and validate()) or not validation_flag:
        compute_demand_all()
        path = save_supply_input()
        save_coarsest(entity_names=['DemandSector', 'EnergyCarrier'])
        save_coarsest(entity_names=['EnergyCarrier'])
        save_coarsest(entity_names=['EnergyService', 'EnergyCarrier'])
        save_coarsest_with_ST()
        copy_supply_parameter(path)
        save_nonenergy_emissions()
        save_end_use_emissions()


def validate():
    """Validate Common and Demand parameters
    returns True or False
    """
    valid = loaders.validate_params("Common")
    if not valid:
        print("Validation failed for Common")
        return False

    valid = loaders.validate_params("Demand")
    if not valid:
        print("Validation failed for Demand")
        return False
    return valid


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


def save_nonenergy_emissions():
    DS_ES_Map = demandio.get_combined("DS_ES_Map")
    ds_es = DS_ES_Map.set_index(
        ['DemandSector', 'EnergyService']).index.unique()
    bottomups = [(ds, es)
                 for ds, es in ds_es if demandio.get_type(ds, es) == "BOTTOMUP"]
    procemissions = []
    for ds, es in bottomups:
        b = NonEnergyEmissions(ds, es)
        emissions = b.compute_quantity()
        if isinstance(emissions, pd.DataFrame):
            procemissions.append(emissions)
    if procemissions:
        emissions = pd.concat(procemissions)
        logger.info("Writing NonEnergyEmissions")
        output = get_output_filepath(
            None, None, output_type='NonEnergyEmissions')
        print("Writing NonEnergyEmissions")
        emissions.to_csv(output, index=False)


def save_end_use_emissions():
    if config.get_config_value("output"):
        config.set_config("demand_output",
                          config.get_config_value("output"))
    emissions = emission.emissions()
    if emissions is not None:
        print("Writing EndUseEmissions-DemandMetDomesticEC")
        logger.info("Writing EndUseEmissions-DemandMetDomesticEC")
        output = get_output_filepath(
            None, None, output_type='EndUseEmissions-DemandMetDomesticEC')
        emissions.to_csv(output, index=False)


def compute_demand_all():
    """Computes demand for all combinations of demand_sector, energy_service
    and energy_carrier
    """

    ds_es_ec = demandio.get_all_ds_es_ec()

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
    ds_es_ec = demandio.get_all_ds_es_ec()

    if demand_sector not in {ds for ds, es, ec in ds_es_ec}:
        print(f"Invalid value for demand_sector parameter, {demand_sector}")
        valid = False
    if energy_service not in {es for ds, es, ec in ds_es_ec}:
        print(f"Invalid value for energy_service parameter, {energy_service}")
        valid = False
    if energy_carrier not in {ec for ds, es, ec in ds_es_ec}:
        print(f"Invalid value for energy_carrier parameter, {energy_carrier}")
        valid = False
    if (demand_sector, energy_service, energy_carrier) not in ds_es_ec:
        print("Invalid combination of demand_sector, energy_service, energy_carrier")
        valid = False

    return valid


def make_groups_and_sum(ds_es_EC):
    demands = [compute_demand(ds, es, EC) for ds, es, EC in ds_es_EC]
    groups = {}

    for ds, es, EC in ds_es_EC:
        d = compute_demand_balancing(ds, es, EC)
        cols = tuple(ioutils.get_all_structure_columns(d.reset_index()))
        groups.setdefault(cols, []).append(d)

    datasets = []
    for granuarity, dfs in groups.items():
        df = sum_series(dfs).reset_index()
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

        datasets.append(df)

    return datasets


def save_supply_input():
    """saves demand for every EnergyCarrier in EndUseDemandEnergy.csv
    """
    ds_es_ec = demandio.get_all_ds_es_ec()
    path_all = os.path.join(
        filemanager.get_output_path("Demand"), "EndUseDemandEnergy.csv")

    fine_cols = ['EnergyCarrier'] + \
        finest_balancing_columns(ds_es_ec) + \
        ['EndUseDemandEnergy', 'TotalEnergyDemand']

    print("Writing results for supply input")
    logger.info("Writing results for supply input")
    with open(path_all, "w", newline='') as f:
        csvf = csv.DictWriter(f, dialect='excel', fieldnames=fine_cols)
        csvf.writeheader()
        for EC in fs.unique_list(ec for _, _, ec in ds_es_ec):
            ds_es_EC = [(ds, es, ec)
                        for ds, es, ec in ds_es_ec if ec == EC]

            dfs = make_groups_and_sum(ds_es_EC)
            for df in dfs:
                csvf.writerows(df.to_dict(orient='records'))

    return path_all


def copy_supply_parameter(srcpath):
    """Copies EndUseDemandEnergy to supply paramters data
    """
    supplypath = filemanager.find_filepath('EndUseDemandEnergy')
    scenario = config.get_config_value("scenario")
    scenario_ = os.path.join("Scenarios", scenario)
    path = supplypath.replace("Default Data", scenario_)
    if os.path.exists(path):
        warning = f"Results of previous run are present at {path}. Results will be overwritten, press n to cancel, enter to continue (y/n)"
        check = input(warning)
        if check and check.strip() and check.strip().lower()[0] == "n":
            return
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copy(srcpath, path)


def get_csvs(root):
    def csvs():
        for base, dirs, files in os.walk(root):
            yield from [os.path.join(base, f) for f in files if f.endswith(".csv")]

    return list(csvs())


def create_output_dirs():
    """This should be called from main process in order to avoid directory
    creation from different processes
    """
    ds_es = demandio.get_combined("DS_ES_Map")
    ds_es = ds_es.set_index(['DemandSector', 'EnergyService'])

    base = Path(filemanager.get_output_path("Demand"))
    for ds, es in ds_es.index.unique():
        p1 = create_base(base, "DemandSector", ds, es)
        p2 = create_base(base, "EnergyService", es)
    p3 = create_base(base, "EnergyCarrier")
    p4 = create_base(base, "TotalNumInstances")


def clean_output():
    """Deletes files from output directory.
    """
    outputpath = filemanager.get_output_path("Demand")
    csv_files = get_csvs(outputpath)
    if csv_files:
        warning = f"Results of previous run are present at {outputpath}. Results will be overwritten, press n to cancel, enter to continue (y/n)"
        check = input(warning)
        if check and check.strip() and check.strip().lower()[0] == "n":
            sys.exit(1)
        for p in csv_files:
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
    if not Path(filemanager.scenario_path()).is_dir():
        print(f"Scenario {scenario} does not exist.")
        sys.exit(1)

    if output:
        config.set_config("output", output)
    config.set_config("numthreads", str(numthreads))
    config.set_config("validation", str(validation))

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
        elif demand_sector and not any([energy_carrier, energy_service]):
            clean_output()
            demandio.add_demand_sector_filters(demand_sector)
            complete_run()
        elif any([energy_service, energy_carrier]):  # any but not all!
            print(
                "Partial inputs given for energy_service and energy_carrier")
            print(
                "energy_service and energy_carrier , either all should be given or none.")
            return
        else:  # none from ds,es,ec is given
            clean_output()
            complete_run()
    except Exception as e:
        logger.exception(e)
        print("Demand processing failed.")
    finally:
        while not get_queue().empty():
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
              help="Level for logging: one of  DEBUG, INFO, WARN or ERROR (default: INFO)",
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

    if only demand_sector is provided but not energy_service and energy_carrier
    then complete run performed only for given demand_sector
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
