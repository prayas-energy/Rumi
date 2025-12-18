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
"""This is postprocessing module to support computation of emissions.
"""
from rumi.io import system
from rumi.io import demand as demandio
from rumi.io import loaders
from rumi.io import config
from rumi.io import filemanager
from rumi.io import utilities
from rumi.io import functionstore as fs
from rumi.processing import utilities as putils
import operator as op

import logging
import functools
import pandas as pd
import os

logger = logging.getLogger(__name__)


def emission_types_exist():
    """checks if emissions computation is doable or not.
    Some minimum set of parameters are required to do emissions postprocessing.
    if those parameters are absent this will return False.
    """
    EmissionTypes = loaders.get_parameter("EmissionTypes")

    if not isinstance(EmissionTypes, pd.DataFrame):
        return False
    return True


def get_demand_output_path():
    demand_output = config.get_config_value("demand_output")
    if demand_output:
        path = filemanager.get_custom_output_path("Demand", demand_output)
    else:
        scenario_path = filemanager.scenario_location()
        path = os.path.join(scenario_path, "Demand", 'Output')

    return path


def demand_filepath(ds, es, ec, st=None):
    path = get_demand_output_path()
    path = os.path.join(path, "DemandSector", ds, es)
    if st:
        filename = f"{ds}_{es}_{st}_{ec}_Demand.csv"
    else:
        filename = f"{ds}_{es}_{ec}_Demand.csv"
    return os.path.join(path, filename)


def aggregate_enduse(f):

    def process_levels(data):
        ds = []
        cols = []
        for l in data.colindicator.unique():
            d = utilities.filter_empty(data.query(f"colindicator == '{l}'"))
            ds.append(d)
            c = utilities.get_all_structure_columns(d)
            cols.append(set(c))

        coarsest = list(functools.reduce(op.and_, cols[1:], cols[0]))
        total = 0
        for d in ds:
            g = utilities.groupby(d, coarsest, "EndUseDemandEnergy")
            total = total + g

        return total.reset_index()

    def wrapper(*args):
        e = f(*args)
        ecdata = []
        for ec in e.EnergyCarrier.unique():
            d = e.query(f"EnergyCarrier == '{ec}'").copy()
            colindicator = d.T.isnull().apply(lambda x: x.apply(lambda y: str(int(y)))).sum()
            d['colindicator'] = colindicator
            d = process_levels(d)
            d['EnergyCarrier'] = ec
            ecdata.append(d)

        return pd.concat(ecdata)

    return wrapper


@functools.lru_cache()
@aggregate_enduse
def get_end_use_demand_energy():
    """Reads demand output EndUseDemandEnergy.csv if exists, else
    returns this data from Supply input parameters
    """
    try:
        return read_demand_output("EndUseDemandEnergy")
    except Exception as e:
        logger.warning(
            "Demand procesing output EndUseDemandEnergy.csv not found")
        logger.warning(
            "EndUseDemandEnergy.csv from Supply parameters will be used")

        # not using loaders.get_parameter because it returns dataframe
        # with na values as "". That creates problem in merge!
        # fix it later after removing na_values = "" in loaders.read_csv
        filepath = filemanager.find_filepath("EndUseDemandEnergy")
        return pd.read_csv(filepath)


def extract_demand(energy_carrier):
    """get demand for given EvergyCarrier from EndUseDemandEnergy.csv
    """
    end_use_demand = get_end_use_demand_energy()  # this is cached , so read only once
    ed = end_use_demand.query(f"EnergyCarrier == '{energy_carrier}'")
    ed = utilities.filter_empty(ed)
    season_wise = putils.seasonwise_timeslices(ed, "EndUseDemandEnergy")

    if 'SeasonEndUseDemandEnergy' in season_wise.columns:
        season_wise = season_wise.rename(
            columns={"SeasonEndUseDemandEnergy": "SeasonEnergyDemand",
                     "EndUseDemandEnergy": "EnergyDemand"})
    else:
        season_wise = season_wise.rename(
            columns={"EndUseDemandEnergy": "EnergyDemand"})

    return season_wise


@functools.lru_cache(maxsize=None)
def read_demand(*args):
    """reads demand file for ds, es, ec, st combination
    """
    ds, es, ec, st = args
    if no_demand_output_exists():
        logger.debug(f"Reading Demand for {args} from EndUseDemandEnergy")
        return extract_demand(ec)
    filepath = demand_filepath(*args)

    logger.debug(f"Reading Demand for {args} from  {filepath}")
    demand = pd.read_csv(filepath)
    demand['EnergyCarrier'] = ec
    demand['DemandSector'] = ds
    demand['EnergyService'] = es
    if st:
        demand['ServiceTech'] = st
    return demand


def get_physical_bottomup_ds_es_ec_st():
    ds_es_ec = filter_physical(get_ds_es_ec(bottomup=True))
    ds_es_ec_st = [(ds, es, ec, st) for ds, es, ec in ds_es_ec
                   for st in demandio.get_service_techs(ds, es, ec)]

    return ds_es_ec_st


def get_physical_nonbottomup_ds_es_ec_st():
    """Returns nonbottomup tuples of ds,es,ec,st for physical carriers.
    st is always None
    """
    ds_es_ec = filter_physical(get_ds_es_ec(bottomup=False))
    return [(*item, None) for item in ds_es_ec]


def get_ds_es_ec(bottomup=True):

    m = demandio
    ds_es_func = m.get_bottomup_ds_es if bottomup else m.get_nonbottomup_ds_es

    ds_es_ec = demandio.get_all_ds_es_ec_()
    ds_es = [(ds, es) for ds, es in ds_es_func()]

    return [(ds, es, ec) for ds, es, ec in ds_es_ec
            if (ds, es) in ds_es]


def filter_physical(ds_es_ec):
    """Consider only Physical Carriers
    """
    primary = loaders.get_parameter('PhysicalPrimaryCarriers')
    derived = loaders.get_parameter('PhysicalDerivedCarriers')

    return [(ds, es, ec) for ds, es, ec in ds_es_ec
            if ec in primary.EnergyCarrier.values or
            ec in derived.EnergyCarrier.values]


def demand_filepath(ds, es, ec, st=None):
    path = get_demand_output_path()
    path = os.path.join(path, "DemandSector", ds, es)
    if st:
        filename = f"{ds}_{es}_{st}_{ec}_Demand.csv"
    else:
        filename = f"{ds}_{es}_{ec}_Demand.csv"
    return os.path.join(path, filename)


@functools.lru_cache()
def demand_output_status():
    """Returns a dictionary of DS,ES,EC,ST(optional) vs if corresponding
    output exists
    """
    def exists(ds, es, ec, st=None):
        path = demand_filepath(ds, es, ec, st)
        return os.path.exists(path)

    nonbottomup = get_physical_nonbottomup_ds_es_ec_st()
    bottomup = get_physical_bottomup_ds_es_ec_st()
    d = {item: exists(*item) for item in bottomup}
    d.update({item: exists(*item) for item in nonbottomup})
    return d


def all_demand_outputs_exist():
    """returns True if all the demand outputs required for emission computation
    exist.
    """
    return all(demand_output_status().values())


def no_demand_output_exists():
    """Returns True if demand outputs are absent alltogether
    """
    return not any(demand_output_status().values())


def check_enduse_common():
    """checks if one of ST_EmissionDetails or PhysicalCarrierEmissions exists
    and also if the supply outputs required to calculate EndUseEmissions exist
    """
    PhysicalCarrierEmissions = loaders.get_parameter(
        "PhysicalCarrierEmissions")
    valid = isinstance(PhysicalCarrierEmissions, pd.DataFrame)

    if not valid:
        ds_es_ec_st = get_physical_bottomup_ds_es_ec_st()
        ds_bottomup_set = {tup[0] for tup in ds_es_ec_st}

        for ds in ds_bottomup_set:
            ST_EmissionDetails = loaders.get_parameter("ST_EmissionDetails",
                                                       demand_sector=ds)
            if isinstance(ST_EmissionDetails, pd.DataFrame):
                valid = True
                break

    if not valid:
        logger.warning(
            "At least one of ST_EmissionDetails or PhysicalCarrierEmissions should be present, for EndUseEmissions computation")
        logger.warning(
            "Skipping EndUseEmissions computation as neither ST_EmissionDetails nor PhysicalCarrierEmissions is present")
        print(
            "Skipping EndUseEmissions computation as neither ST_EmissionDetails nor PhysicalCarrierEmissions is present")

        return False

    return valid


def compute_fractions(end_use,
                      end_use_met_dom,
                      end_use_met_imp,
                      physical_carriers):
    """Computes MetDemandEnergyFraction and DemandMetByDomEnergyFraction
    """
    logger.debug(
        "Computing MetDemandEnergyFraction and DemandMetByDomEnergyFraction")
    cols = utilities.get_all_structure_columns(
        end_use_met_dom) + ['EnergyCarrier']
    df = end_use_met_dom.merge(end_use_met_imp)
    df = df.merge(end_use)
    df = df.merge(physical_carriers)

    A = df['EndUseDemandMetByDom'] * \
        df['DomEnergyDensity']
    B = df['EndUseDemandMetByImp'] * \
        df['ImpEnergyDensity']
    MetDemandEnergyFraction = (A+B)/df['EndUseDemandEnergy']
    DemandMetByDomEnergyFraction = A/(A+B)

    df['MetDemandEnergyFraction'] = MetDemandEnergyFraction.fillna(
        0).rename("MetDemandEnergyFraction")
    df['DemandMetByDomEnergyFraction'] = DemandMetByDomEnergyFraction.fillna(
        0).rename("DemandMetByDomEnergyFraction")
    return df


def combine_physical_carriers():
    PhysicalPrimaryCarriers = loaders.get_parameter(
        "PhysicalPrimaryCarriersEnergyDensity")
    PhysicalDerivedCarriers = loaders.get_parameter(
        "PhysicalDerivedCarriersEnergyDensity")
    # PDC = PhysicalDerivedCarriers.rename(
    # columns = {"EnergyDensity": "DomEnergyDensity"})
    # PDC['ImpEnergyDensity'] = PDC['DomEnergyDensity']
    return pd.concat([PhysicalDerivedCarriers, PhysicalPrimaryCarriers])


def handle_day_no(data):
    """ If the DayNo column is present and is non empty for an EnergyCarrier
    then for each of these two supply outputs, we need to take the average
    value of the output values over all the DayNos that occur for each
    combination of G*, Year, Season, DayType and DaySlice (if applicable)
    for that EnergyCarrier while calculating MetDemandEnergyFraction.
    """
    if "DayNo" not in data.columns:
        return data
    dfs = []
    for ec in data.EnergyCarrier.unique():
        subset = utilities.filter_empty(data[data.EnergyCarrier == ec])
        if "DayNo" in subset.columns:
            cols = utilities.get_all_structure_columns(subset)
            cols.remove('DayNo')
            subset = subset.groupby(cols, sort=False, as_index=False).mean()
        dfs.append(subset)
    return pd.concat(dfs)


def energy_demand_met_(Demand,
                       fractions):
    """computes EnergyDemandMet Dom and Imp for a specific end-use Demand
    """

    demand = demand_col(Demand)
    df = fractions.merge(Demand)
    df["EnergyDemandMetImp"] = df[demand] *\
        df['MetDemandEnergyFraction'] *\
        df['DemandMetByImpEnergyFraction'] / df["ImpEnergyDensity"]

    df["EnergyDemandMetDom"] = df[demand] *\
        df['MetDemandEnergyFraction'] *\
        df['DemandMetByDomEnergyFraction'] / df["DomEnergyDensity"]

    return df


def compute_energy_demand_met(fractions, ds_es_ec_st):
    """Computes EnergyDemandMetDom/Imp for given set of ds,es,ec,st.
    st will be None for nonbottomup type of input types
    """

    demand_met = []
    filter_empty = utilities.filter_empty

    for ds, es, ec, st in ds_es_ec_st:
        query = f"EnergyCarrier == '{ec}'"
        fractions_ = filter_empty(fractions.query(query))

        Demand = read_demand(ds, es, ec, st)
        # read_demand also adds EnergyCarrier/ServiceTech column in Demand
        d = energy_demand_met_(Demand,
                               fractions_)
        demand_met.append(d)

    EnergyDemandMet = pd.concat(demand_met)

    return EnergyDemandMet


class EmissionDemandOnly:

    def __init__(self):
        self.load_data()
        self.fractions = self.compute_fractions()

    def initilize_met_demand_params(self):
        # becasue supply outputs are not available
        # we will have to do some assumptions
        df = self.EndUseDemandEnergy.copy()
        df = df.merge(self.physical_carriers)
        df['EndUseDemandMetByDom'] = df.EndUseDemandEnergy / df.DomEnergyDensity
        del df['EndUseDemandEnergy']
        del df['DomEnergyDensity']
        del df['ImpEnergyDensity']
        self.EndUseDemandMetByDom = df.copy()
        self.EndUseDemandMetByImp = self.EndUseDemandMetByDom.copy()
        self.EndUseDemandMetByImp.rename(
            columns={"EndUseDemandMetByDom": "EndUseDemandMetByImp"}, inplace=True)
        self.EndUseDemandMetByImp['EndUseDemandMetByImp'] = 0

    def load_data(self):
        self.EndUseDemandEnergy = get_end_use_demand_energy()
        self.physical_carriers = combine_physical_carriers()
        self.initilize_met_demand_params()

    def compute(self):
        return compute_emissions(self.fractions)

    def compute_fractions(self):
        """Computes MetDemandEnergyFraction and DemandMetByDomEnergyFraction
        """
        logger.debug(
            "Computing MetDemandEnergyFraction and DemandMetByDomEnergyFraction")

        df = self.EndUseDemandMetByDom.merge(
            self.EndUseDemandMetByImp)
        df = df.merge(self.EndUseDemandEnergy)
        df = df.merge(self.physical_carriers)

        A = df['EndUseDemandMetByDom'] * \
            df['DomEnergyDensity']
        B = df['EndUseDemandMetByImp'] * \
            df['ImpEnergyDensity']
        MetDemandEnergyFraction = (A+B)/df['EndUseDemandEnergy']
        DemandMetByDomEnergyFraction = A/(A+B)

        df['MetDemandEnergyFraction'] = MetDemandEnergyFraction.fillna(
            0).rename("MetDemandEnergyFraction")
        df['DemandMetByDomEnergyFraction'] = DemandMetByDomEnergyFraction.fillna(
            0).rename("DemandMetByDomEnergyFraction")
        df['DemandMetByImpEnergyFraction'] = 1 - \
            df['DemandMetByDomEnergyFraction']
        return df


@functools.lru_cache()
def load_logger(name):
    global logger
    logger = logging.getLogger(name)


def compute_emissions(fractions):
    """Computes emission values for end use demand.
    """
    if no_demand_output_exists():
        print("No Demand output exists")
        ds_es_ec_st = [(None, None, ec, None)
                       for ec in fractions["EnergyCarrier"].unique()]

        if len(ds_es_ec_st) > 0:
            emission_data = loaders.get_parameter("PhysicalCarrierEmissions")
            if isinstance(emission_data, pd.DataFrame):
                EnergyDemandMet_ec = compute_energy_demand_met(fractions,
                                                               ds_es_ec_st)
                Emissions = compute_end_use_emission(EnergyDemandMet_ec,
                                                     emission_data)
            else:
                Emissions = pd.DataFrame()
        else:
            Emissions = pd.DataFrame()
    else:
        ds_es_ec_st = get_physical_nonbottomup_ds_es_ec_st()

        if len(ds_es_ec_st) > 0:
            emission_data = loaders.get_parameter("PhysicalCarrierEmissions")
            if isinstance(emission_data, pd.DataFrame):
                EnergyDemandMet_nb = compute_energy_demand_met(fractions,
                                                               ds_es_ec_st)
                Emission_nb = compute_end_use_emission(EnergyDemandMet_nb,
                                                       emission_data)
            else:
                Emission_nb = pd.DataFrame()
        else:
            Emission_nb = pd.DataFrame()

        ds_es_ec_st = get_physical_bottomup_ds_es_ec_st()

        if len(ds_es_ec_st) > 0:
            ds_bottomup_set = tuple({tup[0] for tup in ds_es_ec_st})
            emission_data = combine_st_emission_data(ds_bottomup_set)
            EnergyDemandMet_b = compute_energy_demand_met(fractions,
                                                          ds_es_ec_st)
            Emission_b = compute_end_use_emission(EnergyDemandMet_b,
                                                  emission_data)
        else:
            Emission_b = pd.DataFrame()

        Emissions = pd.concat([Emission_nb, Emission_b])

    indexcols = get_enduse_emission_columns(Emissions)
    Emissions.set_index(indexcols, inplace=True)
    return Emissions['Emission'].reset_index()


def get_enduse_emission_columns(Emission):
    cols = utilities.get_all_structure_columns(Emission)
    indexcols = ['EnergyCarrier', 'EmissionType']
    for item in ['DemandSector', 'EnergyService', 'ServiceTech']:
        if item in Emission.columns:
            indexcols.append(item)

    return indexcols + cols


def get_EC_ST_ES_data_(ds):
    DS_ES_Map = loaders.get_parameter("DS_ES_Map", demand_sector=ds)
    STC_ES_Map = loaders.get_parameter("STC_ES_Map")
    ES_ = DS_ES_Map.query(f"DemandSector == '{ds}'").EnergyService.values
    data = []
    for ES in ES_:
        for STC in [row[0] for row in STC_ES_Map if ES in row[1:]]:
            for ST in demandio.STC_to_STs(STC):
                for EC in demandio.ST_to_ECs(ST):
                    data.append({"EnergyCarrier": EC,
                                 "ServiceTech": ST,
                                 "EnergyService": ES})

    return pd.DataFrame(data)


def get_EC_ST_ES_data():
    d = []
    for ds in loaders.get_parameter("DS_List"):
        d.append(get_EC_ST_ES_data_(ds))
    return pd.concat(d)


@functools.lru_cache()
def combine_st_emission_data(ds_bottomup):
    """For each DS, combines ST_EmissionDetails and PhysicalCarrierEmissions
    with the help of ST->EnergyCarrier mapping to get final emission data
    for that demand sector.
    Finally, the thus combined emission data for each DS is concatenated
    together over all demand sectors and returned.
    """
    PhysicalCarrierEmissions = loaders.get_parameter(
        "PhysicalCarrierEmissions")

    ST_EC_ES_data = get_EC_ST_ES_data()

    if isinstance(PhysicalCarrierEmissions, pd.DataFrame):
        x = PhysicalCarrierEmissions.merge(ST_EC_ES_data, on="EnergyCarrier")
    else:
        x = pd.DataFrame()

    emission_data_df_list = []
    for ds in ds_bottomup:
        ST_EmissionDetails = loaders.get_parameter("ST_EmissionDetails",
                                                   demand_sector=ds)
        if isinstance(ST_EmissionDetails, pd.DataFrame):
            ST_EmissionDetails = ST_EmissionDetails[~ST_EmissionDetails.EnergyCarrier.isnull(
            )]
            y = ST_EmissionDetails.copy()
            if not x.empty:
                z = fs.override_dataframe(x, y, ['EnergyCarrier',
                                                 'ServiceTech',
                                                 'EmissionType',
                                                 'Year'])
            else:
                z = y.copy()
        else:
            z = x.copy()

        if not z.empty:
            z.insert(0, 'DemandSector', ds)
            emission_data_df_list.append(z)

    return pd.concat(emission_data_df_list, ignore_index=True)


def compute_end_use_emission(EnergyDemandMet: pd.DataFrame,
                             emission_data: pd.DataFrame) -> pd.DataFrame:
    """compute end use emissions from Dom and Imp sources and also their sum
    """
    df = EnergyDemandMet.merge(emission_data)
    df['EmissionImp'] = df["EnergyDemandMetImp"] * \
        df["ImpEmissionFactor"]
    df['EmissionDom'] = df["EnergyDemandMetDom"] * \
        df["DomEmissionFactor"]

    df['Emission'] = df['EmissionImp'] + df['EmissionDom']
    return df


def read_demand_output(output_name):
    path = get_demand_output_path()
    logger.debug(f"Reading demand output {output_name} from  {path}")
    return pd.read_csv(os.path.join(path, ".".join([output_name, "csv"])))


def demand_col(data):
    """returns names of demand column that should be used as demand
    """
    if "SeasonEnergyDemand" in data.columns:
        return "SeasonEnergyDemand"
    else:
        return "EnergyDemand"


def emissions():
    """Emission computation api function
    """
    try:
        if not emission_types_exist():
            logger.warning(
                "EmissionTypes parameter is absent, hence emissions can not be computed")
            print(
                "EmissionTypes parameter is absent, hence emissions can not be computed")
            return

        if check_enduse_common():
            if all_demand_outputs_exist():
                # we don't want to check for sypply parameter in this emission processing
                e = EmissionDemandOnly()
                return e.compute()
            else:
                print(
                    "Some of the required demand outputs are absent, hence EndUseEmissions will not be computed")
                logger.error(
                    "Some of the required demand outputs are absent, hence EndUseEmissions will not be computed")
    except Exception as e:
        logger.exception(e)
        raise e
