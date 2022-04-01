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
"""This is postprocessing run after demand and supply processing is run.
Currently computation of emissions is supported.
"""
import time
import os
import logging
import functools
import click
import pandas as pd
import numpy as np
from rumi.io import loaders
from rumi.io.logger import init_logger, get_event
from rumi.io import config
from rumi.io import filemanager
from rumi.io import utilities
from rumi.io import supply
from rumi.io import demand as demandio
from rumi.io import functionstore as fs
from rumi.processing import utilities as putils

logger = logging.getLogger(__name__)


def check_exists(filename, type_):
    if type_ == "Supply":
        folderpath = get_supply_output_path()
    else:
        folderpath = filemanager.get_output_path("Demand")

    path = os.path.join(folderpath, filename)
    return os.path.exists(path)


def check_ect():
    """checks if emission inputs and supply outputs required for
    ECT emissions computation exist.
    """
    ECT_EmissionDetails = supply.get_filtered_parameter("ECT_EmissionDetails")
    PhysicalCarrierEmissions = loaders.get_parameter(
        "PhysicalCarrierEmissions")

    valid = isinstance(PhysicalCarrierEmissions, pd.DataFrame) or \
        isinstance(ECT_EmissionDetails, pd.DataFrame)
    if not valid:
        logger.warning(
            "At least one of ECT_EmissionDetails or PhysicalCarrierEmissions should be present, for ECTEmissions computation")
        logger.warning(
            "Skipping ECTEmissions computation as neither ECT_EmissionDetails nor PhysicalCarrierEmissions is present")
        print(
            "Skipping ECTEmissions computation as neither ECT_EmissionDetails nor PhysicalCarrierEmissions is present")

        return False

    files = ["ECTInputDom",
             "ECTInputImp"]
    files = [".".join([f, "csv"]) for f in files]
    folderpath = get_supply_output_path()
    paths = [os.path.join(folderpath, f) for f in files]
    exists = [os.path.exists(p) for p in paths]
    if not all(exists):
        print("Skipping ECTEmissions computation as the required supply outputs (ECTInputDom, ECTInputImp) are not present")
        logger.warning(
            "Skipping ECTEmissions computation as the required supply outputs (ECTInputDom, ECTInputImp) are not present")

    return all(exists)


def get_supply_output_path():
    supply_output = config.get_config_value("supply_output")
    if supply_output:
        path = filemanager.get_custom_output_path("Supply", supply_output)
    else:
        scenario_path = filemanager.scenario_location()
        path = os.path.join(scenario_path, "Supply", 'Output')

    return os.path.join(path, "Run-Outputs")


def get_demand_output_path():
    demand_output = config.get_config_value("demand_output")
    if demand_output:
        path = filemanager.get_custom_output_path("Demand", demand_output)
    else:
        scenario_path = filemanager.scenario_location()
        path = os.path.join(scenario_path, "Demand", 'Output')

    return path


def read_supply_output(output_name):
    path = get_supply_output_path()
    logger.debug(f"Reading supply output {output_name} from  {path}")
    return pd.read_csv(os.path.join(path, ".".join([output_name, "csv"])))


def read_demand_output(output_name):
    path = get_demand_output_path()
    logger.debug(f"Reading demand output {output_name} from  {path}")
    return pd.read_csv(os.path.join(path, ".".join([output_name, "csv"])))


def supply_compute_emission(emission_data):
    """computes supply emissions = EmissionDom + EmissionImp
    """
    def order_by(v):
        if v.name in cols:
            return utilities.order_by(v)
        elif v.name in orderdata:
            order = orderdata[v.name]
            return v.apply(order.index)
        else:
            return v

    ECTInputDom = read_supply_output("ECTInputDom")
    ECTInputImp = read_supply_output("ECTInputImp")

    dom = compute_ect_emission(ECTInputDom,
                               emission_data,
                               "Dom")
    imp = compute_ect_emission(ECTInputImp,
                               emission_data,
                               "Imp")

    total = dom.merge(imp, how="outer")

    total['Dom'].fillna(0, inplace=True)
    total['Imp'].fillna(0, inplace=True)

    total['Emission'] = total['Dom']+total['Imp']
    del total['Dom']
    del total['Imp']
    cols = utilities.get_all_structure_columns(total)

    ordercols = ['InstYear',
                 'EnergyConvTech',
                 'EmissionType']

    orderdata = {c: find_order(
        dom[c].unique(), imp[c].unique()) for c in ordercols}

    m = loaders.get_parameter("ModelPeriod").values[0]
    orderdata['InstYear'] = list(range(m[0]-1, m[1]+1))

    return total.sort_values(by=ordercols+cols,
                             key=order_by,
                             ignore_index=True)


def find_order(c1, c2):
    return [item for item in c1] +\
        [item for item in c2 if item not in c1]


def ect_emissions():
    """Computes and writes ECTEmissions based on supply processing outputs.
    """
    print("Computing ECTEmissions")
    logger.info("Computing ECTEmissions")
    PhysicalCarrierEmissions = loaders.get_parameter(
        "PhysicalCarrierEmissions")
    ECT_EmissionDetails = supply.get_filtered_parameter("ECT_EmissionDetails")
    EnergyConvTechnologies = loaders.get_parameter("EnergyConvTechnologies")

    emission_data = combine_emission_data(ECT_EmissionDetails,
                                          PhysicalCarrierEmissions,
                                          EnergyConvTechnologies)
    emissions = supply_compute_emission(emission_data)
    write_emissions_results(
        emissions[get_ectemissions_column_order(emissions)])


def write_emissions_results(data, filename="ECTEmissions.csv"):
    """Writes emissions results to the specified file
    """
    outputpath = filemanager.get_output_path("PostProcess")
    filepath = os.path.join(outputpath, filename)
    print(f"Writing results: {filename}")
    logger.info(f"Writing results: {filepath}")
    data.to_csv(filepath, index=False)


def valid_column(data, column):
    """checks if given 'column' is present and if it has nonnull values in it
    """
    if column in data.columns and data[column].isnull().sum() == 0:
        return True
    else:
        return False


def season_wise(ect_input, colname):
    """Multiplies values in the given column with number of days in season
    and weight for given DayType, if required
    """
    if "DayType" not in ect_input.columns:
        return ect_input

    d = utilities.seasons_size()
    d[np.NaN] = 1
    ect_input['NumDays'] = ect_input['Season'].map(d)

    DayTypes = loaders.get_parameter("DayTypes")
    DayTypes = DayTypes.append({"DayType": np.NaN, "Weight": 1},
                               ignore_index=True)

    ect_input = ect_input.merge(DayTypes)

    ect_input['multiplier'] = np.ones_like(ect_input[colname].values)

    if "DayNo" not in ect_input.columns:
        ect_input['multiplier'] = np.where(ect_input["DayType"].isna(),
                                           1,
                                           ect_input["NumDays"] * ect_input["Weight"])
    else:
        ect_input['multiplier'] = np.where(ect_input["DayType"].notna() &
                                           ect_input["DayNo"].isna(),
                                           ect_input["NumDays"] *
                                           ect_input["Weight"],
                                           1)
    ect_input[colname] = ect_input[colname] *\
        ect_input['multiplier']

    del ect_input['NumDays']
    del ect_input['Weight']
    return ect_input


def combine_emission_data(ECT_EmissionDetails,
                          PhysicalCarrierEmissions,
                          EnergyConvTechnologies):
    """Combines ECT_EmissionDetails and PhysicalCarrierEmissions with
    the help of EnergyConvTechnologies to get final emission data
    """
    ect = EnergyConvTechnologies.rename(columns={"InputEC": "EnergyCarrier"})

    if isinstance(PhysicalCarrierEmissions, pd.DataFrame):
        x = PhysicalCarrierEmissions.merge(ect, on="EnergyCarrier")

        ModelPeriod = loaders.get_parameter("ModelPeriod")
        InstYear = pd.Series(range(ModelPeriod.StartYear.iloc[0]-1,
                                   ModelPeriod.EndYear.iloc[0]+1),
                             name="InstYear")
        x = x.merge(InstYear, how = "cross")
    else:
        x = pd.DataFrame()

    if isinstance(ECT_EmissionDetails, pd.DataFrame):
        y = ECT_EmissionDetails.merge(ect, on="EnergyConvTech")
        if not x.empty:
            return fs.override_dataframe(x, y, ['EnergyConvTech',
                                                "EmissionType",
                                                "InstYear"])
        else:
            return y
    else:
        return x


def compute_ect_emission(ECTInput,
                         emission_data,
                         name):
    """Computes EmissionDom or EmissionImp depending on ECTInput and name;
    name can be 'Dom' or 'Imp'
    """
    ect_input = ECTInput.rename(columns={"EnergyConvTechFiltered":
                                         "EnergyConvTech"})
    cols = utilities.get_all_structure_columns(ect_input)
    ect_input = ect_input.merge(emission_data)
    ect_input[name] = ect_input[f'ECTInput{name}'] *\
        ect_input[f'{name}EmissionFactor']

    ect_input = season_wise(ect_input, name)

    othercols = ['EnergyCarrier',
                 'EnergyConvTech',
                 'InstYear',
                 'EmissionType',
                 name]
    return ect_input[cols + othercols]


def get_ectemissions_column_order(emissions):
    """column names and their order in ECTEmissions
    """
    othercols = ['EnergyCarrier',
                 'EmissionType',
                 'EnergyConvTech']

    if 'InstYear' in emissions.columns:
        othercols.append('InstYear')

    geocols = utilities.get_geographic_columns_from_dataframe(emissions)
    timecols = utilities.get_time_columns_from_dataframe(emissions)

    return othercols + geocols + timecols + ["Emission"]


def check_enduse():
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

    path = get_supply_output_path()
    files = ["EndUseDemandMetByDom.csv",
             "EndUseDemandMetByImp.csv"]
    exists = [os.path.exists(os.path.join(path, f)) for f in files]

    if not all(exists):
        print("Skipping EndUseEmissions computation as the required supply outputs (EndUseDemandMetByDom, EndUseDemandMetByImp) are not present")
        logger.warning(
            "Skipping EndUseEmissions computation as the required supply outputs (EndUseDemandMetByDom, EndUseDemandMetByImp) are not present")

    return all(exists)


def emission_types_exist():
    """checks if emissions computation is doable or not.
    Some minimum set of parameters are required to do emissions postprocessing.
    if those parameters are absent this will return False.
    """
    EmissionTypes = loaders.get_parameter("EmissionTypes")

    if not isinstance(EmissionTypes, pd.DataFrame):
        return False
    return True


def combine_physical_carriers():
    PhysicalPrimaryCarriers = loaders.get_parameter("PhysicalPrimaryCarriers")
    PhysicalDerivedCarriers = loaders.get_parameter("PhysicalDerivedCarriers")
    PDC = PhysicalDerivedCarriers.rename(
        columns={"EnergyDensity": "DomEnergyDensity"})
    PDC['ImpEnergyDensity'] = PDC['DomEnergyDensity']
    return pd.concat([PDC, PhysicalPrimaryCarriers])


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


@functools.lru_cache()
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


def enduse_emissions():
    """
    Compute EndUseDemandEmission
    """
    print("Computing EndUseEmissions")
    logger.info("Computing EndUseEmissions")
    EndUseDemandEnergy = get_end_use_demand_energy()
    EndUseDemandMetByDom = handle_day_no(
        read_supply_output("EndUseDemandMetByDom"))
    EndUseDemandMetByImp = handle_day_no(
        read_supply_output("EndUseDemandMetByImp"))

    physical_carriers = combine_physical_carriers()

    fractions = compute_fractions(EndUseDemandEnergy,
                                  EndUseDemandMetByDom,
                                  EndUseDemandMetByImp,
                                  physical_carriers)

    MetDemandEnergyFraction = fractions['MetDemandEnergyFraction']
    DemandMetByDomEnergyFraction = fractions['DemandMetByDomEnergyFraction']

    fractions['DemandMetByImpEnergyFraction'] = 1 - \
        DemandMetByDomEnergyFraction

    emissions_ = compute_emissions(fractions)

    write_emissions_results(emissions_, "EndUseEmissions.csv")


def compute_emissions(fractions):
    """Computes emission values for end use demand.
    """
    if no_demand_output_exists():
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
            ds_bottomup_set = {tup[0] for tup in ds_es_ec_st}
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


def combine_st_emission_data(ds_bottomup_set):
    """For each DS, combines ST_EmissionDetails and PhysicalCarrierEmissions
    with the help of ST->EnergyCarrier mapping to get final emission data
    for that demand sector.
    Finally, the thus combined emission data for each DS is concatenated
    together over all demand sectors and returned.
    """
    ST_Info = loaders.get_parameter("ST_Info")
    PhysicalCarrierEmissions = loaders.get_parameter(
        "PhysicalCarrierEmissions")

    if isinstance(PhysicalCarrierEmissions, pd.DataFrame):
        x = PhysicalCarrierEmissions.merge(ST_Info, on="EnergyCarrier")

        ModelPeriod = loaders.get_parameter("ModelPeriod")
        Year = pd.Series(range(ModelPeriod.StartYear.iloc[0],
                               ModelPeriod.EndYear.iloc[0]+1),
                         name="Year")
        x = x.merge(Year, how = "cross")
    else:
        x = pd.DataFrame()

    emission_data_df_list = []

    for ds in ds_bottomup_set:
        ST_EmissionDetails = loaders.get_parameter("ST_EmissionDetails",
                                                   demand_sector=ds)
        if isinstance(ST_EmissionDetails, pd.DataFrame):
            y = ST_EmissionDetails.merge(ST_Info, on="ServiceTech")

            if not x.empty:
                z = fs.override_dataframe(x, y, ['ServiceTech',
                                                 'EmissionType',
                                                 'Year'])
            else:
                z = y
        else:
            z = x

        if not z.empty:
            z.insert(0, 'DemandSector', ds)
            emission_data_df_list.append(z)

    return pd.concat(emission_data_df_list, ignore_index = True)


def get_bottomup_emission_data(EnergyDemandMet):
    if 'ServiceTech' in EnergyDemandMet.columns:
        return combine_st_emission_data()
    else:
        return loaders.get_parameter("PhysicalCarrierEmissions")


def compute_end_use_emission(EnergyDemandMet, emission_data):
    """compute end use emissions from Dom and Imp sources and also their sum
    """
    df = EnergyDemandMet.merge(emission_data)
    df['EmissionImp'] = df["EnergyDemandMetImp"] * \
        df["ImpEmissionFactor"]
    df['EmissionDom'] = df["EnergyDemandMetDom"] * \
        df["DomEmissionFactor"]

    df['Emission'] = df['EmissionImp'] + df['EmissionDom']
    return df


def set_index(data, indexcols=['EnergyCarrier']):
    """sets index to all structural columns and additional indexcols
    """
    data = data.reset_index()
    cols = utilities.get_all_structure_columns(data)
    return data.set_index(cols + indexcols)


def demand_col(data):
    """returns names of demand column that should be used as demand
    """
    if "SeasonEnergyDemand" in data.columns:
        return "SeasonEnergyDemand"
    else:
        return "EnergyDemand"


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


def demand_filepath(ds, es, ec, st=None):
    path = get_demand_output_path()
    if st:
        filename = f"{ds}_{es}_{st}_{ec}_Demand.csv"
    else:
        filename = f"{ds}_{es}_{ec}_Demand.csv"
    return os.path.join(path, filename)


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


def get_ds_es_ec(bottomup=True):

    m = demandio
    ds_es_func = m.get_bottomup_ds_es if bottomup else m.get_nonbottomup_ds_es

    DS_ES_EC_DemandGranularity_Map = loaders.get_parameter(
        "DS_ES_EC_DemandGranularity_Map")
    ds_es_ec = DS_ES_EC_DemandGranularity_Map.set_index(
        ['DemandSector', 'EnergyService', 'EnergyCarrier']).index
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


def energy_demand_met_(Demand,
                       fractions):
    """computes EnergyDemandMet Dom and Imp for a specific end-use Demand
    """

    demand = demand_col(Demand)
    df = fractions.merge(Demand)
    df[f"EnergyDemandMetImp"] = df[demand] *\
        df['MetDemandEnergyFraction'] *\
        df['DemandMetByImpEnergyFraction'] / df["ImpEnergyDensity"]

    df[f"EnergyDemandMetDom"] = df[demand] *\
        df['MetDemandEnergyFraction'] *\
        df['DemandMetByDomEnergyFraction'] / df["DomEnergyDensity"]

    return df


def get_physical_nonbottomup_ds_es_ec_st():
    """Returns nonbottomup tuples of ds,es,ec,st for physical carriers.
    st is always None
    """
    ds_es_ec = filter_physical(get_ds_es_ec(bottomup=False))
    return [(*item, None) for item in ds_es_ec]


def emissions(model_instance_path,
              scenario: str,
              output: str,
              demand_output,
              supply_output,
              logger_level: str):
    """Emission computation api function
    """
    global logger

    if not loaders.sanity_check_cmd_args("Common",
                                         model_instance_path,
                                         scenario,
                                         logger_level,
                                         1,
                                         "rumi_postprocess"):
        return

    init_config(model_instance_path,
                scenario,
                output,
                supply_output,
                demand_output)

    if not os.path.exists(filemanager.scenario_path()):
        print(f"Invalid scenario: {scenario}")
        return

    init_logger("PostProcess", logger_level)
    logger = logging.getLogger("rumi.processing.emissions")

    try:
        if not emission_types_exist():
            logger.warning(
                "EmissionTypes parameter is absent, hence emissions can not be computed")
            print("EmissionTypes parameter is absent, hence emissions can not be computed")
            return

        if check_ect():
            ect_emissions()

        if check_enduse():
            if all_demand_outputs_exist() or no_demand_output_exists():
                enduse_emissions()
            else:
                print(
                    "Some of the required demand outputs are absent, hence EndUseEmissions will not be computed")
                logger.error(
                    "Some of the required demand outputs are absent, hence EndUseEmissions will not be computed")
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        time.sleep(1)
        get_event().set()


def init_config(model_instance_path,
                scenario,
                output,
                supply_output,
                demand_output):
    """initialize config. sets necessary configuration parameters.
    """
    config.initialize_config(model_instance_path, scenario)
    if output:
        config.set_config("output", output)
    if supply_output:
        config.set_config("supply_output", supply_output)
    if demand_output:
        config.set_config("demand_output", demand_output)


@click.command()
@click.option("-m", "--model_instance_path",
              type=click.Path(exists=True),
              help="Path of the model instance root folder")
@click.option("-s", "--scenario",
              help="Name of the scenario within specified model")
@click.option("-o", "--output",
              help="Path of the output folder",
              default=None)
@click.option("-D", "--demand_output",
              type=click.Path(exists=True),
              help="Path of Demand processing output folder",
              default=None)
@click.option("-S", "--supply_output",
              type=click.Path(exists=True),
              help="Path of Supply processing output folder",
              default=None)
@click.option("-l", "--logger_level",
              help="Level for logging: one of INFO, WARN, DEBUG or ERROR. (default: INFO)",
              default="INFO")
def main(model_instance_path,
         scenario: str,
         output: str,
         demand_output,
         supply_output,
         logger_level: str):
    """Post processing script. Supports computation of ECT emission  and EndUse emission.
    """

    emissions(model_instance_path,
              scenario,
              output,
              demand_output,
              supply_output,
              logger_level)


if __name__ == "__main__":
    main()
