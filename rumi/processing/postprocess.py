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
import os
import logging
import functools
import click
import pandas as pd
import numpy as np
from rumi.io import loaders
from rumi.io import system
from rumi.io import config
from rumi.io import constant
from rumi.io import filemanager
from rumi.io import utilities
from rumi.io import supply
from rumi.io import demand as demandio
from rumi.io import functionstore as fs
from rumi.processing import utilities as putils
from rumi.processing import emission

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
    write_postprocess_results(
        emissions[get_ectemissions_column_order(emissions)])


def write_postprocess_results(data, filename="ECTEmissions.csv"):
    """Writes postprocess results to the specified file
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


def season_wise(data, colname):
    """Multiplies values in the given column with number of days in season
    and weight for given DayType, if required
    """
    if "DayType" not in data.columns:
        return data

    d = utilities.seasons_size()
    d[np.NaN] = 1
    data['NumDays'] = data['Season'].map(d)

    DayTypes = loaders.get_parameter("DayTypes")
    DayTypes = pd.concat([DayTypes,
                          pd.DataFrame({"DayType": [np.NaN],
                                        "Weight": [1]})]).reset_index(drop=True)
    data = data.merge(DayTypes)

    data['multiplier'] = np.ones_like(data[colname].values)

    if "DayNo" not in data.columns:
        data['multiplier'] = np.where(data["DayType"].isna(),
                                      1,
                                      data["NumDays"] * data["Weight"])
    else:
        data['multiplier'] = np.where(data["DayType"].notna() &
                                      data["DayNo"].isna(),
                                      data["NumDays"] *
                                      data["Weight"],
                                      1)
    data[colname] = data[colname] *\
        data['multiplier']

    del data['NumDays']
    del data['Weight']
    return data


def combine_emission_data(ECT_EmissionDetails,
                          PhysicalCarrierEmissions,
                          EnergyConvTechnologies):
    """Combines ECT_EmissionDetails and PhysicalCarrierEmissions with
    the help of EnergyConvTechnologies to get final emission data
    """
    ect = EnergyConvTechnologies.rename(columns={"InputEC": "EnergyCarrier"})

    if isinstance(PhysicalCarrierEmissions, pd.DataFrame):
        x = PhysicalCarrierEmissions.merge(ect, on="EnergyCarrier")
        del x['Year']
        ModelPeriod = loaders.get_parameter("ModelPeriod")
        InstYear = pd.Series(range(ModelPeriod.StartYear.iloc[0]-1,
                                   ModelPeriod.EndYear.iloc[0]+1),
                             name="InstYear")
        x = x.merge(InstYear, how="cross")
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


def emission_types_exist():
    """checks if emissions computation is doable or not.
    Some minimum set of parameters are required to do emissions postprocessing.
    if those parameters are absent this will return False.
    """
    EmissionTypes = loaders.get_parameter("EmissionTypes")

    if not isinstance(EmissionTypes, pd.DataFrame):
        return False
    return True


class EndUseEmissions(emission.EmissionDemandOnly):

    def initilize_met_demand_params(self):
        self.EndUseDemandMetByDom = emission.handle_day_no(
            read_supply_output("EndUseDemandMetByDom"))
        self.EndUseDemandMetByImp = emission.handle_day_no(
            read_supply_output("EndUseDemandMetByImp"))
    
    def write_results(self, emissions_):
        write_postprocess_results(emissions_, "EndUseEmissions.csv")


        
def demand_filepath(ds, es, ec, st=None):
    path = get_demand_output_path()
    path = os.path.join(path, "DemandSector", ds, es)
    if st:
        filename = f"{ds}_{es}_{st}_{ec}_Demand.csv"
    else:
        filename = f"{ds}_{es}_{ec}_Demand.csv"
    return os.path.join(path, filename)


def get_physical_bottomup_ds_es_ec_st():
    ds_es_ec = filter_physical(get_ds_es_ec(bottomup=True))
    ds_es_ec_st = [(ds, es, ec, st) for ds, es, ec in ds_es_ec
                   for st in demandio.get_service_techs(ds, es, ec)]

    return ds_es_ec_st


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


def aggregate_to_coarset(data, target):
    groupcols = putils.get_coarsest(data)
    result_ = 0
    for d in data:
        result_ = result_ + utilities.groupby(d, groupcols, target=target)
    return result_.reset_index()


def get_physical_nonbottomup_ds_es_ec_st():
    """Returns nonbottomup tuples of ds,es,ec,st for physical carriers.
    st is always None
    """
    ds_es_ec = filter_physical(get_ds_es_ec(bottomup=False))
    return [(*item, None) for item in ds_es_ec]


@functools.lru_cache()
def load_logger(name):
    global logger
    logger = logging.getLogger(name)


def get_supply_output_path():
    supply_output = config.get_config_value("supply_output")
    if supply_output:
        path = filemanager.get_custom_output_path("Supply", supply_output)
    else:
        scenario_path = filemanager.scenario_location()
        path = os.path.join(scenario_path, "Supply", 'Output')

    return os.path.join(path, "Run-Outputs")

    
def check_enduse_supply():
    
    path = get_supply_output_path()
    files = ["EndUseDemandMetByDom.csv",
             "EndUseDemandMetByImp.csv"]
    exists = [os.path.exists(os.path.join(path, f)) for f in files]

    if not all(exists):
        print("Skipping EndUseEmissions computation as the required supply outputs (EndUseDemandMetByDom, EndUseDemandMetByImp) are not present")
        logger.warning(
            "Skipping EndUseEmissions computation as the required supply outputs (EndUseDemandMetByDom, EndUseDemandMetByImp) are not present")

    return all(exists)

    
def emissions(model_instance_path,
              scenario: str,
              output: str,
              demand_output,
              supply_output,
              logger_level: str,
              no_shutdown=True):
    """Emission computation api function
    """

    try:
        system.SystemLauncher(model_instance_path,
                              scenario,
                              "PostProcess",
                              logger_level,
                              output=output,
                              supply_output=supply_output,
                              demand_output=demand_output)
    except system.ScenarioError as se:
        return

    load_logger("rumi.processing.postprocess")

    try:
        if not emission_types_exist():
            logger.warning(
                "EmissionTypes parameter is absent, hence emissions can not be computed")
            print(
                "EmissionTypes parameter is absent, hence emissions can not be computed")
            return

        if check_ect():
            ect_emissions()

        if emission.check_enduse_common() and check_enduse_supply():
            if emission.all_demand_outputs_exist() or \
               emission.no_demand_output_exists():
                e = EndUseEmissions()
                e.write_results(e.compute())
            else:
                print(
                    "Some of the required demand outputs are absent, hence EndUseEmissions will not be computed")
                logger.error(
                    "Some of the required demand outputs are absent, hence EndUseEmissions will not be computed")
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        if no_shutdown:
            logger.debug(
                "Not shutting down logger after emission computation!")
        else:
            logger.debug("shutting down!")
            system.SystemLauncher.shutdown()


def supply_output_exists(outputname):
    folderpath = get_supply_output_path()
    filepath = os.path.join(folderpath, ".".join([outputname, "csv"]))
    return os.path.exists(filepath) and os.path.isfile(filepath)


def check_tpes_inputs():
    """checks if tpes inputs and supply outputs required for
    TPES  computation exist.
    """
    EnergyConvTechnologies = supply.get_filtered_parameter(
        "EnergyConvTechnologies")
    common_names = ["PhysicalPrimaryCarriers",
                    "NonPhysicalPrimaryCarriers",
                    "PhysicalDerivedCarriers",
                    "NonPhysicalDerivedCarriers"]

    supply_names = ["EnergyConvTechnologies",
                    "ECT_EfficiencyCostMaxAnnualUF"]

    common_params = [loaders.get_parameter(p) for p in common_names]
    supply_params = [supply.get_filtered_parameter(p) for p in supply_names]

    common_valid = any(isinstance(p, pd.DataFrame) for p in common_params)
    # supply_valid = all(isinstance(p, pd.DataFrame) for p in supply_params)

    if not common_valid:
        logger.warning(
            f"At least one of {common_names} should be present in common paramters to do TPES computation")
        logger.warning(
            "Skipping TPES computation as none of the carriers information is present")
        print(
            "Skipping TPES computation as none of the carriers information is present")

        return False

    files = ["DomesticProd",
             "Import",
             "OutputFromECTiy"]

    if (supply_output_exists("DomesticProd") and supply_output_exists("Import")) or\
       supply_output_exists("OutputFromECTiy"):

        return True
    else:
        print(
            f"Skipping TPES computation as the required supply few of outputs {files} are not present")
        logger.warning(
            f"Skipping TPES computation as the required supply few of outputs {files} are not present")

        return False


def get_tpes_energycarriers(DomesticProd, Import, OutputFromECTiy):
    """get all energycarriers that contribute to TPES
    """
    EnergyConvTechnologies = supply.get_filtered_parameter(
        "EnergyConvTechnologies")
    NonPhysicalPrimaryCarriers = loaders.get_parameter(
        "NonPhysicalPrimaryCarriers")
    PhysicalPrimaryCarriers = loaders.get_parameter(
        "PhysicalPrimaryCarriersEnergyDensity")

    ect_ec = set(EnergyConvTechnologies.InputEC)
    ec = ect_ec & set(NonPhysicalPrimaryCarriers.EnergyCarrier)

    if supply_output_exists("DomesticProd") and\
       supply_output_exists("Import"):
        A = (set(DomesticProd.EnergyCarrier) & set(PhysicalPrimaryCarriers.EnergyCarrier)) | \
            (set(Import.EnergyCarrier) & set(PhysicalPrimaryCarriers.EnergyCarrier))
    else:
        A = set()

    if isinstance(NonPhysicalPrimaryCarriers, pd.DataFrame) and \
       supply_output_exists("OutputFromECTiy"):
        nppec = NonPhysicalPrimaryCarriers.rename(
            columns={'EnergyCarrier': "InputEC"})
        B = set(EnergyConvTechnologies.merge(
            nppec).merge(OutputFromECTiy).OutputDEC)
    else:
        B = set()

    return A | B


def get_coarsest_balancing_columns(ecs):
    """Find coarsest time and geographics granularity for given energycarries and
    return columns for the same
    """
    g = min([demandio.balancing_area(e) for e in ecs],
            key=lambda x: len(constant.GEO_COLUMNS[x]))
    t = min([demandio.balancing_time(e) for e in ecs],
            key=lambda x: len(constant.TIME_COLUMNS[x]))
    return constant.TIME_COLUMNS[t] + constant.GEO_COLUMNS[g]


@functools.lru_cache(maxsize=None)
def get_unit_fatcor(EnergyCarrier):
    """returns factor to convert EnergyUnit of EnergyCarrier to TPES unit
    """
    PPC = loaders.get_parameter("PhysicalPrimaryCarriers")
    NPDC = loaders.get_parameter("NonPhysicalDerivedCarriers")
    PDC = loaders.get_parameter("PhysicalDerivedCarriers")

    if EnergyCarrier in PPC.EnergyCarrier.values:
        unit = PPC[PPC.EnergyCarrier == EnergyCarrier]['EnergyUnit'].values[0]
    elif EnergyCarrier in PDC.EnergyCarrier.values:
        unit = PDC[PDC.EnergyCarrier == EnergyCarrier]['EnergyUnit'].values[0]
    elif EnergyCarrier in NPDC.EnergyCarrier.values:
        unit = NPDC[NPDC.EnergyCarrier ==
                    EnergyCarrier]['EnergyUnit'].values[0]
    else:
        raise Exception(f"Unexpected EnergyCarrier {EnergyCarrier}")

    EnergyUnitConversion = loaders.get_config_parameter("EnergyUnitConversion")
    euc = EnergyUnitConversion.set_index("EnergyUnit")
    TPES_EU = config.get_config_value("TPES_EU")
    return euc.loc[unit, TPES_EU]


def tpes_nppec(granularity_columns,
               OutputFromECTiy):
    """Computes TPES contribution by derived carriers
    """
    OutputFromECTiy = season_wise(OutputFromECTiy, "OutputFromECTiy")
    NonPhysicalPrimaryCarriers = loaders.get_parameter(
        "NonPhysicalPrimaryCarriers")
    EnergyConvTechnologies = supply.get_filtered_parameter(
        "EnergyConvTechnologies")
    PhysicalDerivedCarriers = loaders.get_parameter(
        "PhysicalDerivedCarriersEnergyDensity")
    NonPhysicalDerivedCarriers = loaders.get_parameter(
        "NonPhysicalDerivedCarriers")

    nppec = NonPhysicalPrimaryCarriers.rename(
        columns={'EnergyCarrier': "InputEC"})
    data = EnergyConvTechnologies.merge(nppec).merge(OutputFromECTiy)

    PDC = PhysicalDerivedCarriers.rename(
        columns={"EnergyCarrier": "OutputDEC"})
    NPDC = NonPhysicalDerivedCarriers.rename(
        columns={"EnergyCarrier": "OutputDEC"})

    nppec_pd_data = data.merge(PDC).rename(
        columns={"OutputDEC": "EnergyCarrier"})
    nppec_npd_data = data.merge(NPDC).rename(
        columns={"OutputDEC": "EnergyCarrier"})

    tpes_dfs = []
    tpes_pd = compute_tpes_(nppec_pd_data,
                            granularity_columns,
                            entity='EnergyConvTech',
                            colname="OutputFromECTiy",
                            density="EnergyDensity",
                            conv_eff='ConvEff')
    if len(tpes_pd) > 0:
        tpes_dfs.append(tpes_pd)

    tpes_npd = compute_tpes_(nppec_npd_data,
                             granularity_columns,
                             entity='EnergyConvTech',
                             colname="OutputFromECTiy",
                             conv_eff='ConvEff')

    if len(tpes_npd) > 0:
        tpes_dfs.append(tpes_npd)
    return pd.concat(tpes_dfs)


def compute_tpes_(data,
                  granularity_columns,
                  entity='EnergyCarrier',
                  colname='DomesticProd',
                  density=None,
                  conv_eff=None):
    """Helper function to compute TPES.
    """
    tpes = []

    for ec in data[entity].unique():
        data_ec = utilities.filter_empty(
            data.query(f"{entity} == '{ec}'"))
        indexcols = utilities.get_all_structure_columns(data_ec)
        if 'InstYear' in data_ec:
            indexcols.append('InstYear')
        data_ec = data_ec.set_index(indexcols)

        if conv_eff:
            ConvEff = get_conversion_eff_vector(ec)
            data_ec = data_ec.join(ConvEff)
            ConvEff = data_ec['ConvEff']
        else:
            ConvEff = 1

        energy_density = data_ec[density] if density else 1
        ec_ = ec if entity == 'EnergyCarrier' else data_ec['EnergyCarrier'].values[0]
        conversion_factor = get_unit_fatcor(ec_)

        data_ec['TPES'] = data_ec[colname]/ConvEff * \
            energy_density * conversion_factor
        tpes_ = data_ec.reset_index().groupby(
            granularity_columns, sort=False)['TPES'].sum()
        tpes_ = tpes_.reset_index()
        if entity == 'EnergyCarrier':
            tpes_['EnergyCarrier'] = ec
        else:
            tpes_['EnergyCarrier'] = data_ec['InputEC'].values[0]

        tpes.append(tpes_)

    cols = granularity_columns + ['EnergyCarrier']
    if tpes:
        return pd.concat(tpes).groupby(cols, sort=False)['TPES'].sum()
    else:
        return pd.DataFrame()


def get_conversion_eff_vector(EnergyConvTech):
    """Conversion Efficiency for given EnergyConvTech
    """
    ECT_EfficiencyCostMaxAnnualUF = supply.get_filtered_parameter(
        "ECT_EfficiencyCostMaxAnnualUF").replace("",
                                                 np.NaN)
    eff = utilities.filter_empty(ECT_EfficiencyCostMaxAnnualUF.query(
        f"EnergyConvTech == '{EnergyConvTech}'"))
    indexcols = utilities.get_all_structure_columns(eff)
    indexcols.append('InstYear')
    return eff.set_index(indexcols)['ConvEff']


def tpes_ppec(granularity_columns,
              DomesticProd,
              Import):
    """Computes TPES contribution by physical primary carriers
    """
    PhysicalPrimaryCarriers = loaders.get_parameter(
        "PhysicalPrimaryCarriersEnergyDensity")

    domestic_prod = season_wise(
        DomesticProd, "DomesticProd").merge(PhysicalPrimaryCarriers)
    import_ = season_wise(Import, "Import").merge(PhysicalPrimaryCarriers)

    tpes_dom_prod = compute_tpes_(domestic_prod,
                                  granularity_columns,
                                  entity='EnergyCarrier',
                                  colname="DomesticProd",
                                  density="DomEnergyDensity")

    tpes_import = compute_tpes_(import_,
                                granularity_columns,
                                entity='EnergyCarrier',
                                colname="Import",
                                density="ImpEnergyDensity")

    tpes_PPEC = pd.concat([tpes_dom_prod, tpes_import])
    return tpes_PPEC


def compute_total_tpes():
    DomesticProd, Import, OutputFromECTiy = None, None, None

    if supply_output_exists("DomesticProd"):
        DomesticProd = read_supply_output("DomesticProd")
    if supply_output_exists("Import"):
        Import = read_supply_output("Import")
    if supply_output_exists("OutputFromECTiy"):
        OutputFromECTiy = read_supply_output("OutputFromECTiy")

    ecs = get_tpes_energycarriers(DomesticProd, Import, OutputFromECTiy)
    granularity_columns = get_coarsest_balancing_columns(ecs)
    TPES_PPEC = tpes_ppec(granularity_columns,
                          DomesticProd,
                          Import)
    TPES_NPPEC = tpes_nppec(granularity_columns, OutputFromECTiy)
    TPES = pd.concat([TPES_PPEC, TPES_NPPEC]).reset_index().rename(
        columns={0: 'TPES'})
    return TPES.groupby(['EnergyCarrier'] + granularity_columns, sort=False)['TPES'].sum()


def tpes(model_instance_path,
         scenario: str,
         output: str,
         supply_output,
         logger_level: str,
         no_shutdown=True):
    """api function for TPES
    """

    try:
        system.SystemLauncher(model_instance_path,
                              scenario,
                              "PostProcess",
                              logger_level,
                              output=output,
                              supply_output=supply_output)
    except system.ScenarioError as se:
        return

    load_logger("rumi.processing.postprocess")

    try:
        logger.info("Computing TPES")
        print("Computing TPES")
        if not check_tpes_inputs():
            print("Aborting TPES computation")
            return

        EnergyUnitConversion = loaders.get_config_parameter(
            "EnergyUnitConversion")

        TPES_EU = config.get_config_value("TPES_EU")
        logger.debug(f"Using TPES_EU = {TPES_EU}")

        if TPES_EU and TPES_EU not in EnergyUnitConversion.columns:
            logger.error(
                "TPES_EU unit has invalid value. Only units given in EnergyUnitConversion parameter can be used.")
            print("TPES_EU unit has invalid value. Only units given in EnergyUnitConversion parameter can be used.")
            print("Aborting TPES computation")

            return
        elif not TPES_EU:
            logger.error(
                "Value for TPES_EU is not given. It should be given in Config.yml")
            print("Value for TPES_EU is not given. It should be given in Config.yml")
            print("Aborting TPES computation")
            return

        TPES = compute_total_tpes().reset_index()
        write_postprocess_results(TPES, "TPES.csv")

    except Exception as e:
        logger.exception(e)
        raise (e)
    finally:
        if no_shutdown:
            logger.debug("Not shutting down logger after TPES computation")
        else:
            logger.debug("shutting down!")
            system.SystemLauncher.shutdown()


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
              help="Level for logging: one of DEBUG, INFO, WARN or ERROR. (default: INFO)",
              default="INFO")
@click.option("--compute_emission/--no-emission",
              help="Enable/disable validation (default: Enabled)",
              default=False)
@click.option("--compute_tpes/--no-tpes",
              help="Enable/disable validation (default: Enabled)",
              default=False)
def main(model_instance_path,
         scenario: str,
         output: str,
         demand_output,
         supply_output,
         logger_level: str,
         compute_emission: bool,
         compute_tpes: bool):
    """Post processing script. Supports computation of ECT emission  and EndUse emission and TPES (Total Primary Energy Supply)
    """
    if not loaders.sanity_check_cmd_args("Common",
                                         model_instance_path,
                                         scenario,
                                         logger_level,
                                         1,
                                         "rumi_postprocess"):
        return

    bothfalse = not compute_emission and not compute_tpes

    if compute_emission or bothfalse:
        emissions(model_instance_path,
                  scenario,
                  output,
                  demand_output,
                  supply_output,
                  logger_level,
                  no_shutdown=not (compute_emission and not compute_tpes))
    if compute_tpes or bothfalse:
        tpes(model_instance_path,
             scenario,
             output,
             supply_output,
             logger_level,
             no_shutdown=False)


if __name__ == "__main__":
    main()
