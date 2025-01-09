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

from __future__ import division

import sys
import shutil
import typing
import logging
import csv
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path

from rumi.io.logger import init_logger, get_event
from rumi.io import config
from rumi.io import loaders
from rumi.io import supply

global logger

                        #########################################
#####################           Set up logger                   #############
                        #########################################

LOGFILE_NAME = "rumi_supply.log"
LOGFORMAT_STR = "%(asctime)s %(levelname)s %(name)s %(message)s"

logging.basicConfig(filename = LOGFILE_NAME, format = LOGFORMAT_STR, 
                    filemode='w', level = logging.INFO)
logger = logging.getLogger("rumi.processing.supply")

logger.info("Entry")

                        #########################################
#####################           Command line arguments          #############
                        #########################################

SCENARIOS_FOLDER_NAME = "Scenarios"
SUPPLY_FOLDER_NAME = "Supply"
OUTPUT_FOLDER_NAME = "Output"
PARAM_FOLDER_NAME = "Run-Inputs"
VAR_FOLDER_NAME = "Run-Outputs"

def script_exit(exit_code):
    logger.info("Exit")
    logging.shutdown()
    shutil.copy(LOGFILE_NAME, output_folder_path)
    sys.exit(exit_code)

def parse_arguments():
    parser = ArgumentParser(description = "Supply processing for the given model")

    parser.add_argument("-o", "--output_folder",
                        help = "Path of the output folder" + \
                        " (default is the output folder within the scenario)")

    required = parser.add_argument_group("required named arguments")
   
    required.add_argument("-m", "--model_instance_path", 
                          help = "Path of the model instance top-level folder",
                          required = True)
    required.add_argument("-s", "--scenario",
                          help = "Name of the scenario within specified model",
                          required = True)

    args = parser.parse_args()

    return args

def process_config_params():
    config_location = config.get_config_value("config_location")
    solver_name = config.get_config_value("solver_name")
    solver_executable = config.get_config_value("solver_executable")
    solver_options_file = config.get_config_value("solver_options_file")
    symbolic_solver_labels = config.get_config_value("symbolic_solver_labels")

    if ((solver_name is not None) and (solver_name != "None")):
        print("Solver name:", solver_name)
    else:
        solver_name = None

    if ((solver_executable is not None) and (solver_executable != "None")):
        print("Solver executable:", solver_executable)
    else:
        solver_executable = None

    if ((solver_options_file is not None) and (solver_options_file != "None")):
        print("Solver options file:", solver_options_file)
    else:
        solver_options_file = None

    if ((symbolic_solver_labels is not None) and (symbolic_solver_labels != "None")):
        symbolic_solver_labels = (symbolic_solver_labels.lower() == "true")
        print("Symbolic solver labels:", symbolic_solver_labels)
    else:
        symbolic_solver_labels = False

    solver_options_dict = dict()

    if (solver_options_file is not None):
        solver_options_file = Path(config_location) / solver_options_file
        if (not solver_options_file.is_file()):
            print("Warning: Specified solver option file %s does not exist" % 
                  solver_options_file)
        else:
            with open(solver_options_file, newline = '') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    solver_options_dict[row[0]] = row[1]
            
            print("The following solver options have been read from", solver_options_file)
            print(solver_options_dict)

    return solver_name, solver_executable, solver_options_dict, \
           symbolic_solver_labels


args = parse_arguments()


model_instance_path = Path(args.model_instance_path)
scenario_path = model_instance_path / SCENARIOS_FOLDER_NAME / args.scenario

if (args.output_folder is not None):
    output_folder_path = Path(args.output_folder) / args.scenario
else:
    output_folder_path = scenario_path

output_folder_path = output_folder_path / SUPPLY_FOLDER_NAME / OUTPUT_FOLDER_NAME


print("Model instance path:", model_instance_path)
print("Scenario path:", scenario_path)
print("Output folder:", output_folder_path)



if (not model_instance_path.is_dir()):
    print("ERROR: Model instance path does not exist")
    sys.exit(-1)

if (not scenario_path.is_dir()):
    print("ERROR: Scenario does not exist")
    sys.exit(-1)

if (output_folder_path.is_dir()):
    print("\nWARNING: Output folder already exist; " + \
          "any previous outputs will be overwritten")

    user_input = input("Enter Y or y to continue; anything else to cancel: ")
    if user_input.strip().lower()[:1] != "y":
        script_exit(1)
    print()

output_path_param = output_folder_path / PARAM_FOLDER_NAME
output_path_var = output_folder_path / VAR_FOLDER_NAME

if (not output_path_var.is_dir()):
    output_path_var.mkdir(parents = True)

if (not output_path_param.is_dir()):
    output_path_param.mkdir(parents = True)

config.initialize_config(model_instance_path,
                         args.scenario)

solver_name, solver_executable, solver_options_dict, \
symbolic_solver_labels = process_config_params()

if (solver_name is None):
    print("ERROR: No solver specified")
    script_exit(-1)


                        #########################################
#####################           Validate parameters             #############
                        #########################################

def validate_params():
    try:
        logger.info("Validation of Common Parameters")
        common_validation_ret_val = loaders.validate_params("Common")
    except Exception as e:
        logger.exception(e)
        common_validation_ret_val = False

    try:
        logger.info("Validation of Supply Parameters")
        supply_validation_ret_val = loaders.validate_params("Supply")
    except Exception as e:
        logger.exception(e)
        supply_validation_ret_val = False

#    logger.info("Validation of Config Parameters")
#    config_validation_ret_val = loaders.validate_params("Config")

    return common_validation_ret_val, supply_validation_ret_val#, config_validation_ret_val

print("Validating parameter data")
common_validation_ret_val, supply_validation_ret_val = validate_params()
#common_validation_ret_val, supply_validation_ret_val, config_validation_ret_val = validate_params()

if (common_validation_ret_val == False):
    print("ERROR: Common parameters validation failed")
    script_exit(-1)

if (supply_validation_ret_val == False):
    print("ERROR: Supply parameters validation failed")
    script_exit(-1)
'''
if (config_validation_ret_val == False):
    print("ERROR: Config parameters validation failed")
    script_exit(-1)
'''

                         #########################################
#####################            Model Data Fetch                #############
                         #########################################

VERY_LARGE_VALUE = float(1e18)

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365
HOURS_PER_YEAR = (HOURS_PER_DAY * DAYS_PER_YEAR)
MONTHS_PER_YEAR = 12

MAX_TIME_LEVELS = 4
MAX_GEOGRAPHY_LEVELS = 4

SOURCE_SUFFIX_STR = "Src"
DESTINATION_SUFFIX_STR = "Dest"

CONFIG_PARAM = "CONFIG"
COMMON_PARAM = "COMMON"
SUPPLY_PARAM = "SUPPLY"

MODEL_PERIOD_PARAM_NAME = "ModelPeriod"
SEASONS_PARAM_NAME = "Seasons"
DAYTYPES_PARAM_NAME = "DayTypes"
DAYSLICES_PARAM_NAME = "DaySlices"
MODEL_GEOG_PARAM_NAME = "ModelGeography"
SUBGEOG1_PARAM_NAME = "SubGeography1"
SUBGEOG2_PARAM_NAME = "SubGeography2"
SUBGEOG3_PARAM_NAME = "SubGeography3"
PPEC_PARAM_NAME = "PhysicalPrimaryCarriers"
NPPEC_PARAM_NAME = "NonPhysicalPrimaryCarriers"
PDEC_PARAM_NAME = "PhysicalDerivedCarriers"
NPDEC_PARAM_NAME = "NonPhysicalDerivedCarriers"
PPEC_ENERGYDENSITY_PARAM_NAME = "PhysicalPrimaryCarriersEnergyDensity"
PDEC_ENERGYDENSITY_PARAM_NAME = "PhysicalDerivedCarriersEnergyDensity"
UNMET_DEMAND_VALUE_PARAM_NAME = "UnmetDemandValue"
ENERGYUNITCONV_PARAM_NAME = "EnergyUnitConversion"

STARTYEAR_COLUMN_NAME = "StartYear"
ENDYEAR_COLUMN_NAME = "EndYear"
SEASON_COLUMN_NAME = "Season"
STARTMONTH_COLUMN_NAME = "StartMonth"
STARTDATE_COLUMN_NAME = "StartDate"
STARTDAYOFYEAR_COLUMN_NAME = "StartDayOfYear"
ENDDAYOFYEAR_COLUMN_NAME = "EndDayOfYear"
DAYTYPE_COLUMN_NAME = "DayType"
WEIGHT_COLUMN_NAME = "Weight"
DAYSLICE_COLUMN_NAME = "DaySlice"
STARTHOUR_COLUMN_NAME = "StartHour"
ENDHOUR_COLUMN_NAME = "EndHour"
DURATION_COLUMN_NAME = "Duration"

YEAR_COLUMN_NAME = "Year"

MODELGEOGRAPHY_COLUMN_NAME = "ModelGeography"
SUBGEOGRAPHY1_COLUMN_NAME = "SubGeography1"
SUBGEOGRAPHY2_COLUMN_NAME = "SubGeography2"
SUBGEOGRAPHY3_COLUMN_NAME = "SubGeography3"

SRCMODELGEOGRAPHY_COLUMN_NAME = MODELGEOGRAPHY_COLUMN_NAME + SOURCE_SUFFIX_STR
SRCSUBGEOGRAPHY1_COLUMN_NAME = SUBGEOGRAPHY1_COLUMN_NAME + SOURCE_SUFFIX_STR
SRCSUBGEOGRAPHY2_COLUMN_NAME = SUBGEOGRAPHY2_COLUMN_NAME + SOURCE_SUFFIX_STR
SRCSUBGEOGRAPHY3_COLUMN_NAME = SUBGEOGRAPHY3_COLUMN_NAME + SOURCE_SUFFIX_STR

DESTMODELGEOGRAPHY_COLUMN_NAME = MODELGEOGRAPHY_COLUMN_NAME + DESTINATION_SUFFIX_STR
DESTSUBGEOGRAPHY1_COLUMN_NAME = SUBGEOGRAPHY1_COLUMN_NAME + DESTINATION_SUFFIX_STR
DESTSUBGEOGRAPHY2_COLUMN_NAME = SUBGEOGRAPHY2_COLUMN_NAME + DESTINATION_SUFFIX_STR
DESTSUBGEOGRAPHY3_COLUMN_NAME = SUBGEOGRAPHY3_COLUMN_NAME + DESTINATION_SUFFIX_STR

ENERGYCARRIER_COLUMN_NAME = "EnergyCarrier"
BALANCINGTIME_COLUMN_NAME = "BalancingTime"
BALANCINGAREA_COLUMN_NAME = "BalancingArea"
ENERGYUNIT_COLUMN_NAME = "EnergyUnit"
DOMENERGYDENSITY_COLUMN_NAME = "DomEnergyDensity"
IMPENERGYDENSITY_COLUMN_NAME = "ImpEnergyDensity"
ENERGYDENSITY_COLUMN_NAME = "EnergyDensity"

UNMETDEMANDVALUE_COLUMN_NAME = "UnmetDemandValue"

ENERGYUNIT_COLUMN_NAME = "EnergyUnit"

ENERGYUNIT1_COLUMN_NAME = "EnergyUnit1"
ENERGYUNIT2_COLUMN_NAME = "EnergyUnit2"
ENERGYUNITCONV_COLUMN_NAME = "EnergyUnitConv"

BALTIME_YR_STR = "YEAR"
BALTIME_SE_STR = "SEASON"
BALTIME_DT_STR = "DAYTYPE"
BALTIME_DS_STR = "DAYSLICE"

BALAREA_MG_STR = "MODELGEOGRAPHY"
BALAREA_SG1_STR = "SUBGEOGRAPHY1"
BALAREA_SG2_STR = "SUBGEOGRAPHY2"
BALAREA_SG3_STR = "SUBGEOGRAPHY3"

DUMMY_TIME_STR = ""
DUMMY_GEOG_STR = ""

PEC_INFO_PARAM_NAME = "PEC_Info"
PEC_PRODIMPCONSTRAINTS_PARAM_NAME = "PEC_ProdImpConstraints"
DEC_TAXATION_PARAM_NAME = "DEC_Taxation"
ECT_PARAM_NAME = "EnergyConvTechnologies"
ECT_CAPADDBOUNDS_PARAM_NAME = "ECT_CapAddBounds"
ECT_LIFETIME_PARAM_NAME = "ECT_Lifetime"
ECT_OPERATIONALINFO_PARAM_NAME = "ECT_OperationalInfo"
ECT_EFFICIENCYCOSTMAXANNUALUF_PARAM_NAME = "ECT_EfficiencyCostMaxAnnualUF"
ECT_MAXCUF_PARAM_NAME = "ECT_Max_CUF"
ECT_LEGACYCAPACITY_PARAM_NAME = "ECT_LegacyCapacity"
ECT_LEGACYRETIREMENT_PARAM_NAME = "ECT_LegacyRetirement"
EST_PARAM_NAME = "EnergyStorTechnologies"
EST_LIFETIME_PARAM_NAME = "EST_Lifetime"
EST_CAPADDBOUNDS_PARAM_NAME = "EST_CapAddBounds"
EST_DERATINGDEPTHOFDISCHARGE_PARAM_NAME = "EST_DeratingDepthOfDischarge"
EST_EFFICIENCYCOST_PARAM_NAME = "EST_EfficiencyCost"
EST_LEGACYDETAILS_PARAM_NAME = "EST_LegacyDetails"
EC_TRANSFERS_PARAM_NAME = "EC_Transfers"
ENDUSE_DEMANDENERGY_PARAM_NAME = "EndUseDemandEnergy"

USERCONSTRAINTS_PARAM_NAME = "UserConstraints"

CONSTRAINT_DICT_VECTORS_KEY = "VECTORS"
CONSTRAINT_DICT_BOUNDS_KEY = "BOUNDS"
USER_CONSTRAINT_NAME_PREFIX = "UserConstraint#"

NONENERGYSHARE_COLUMN_NAME = "NonEnergyShare"
DOMESTICPRICE_COLUMN_NAME = "DomesticPrice"
AVTAXOHDOM_COLUMN_NAME = "AVTaxOHDom"
FIXEDTAXOHDOM_COLUMN_NAME = "FixedTaxOHDom"
IMPORTPRICE_COLUMN_NAME = "ImportPrice"
AVTAXOHIMP_COLUMN_NAME = "AVTaxOHImp"
FIXEDTAXOHIMP_COLUMN_NAME = "FixedTaxOHImp"

MAXDOMESTICPRODUCT_COLUMN_NAME = "MaxDomesticProd"
MAXIMPORT_COLUMN_NAME = "MaxImport"

FIXEDTAXOH_COLUMN_NAME = "FixedTaxOH"

ENERGYCONVTECH_COLUMN_NAME = "EnergyConvTech"
INPUTEC_COLUMN_NAME = "InputEC"
OUTPUTDEC_COLUMN_NAME = "OutputDEC"
ANNUALOUTPUTPERUNITCAPACITY_COLUMN_NAME = "AnnualOutputPerUnitCapacity"

MAXCAPACITY_COLUMN_NAME = "MaxCapacity"
MINCAPACITY_COLUMN_NAME = "MinCapacity"

LIFETIME_COLUMN_NAME = "Lifetime"

INSTYEAR_COLUMN_NAME = "InstYear"
SELFCONS_COLUMN_NAME = "SelfCons"
CAPACITYDERATING_COLUMN_NAME = "CapacityDerating"
MAXRAMPUPRATE_COLUMN_NAME = "MaxRampUpRate"
MAXRAMPDOWNRATE_COLUMN_NAME = "MaxRampDownRate"

CONVEFF_COLUMN_NAME = "ConvEff"
FIXEDCOST_COLUMN_NAME = "FixedCost"
VARCOST_COLUMN_NAME = "VarCost"
MAXANNUALUF_COLUMN_NAME = "MaxAnnualUF"

MAXUF_COLUMN_NAME = "MaxUF"

LEGACYCAPACITY_COLUMN_NAME = "LegacyCapacity"

RETCAPACITY_COLUMN_NAME = "RetCapacity"

EC_DOM_STR = "EC_DOM"
EC_IMP_STR = "EC_IMP"

DAILY_RESET_STR = "DAILY"
SEASONAL_RESET_STR = "SEASONAL"
ANNUAL_RESET_STR = "ANNUAL"
NEVER_RESET_STR = "NEVER"

ENERGYSTORTECH_COLUMN_NAME = "EnergyStorTech"
STOREDEC_COLUMN_NAME = "StoredEC"
DOMORIMP_COLUMN_NAME = "DomOrImp"
MAXCHARGERATE_COLUMN_NAME = "MaxChargeRate"
MAXDISCHARGERATE_COLUMN_NAME = "MaxDischargeRate"
STORPERIODICITY_COLUMN_NAME = "StorPeriodicity"

LIFETIMEYEARS_COLUMN_NAME = "LifetimeYears"
LIFETIMECYCLES_COLUMN_NAME = "LifetimeCycles"

MAXCAP_COLUMN_NAME = "MaxCap"
MINCAP_COLUMN_NAME = "MinCap"

DEPTHOFDISCHARGE_COLUMN_NAME = "DepthOfDischarge"

EFFICIENCY_COLUMN_NAME = "Efficiency"

BALLIFETIME_COLUMN_NAME = "BalLifetime"
BALCYCLES_COLUMN_NAME = "BalCycles"

TRANSITCOST_COLUMN_NAME = "TransitCost"
TRANSITLOSS_COLUMN_NAME = "TransitLoss"
MAXTRANSIT_COLUMN_NAME = "MaxTransit"

ENDUSEDEMANDENERGY_COLUMN_NAME = "EndUseDemandEnergy"
ENDUSEDEMAND_COLUMN_NAME = "EndUseDemand"

time_columns_dict = \
{
    1: [YEAR_COLUMN_NAME],
    2: [YEAR_COLUMN_NAME, SEASON_COLUMN_NAME],
    3: [YEAR_COLUMN_NAME, SEASON_COLUMN_NAME, DAYTYPE_COLUMN_NAME],
    4: [YEAR_COLUMN_NAME, SEASON_COLUMN_NAME, DAYTYPE_COLUMN_NAME, 
        DAYSLICE_COLUMN_NAME]
}

geog_columns_dict = \
{
    1: [MODELGEOGRAPHY_COLUMN_NAME],
    2: [MODELGEOGRAPHY_COLUMN_NAME, SUBGEOGRAPHY1_COLUMN_NAME],
    3: [MODELGEOGRAPHY_COLUMN_NAME, SUBGEOGRAPHY1_COLUMN_NAME, 
        SUBGEOGRAPHY2_COLUMN_NAME],
    4: [MODELGEOGRAPHY_COLUMN_NAME, SUBGEOGRAPHY1_COLUMN_NAME, 
        SUBGEOGRAPHY2_COLUMN_NAME, SUBGEOGRAPHY3_COLUMN_NAME]
}

src_geog_columns_dict = \
{
    1: [SRCMODELGEOGRAPHY_COLUMN_NAME],
    2: [SRCMODELGEOGRAPHY_COLUMN_NAME, SRCSUBGEOGRAPHY1_COLUMN_NAME],
    3: [SRCMODELGEOGRAPHY_COLUMN_NAME, SRCSUBGEOGRAPHY1_COLUMN_NAME, 
        SRCSUBGEOGRAPHY2_COLUMN_NAME],
    4: [SRCMODELGEOGRAPHY_COLUMN_NAME, SRCSUBGEOGRAPHY1_COLUMN_NAME, 
        SRCSUBGEOGRAPHY2_COLUMN_NAME, SRCSUBGEOGRAPHY3_COLUMN_NAME]
}

dest_geog_columns_dict = \
{
    1: [DESTMODELGEOGRAPHY_COLUMN_NAME],
    2: [DESTMODELGEOGRAPHY_COLUMN_NAME, DESTSUBGEOGRAPHY1_COLUMN_NAME],
    3: [DESTMODELGEOGRAPHY_COLUMN_NAME, DESTSUBGEOGRAPHY1_COLUMN_NAME, 
        DESTSUBGEOGRAPHY2_COLUMN_NAME],
    4: [DESTMODELGEOGRAPHY_COLUMN_NAME, DESTSUBGEOGRAPHY1_COLUMN_NAME, 
        DESTSUBGEOGRAPHY2_COLUMN_NAME, DESTSUBGEOGRAPHY3_COLUMN_NAME]
}

bal_time_str_level_map = \
{
    BALTIME_YR_STR: 1,
    BALTIME_SE_STR: 2,
    BALTIME_DT_STR: 3,
    BALTIME_DS_STR: 4
}

bal_area_str_level_map = \
{
    BALAREA_MG_STR: 1,
    BALAREA_SG1_STR: 2,
    BALAREA_SG2_STR: 3,
    BALAREA_SG3_STR: 4
}

days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]    # month indices start from 0

def get_finest_bal_time(m):
    bal_time_values = set(m.ec_bal_time_map.values())

    if BALTIME_DS_STR in bal_time_values: return BALTIME_DS_STR
    if BALTIME_DT_STR in bal_time_values: return BALTIME_DT_STR
    if BALTIME_SE_STR in bal_time_values: return BALTIME_SE_STR
    return BALTIME_YR_STR

def get_finest_bal_area(m):
    bal_area_values = set(m.ec_bal_area_map.values())

    if BALAREA_SG3_STR in bal_area_values: return BALAREA_SG3_STR
    if BALAREA_SG2_STR in bal_area_values: return BALAREA_SG2_STR
    if BALAREA_SG1_STR in bal_area_values: return BALAREA_SG1_STR
    return BALAREA_MG_STR

def calc_duration_of_seasons(df_se):
    cumul_days_by_month = [0]
    for i in range(1, MONTHS_PER_YEAR):
        cumul_days_by_month.append(cumul_days_by_month[i - 1] + \
                                   days_per_month[i - 1])

    df_se[STARTDAYOFYEAR_COLUMN_NAME] = \
        (df_se[STARTMONTH_COLUMN_NAME] - 1).map(lambda x: cumul_days_by_month[x]) + \
        df_se[STARTDATE_COLUMN_NAME]

    df_se[ENDDAYOFYEAR_COLUMN_NAME] = \
        np.roll(df_se[STARTDAYOFYEAR_COLUMN_NAME], - 1)

    df_se[DURATION_COLUMN_NAME] = \
        (df_se[ENDDAYOFYEAR_COLUMN_NAME] - df_se[STARTDAYOFYEAR_COLUMN_NAME]) % \
        DAYS_PER_YEAR

def calc_duration_of_dayslices(df_ds):
    df_ds[ENDHOUR_COLUMN_NAME] = np.roll(df_ds[STARTHOUR_COLUMN_NAME], - 1)
    df_ds[DURATION_COLUMN_NAME] = (df_ds[ENDHOUR_COLUMN_NAME] - 
                                   df_ds[STARTHOUR_COLUMN_NAME]) % HOURS_PER_DAY

def get_param(param_name, param_type):
    if (param_type == CONFIG_PARAM):
        return loaders.get_config_parameter(param_name)
    if (param_type == COMMON_PARAM):
        return loaders.get_parameter(param_name)
    if (param_type == SUPPLY_PARAM):
        return supply.get_filtered_parameter(param_name)

def get_time_params(m):
    m.model_period_data: pd.DataFrame = get_param(MODEL_PERIOD_PARAM_NAME,
                                                  COMMON_PARAM)
    m.StartYear = m.model_period_data[STARTYEAR_COLUMN_NAME][0]
    m.EndYear = m.model_period_data[ENDYEAR_COLUMN_NAME][0]

    m.num_time_levels: int = 1

    m.seasons_data: pd.DataFrame = get_param(SEASONS_PARAM_NAME, COMMON_PARAM)
    if (m.seasons_data is None):
        return

    calc_duration_of_seasons(m.seasons_data)
    m.num_time_levels += 1

    m.daytypes_data: pd.DataFrame = get_param(DAYTYPES_PARAM_NAME, COMMON_PARAM)
    if (m.daytypes_data is None):
        return

    m.num_time_levels += 1

    m.dayslices_data: pd.DataFrame = get_param(DAYSLICES_PARAM_NAME, COMMON_PARAM)
    if (m.dayslices_data is None):
        return

    calc_duration_of_dayslices(m.dayslices_data)
    m.num_time_levels += 1

def get_geography_params(m):
    m.model_geog: str = get_param(MODEL_GEOG_PARAM_NAME, COMMON_PARAM)
    m.num_geography_levels: int = 1

    m.subgeog1_data: List[str] = get_param(SUBGEOG1_PARAM_NAME, COMMON_PARAM)
    if ((m.subgeog1_data is None) or (not m.subgeog1_data)):
        return

    m.num_geography_levels +=1

    m.subgeog2_data: Dict[str: List[str]] = get_param(SUBGEOG2_PARAM_NAME, 
                                                      COMMON_PARAM)
    if ((m.subgeog2_data is None) or (not m.subgeog2_data)):
        return

    m.num_geography_levels +=1

    m.subgeog3_data: Dict[str: List[str]] = get_param(SUBGEOG3_PARAM_NAME, 
                                                      COMMON_PARAM)
    if ((m.subgeog3_data is None) or (not m.subgeog3_data)):
        return

    m.num_geography_levels +=1

def get_common_params_data(m):
    get_time_params(m)
    get_geography_params(m)

    m.ppec_data: pd.DataFrame = get_param(PPEC_PARAM_NAME, COMMON_PARAM)
    m.nppec_data: pd.DataFrame = get_param(NPPEC_PARAM_NAME, COMMON_PARAM)
    m.pdec_data: pd.DataFrame = get_param(PDEC_PARAM_NAME, COMMON_PARAM)
    m.npdec_data: pd.DataFrame = get_param(NPDEC_PARAM_NAME, COMMON_PARAM)
    m.ppec_energy_density_data: pd.DataFrame = \
                    get_param(PPEC_ENERGYDENSITY_PARAM_NAME, COMMON_PARAM)
    m.pdec_energy_density_data: pd.DataFrame = \
                    get_param(PDEC_ENERGYDENSITY_PARAM_NAME, COMMON_PARAM)
    m.unmet_demand_value_data: pd.DataFrame = \
                    get_param(UNMET_DEMAND_VALUE_PARAM_NAME, COMMON_PARAM)

    m.seasons_data_dict = m.seasons_data.set_index(SEASON_COLUMN_NAME).to_dict()
    m.daytypes_data_dict = m.daytypes_data.set_index(DAYTYPE_COLUMN_NAME).to_dict()
    m.dayslices_data_dict = m.dayslices_data.set_index(DAYSLICE_COLUMN_NAME).to_dict()
    
    m.ppec_data_dict = m.ppec_data.set_index(ENERGYCARRIER_COLUMN_NAME).to_dict()
    m.pdec_data_dict = m.pdec_data.set_index(ENERGYCARRIER_COLUMN_NAME).to_dict()
    m.npdec_data_dict = m.npdec_data.set_index(ENERGYCARRIER_COLUMN_NAME).to_dict()
    m.ppec_energy_density_data_dict = m.ppec_energy_density_data.set_index(
                                                [ENERGYCARRIER_COLUMN_NAME, 
                                                 YEAR_COLUMN_NAME]).to_dict()
    m.pdec_energy_density_data_dict = m.pdec_energy_density_data.set_index(
                                                [ENERGYCARRIER_COLUMN_NAME, 
                                                 YEAR_COLUMN_NAME]).to_dict()
    m.unmet_demand_value_data_dict = m.unmet_demand_value_data.set_index(
                                                [ENERGYCARRIER_COLUMN_NAME, 
                                                 YEAR_COLUMN_NAME]).to_dict()

    m.ect_data: pd.DataFrame = get_param(ECT_PARAM_NAME, SUPPLY_PARAM)
    m.ect_data_dict = m.ect_data.set_index(ENERGYCONVTECH_COLUMN_NAME).to_dict()

    m.est_data: pd.DataFrame = get_param(EST_PARAM_NAME, SUPPLY_PARAM)
    m.est_data_dict = m.est_data.set_index(ENERGYSTORTECH_COLUMN_NAME).to_dict()

    m.ec_bal_time_map: Dict[str: str] = \
    {
        **m.ppec_data_dict[BALANCINGTIME_COLUMN_NAME],
        **m.pdec_data_dict[BALANCINGTIME_COLUMN_NAME],
        **m.npdec_data_dict[BALANCINGTIME_COLUMN_NAME]
    }
    
    m.ec_bal_area_map: Dict[str: str] = \
    {
        **m.ppec_data_dict[BALANCINGAREA_COLUMN_NAME],
        **m.pdec_data_dict[BALANCINGAREA_COLUMN_NAME],
        **m.npdec_data_dict[BALANCINGAREA_COLUMN_NAME]
    }

    m.ec_energy_unit_map: Dict[str: str] = \
    {
        **m.ppec_data_dict[ENERGYUNIT_COLUMN_NAME],
        **m.pdec_data_dict[ENERGYUNIT_COLUMN_NAME],
        **m.npdec_data_dict[ENERGYUNIT_COLUMN_NAME]
    }
    
    m.num_time_levels_to_use = bal_time_str_level_map.get(get_finest_bal_time(m))
    m.num_geography_levels_to_use = bal_area_str_level_map.get(get_finest_bal_area(m))

    m.time_columns_list: List[str] = time_columns_dict.get(m.num_time_levels_to_use)
    m.geog_columns_list: List[str] = geog_columns_dict.get(m.num_geography_levels_to_use)

    m.src_geog_columns_list: List[str] = \
                       src_geog_columns_dict.get(m.num_geography_levels_to_use)
    m.dest_geog_columns_list: List[str] = \
                       dest_geog_columns_dict.get(m.num_geography_levels_to_use)

    m.eu_conv_data: pd.DataFrame = get_param(ENERGYUNITCONV_PARAM_NAME, 
                                             CONFIG_PARAM)

    m.eu_data: pd.DataFrame = m.eu_conv_data[[m.eu_conv_data.columns[0]]]
    m.eu_conv_data = m.eu_conv_data.melt(id_vars = m.eu_conv_data.columns[0], 
                                         var_name = ENERGYUNIT2_COLUMN_NAME, 
                                         value_name = ENERGYUNITCONV_COLUMN_NAME)
    m.eu_conv_data.rename(columns = {m.eu_conv_data.columns[0] : 
                                     ENERGYUNIT1_COLUMN_NAME}, inplace = True)

    m.eu_conv_data_dict = m.eu_conv_data.set_index([ENERGYUNIT1_COLUMN_NAME, 
                                                    ENERGYUNIT2_COLUMN_NAME]).to_dict()

def print_common_params_data(m):
    print(m.model_period_data)
    print(m.seasons_data)
    print(m.daytypes_data)
    print(m.dayslices_data)

    print(m.model_geog)
    print(m.subgeog1_data)
    print(m.subgeog2_data)
    print(m.subgeog3_data)

    print(m.ppec_data)
    print(m.nppec_data)
    print(m.pdec_data)
    print(m.npdec_data)
    print(m.ppec_energy_density_data)
    print(m.pdec_energy_density_data)
    print(m.unmet_demand_value_data)
    print(m.ect_data)
    print(m.est_data)

    print(m.ec_bal_time_map)
    print(m.ec_bal_area_map)
    
    print(m.num_time_levels, m.num_geography_levels)
    print(m.num_time_levels_to_use, m.num_geography_levels_to_use)

    print(m.time_columns_list)
    print(m.geog_columns_list)
    print(m.src_geog_columns_list)
    print(m.dest_geog_columns_list)

    print(m.eu_data)
    print(m.eu_conv_data)

def get_supply_params_data(m):
    m.ec_time_geog_column_list = [ENERGYCARRIER_COLUMN_NAME] + \
                                 m.time_columns_list + \
                                 m.geog_columns_list

    m.ect_year_geog_column_list = [ENERGYCONVTECH_COLUMN_NAME] + \
                                  [YEAR_COLUMN_NAME] + \
                                  m.geog_columns_list

    m.ect_year_column_list = [ENERGYCONVTECH_COLUMN_NAME] + [YEAR_COLUMN_NAME]

    m.ect_inst_year_column_list = [ENERGYCONVTECH_COLUMN_NAME] + \
                                  [INSTYEAR_COLUMN_NAME]

    m.inst_year_ect_year_geog_column_list = [INSTYEAR_COLUMN_NAME] + \
                                            [ENERGYCONVTECH_COLUMN_NAME] + \
                                            [YEAR_COLUMN_NAME] + \
                                            m.geog_columns_list

    m.inst_year_ect_time_geog_column_list = [INSTYEAR_COLUMN_NAME] + \
                                            [ENERGYCONVTECH_COLUMN_NAME] + \
                                            m.time_columns_list + \
                                            m.geog_columns_list

    m.ect_geog_column_list = [ENERGYCONVTECH_COLUMN_NAME] + \
                             m.geog_columns_list

    m.est_year_column_list = [ENERGYSTORTECH_COLUMN_NAME] + [YEAR_COLUMN_NAME]

    m.est_inst_year_column_list = [ENERGYSTORTECH_COLUMN_NAME] + \
                                  [INSTYEAR_COLUMN_NAME]

    m.est_year_geog_column_list = [ENERGYSTORTECH_COLUMN_NAME] + \
                                  [YEAR_COLUMN_NAME] + \
                                  m.geog_columns_list

    m.inst_year_est_year_geog_column_list = [INSTYEAR_COLUMN_NAME] + \
                                            [ENERGYSTORTECH_COLUMN_NAME] + \
                                            [YEAR_COLUMN_NAME] + \
                                            m.geog_columns_list

    m.est_geog_column_list = [ENERGYSTORTECH_COLUMN_NAME] + \
                             m.geog_columns_list

    m.ec_yr_geog_geog_column_list = [ENERGYCARRIER_COLUMN_NAME] + \
                                    [YEAR_COLUMN_NAME] + \
                                    m.src_geog_columns_list + \
                                    m.dest_geog_columns_list

    m.pec_info: pd.DataFrame = get_param(PEC_INFO_PARAM_NAME, SUPPLY_PARAM)
    m.pec_info_dict = m.pec_info.set_index(m.ec_time_geog_column_list).to_dict()

    m.pec_prod_imp_constraints: pd.DataFrame = \
        get_param(PEC_PRODIMPCONSTRAINTS_PARAM_NAME, SUPPLY_PARAM)
    m.pec_prod_imp_constraints_dict = \
        m.pec_prod_imp_constraints.set_index(m.ec_time_geog_column_list).to_dict()

    m.dec_taxation: pd.DataFrame = \
        get_param(DEC_TAXATION_PARAM_NAME, SUPPLY_PARAM)
    m.dec_taxation_dict = \
        m.dec_taxation.set_index(m.ec_time_geog_column_list).to_dict()
    
    m.ect_cap_add_bounds: pd.DataFrame = \
        get_param(ECT_CAPADDBOUNDS_PARAM_NAME, SUPPLY_PARAM)
    m.ect_cap_add_bounds_dict = \
        m.ect_cap_add_bounds.set_index(m.ect_year_geog_column_list).to_dict()

    m.ect_lifetime: pd.DataFrame = \
        get_param(ECT_LIFETIME_PARAM_NAME, SUPPLY_PARAM)
    m.ect_lifetime_dict = \
        m.ect_lifetime.set_index(m.ect_year_column_list).to_dict()
    
    m.ect_operational_info: pd.DataFrame = \
        get_param(ECT_OPERATIONALINFO_PARAM_NAME, SUPPLY_PARAM)
    m.ect_operational_info_dict = \
        m.ect_operational_info.set_index(m.ect_inst_year_column_list).to_dict()
    
    m.ect_efficiency_cost_max_annual_uf: pd.DataFrame = \
        get_param(ECT_EFFICIENCYCOSTMAXANNUALUF_PARAM_NAME, SUPPLY_PARAM)
    m.ect_efficiency_cost_max_annual_uf_dict = \
        m.ect_efficiency_cost_max_annual_uf.set_index(m.inst_year_ect_year_geog_column_list).to_dict()
    
    m.ect_max_cuf: pd.DataFrame = get_param(ECT_MAXCUF_PARAM_NAME, SUPPLY_PARAM)
    m.ect_max_cuf_dict = \
        m.ect_max_cuf.set_index(m.inst_year_ect_time_geog_column_list).to_dict()
    
    m.ect_legacy_capacity: pd.DataFrame = \
        get_param(ECT_LEGACYCAPACITY_PARAM_NAME, SUPPLY_PARAM)
    m.ect_legacy_capacity_dict = \
        m.ect_legacy_capacity.set_index(m.ect_geog_column_list).to_dict()
    
    m.ect_legacy_retirement: pd.DataFrame = \
        get_param(ECT_LEGACYRETIREMENT_PARAM_NAME, SUPPLY_PARAM)
    m.ect_legacy_retirement_dict = \
        m.ect_legacy_retirement.set_index(m.ect_year_geog_column_list).to_dict()
    
    m.est_lifetime: pd.DataFrame = get_param(EST_LIFETIME_PARAM_NAME, SUPPLY_PARAM)
    m.est_lifetime_dict = m.est_lifetime.set_index(m.est_year_column_list).to_dict()

    m.est_cap_add_bounds: pd.DataFrame = \
        get_param(EST_CAPADDBOUNDS_PARAM_NAME, SUPPLY_PARAM)
    m.est_cap_add_bounds_dict = \
        m.est_cap_add_bounds.set_index(m.est_year_geog_column_list).to_dict()

    m.est_derating_depth_of_discharge: pd.DataFrame = \
        get_param(EST_DERATINGDEPTHOFDISCHARGE_PARAM_NAME, SUPPLY_PARAM)
    m.est_derating_depth_of_discharge_dict = \
        m.est_derating_depth_of_discharge.set_index(m.est_inst_year_column_list).to_dict()

    m.est_efficiency_cost: pd.DataFrame = \
        get_param(EST_EFFICIENCYCOST_PARAM_NAME, SUPPLY_PARAM)
    m.est_efficiency_cost_dict = \
        m.est_efficiency_cost.set_index(m.inst_year_est_year_geog_column_list).to_dict()

    m.est_legacy_details: pd.DataFrame = \
        get_param(EST_LEGACYDETAILS_PARAM_NAME, SUPPLY_PARAM)
    m.est_legacy_details_dict = \
        m.est_legacy_details.set_index(m.est_geog_column_list).to_dict()
    
    m.ec_transfers: pd.DataFrame = get_param(EC_TRANSFERS_PARAM_NAME, SUPPLY_PARAM)
    m.ec_transfers_dict = \
        m.ec_transfers.set_index(m.ec_yr_geog_geog_column_list).to_dict()
    
    m.end_use_demand_energy: pd.DataFrame = \
        get_param(ENDUSE_DEMANDENERGY_PARAM_NAME, SUPPLY_PARAM)
    
    m.ec_demand_conv_map: Dict[(str, int): float] = \
    {
        **m.ppec_energy_density_data_dict[DOMENERGYDENSITY_COLUMN_NAME],
        **m.pdec_energy_density_data_dict[ENERGYDENSITY_COLUMN_NAME],
        **dict.fromkeys([(ec, yr)
                         for ec in list(m.npdec_data[ENERGYCARRIER_COLUMN_NAME])
                         for yr in list(range(m.StartYear, m.EndYear + 1))
                        ], 1)
    }

    m.end_use_demand_energy[ENDUSEDEMAND_COLUMN_NAME] = \
        m.end_use_demand_energy[ENDUSEDEMANDENERGY_COLUMN_NAME] / \
        pd.MultiIndex.from_frame(
            m.end_use_demand_energy[[ENERGYCARRIER_COLUMN_NAME, YEAR_COLUMN_NAME]]).map(m.ec_demand_conv_map)

    m.end_use_demand_energy_dict = \
        m.end_use_demand_energy.set_index(m.ec_time_geog_column_list).to_dict()

def print_supply_params_data(m):
    print(m.ec_time_geog_column_list)
    print(m.pec_info)
    print(m.pec_prod_imp_constraints)
    print(m.dec_taxation)
    print(m.ect_cap_add_bounds)
    print(m.ect_lifetime)
    print(m.ect_operational_info)
    print(m.ect_efficiency_cost_max_annual_uf)
    print(m.ect_max_cuf)
    print(m.ect_legacy_capacity)
    print(m.ect_legacy_retirement)
    print(m.est_lifetime)
    print(m.est_cap_add_bounds)
    print(m.est_derating_depth_of_discharge)
    print(m.est_efficiency_cost)
    print(m.est_legacy_details)
    print(m.ec_transfers)
    print(m.ec_demand_conv_map)
    print(m.end_use_demand_energy)

def delete_supply_params_data(m):
    del m.pec_info, m.pec_info_dict
    del m.pec_prod_imp_constraints, m.pec_prod_imp_constraints_dict
    del m.dec_taxation, m.dec_taxation_dict
    del m.ect_cap_add_bounds, m.ect_cap_add_bounds_dict
    del m.ect_lifetime, m.ect_lifetime_dict
    del m.ect_operational_info, m.ect_operational_info_dict
    del m.ect_efficiency_cost_max_annual_uf, m.ect_efficiency_cost_max_annual_uf_dict
    del m.ect_max_cuf, m.ect_max_cuf_dict
    del m.ect_legacy_capacity, m.ect_legacy_capacity_dict
    del m.ect_legacy_retirement, m.ect_legacy_retirement_dict
    del m.est_lifetime, m.est_lifetime_dict
    del m.est_cap_add_bounds, m.est_cap_add_bounds_dict
    del m.est_derating_depth_of_discharge, m.est_derating_depth_of_discharge_dict
    del m.est_efficiency_cost, m.est_efficiency_cost_dict
    del m.est_legacy_details, m.est_legacy_details_dict
    del m.ec_transfers, m.ec_transfers_dict
    del m.end_use_demand_energy, m.end_use_demand_energy_dict


                         #########################################
#####################            Model Definition                #############
                         #########################################

from pyomo.environ import *
from pyomo.core import *
from pyomo.opt import SolverFactory

model = AbstractModel()

get_common_params_data(model)


###############
#    Sets     #
###############

#########                   Energy Units                         #############
def get_energy_units_list(m):
    return list(m.eu_data[ENERGYUNIT_COLUMN_NAME])

model.EU = Set(initialize = get_energy_units_list, ordered = Set.SortedOrder)

#########                   Time and Geography                     #############

def get_year_range(m):
    return list(range(m.StartYear, m.EndYear + 1))

def get_install_year_range(m):
    return list(range(m.StartYear - 1, m.EndYear + 1))

def get_season_names_list(m):
    if (m.num_time_levels_to_use > 1):
        return list(m.seasons_data[SEASON_COLUMN_NAME])
    else:
        return [DUMMY_TIME_STR]

def get_daytype_names_list(m):
    if (m.num_time_levels_to_use > 2):
        return list(m.daytypes_data[DAYTYPE_COLUMN_NAME])
    else:
        return [DUMMY_TIME_STR]

def get_dayslice_names_list(m):
    if (m.num_time_levels_to_use > 3):
        return list(m.dayslices_data[DAYSLICE_COLUMN_NAME])
    else:
        return [DUMMY_TIME_STR]

def get_model_geog_name(m):
    return [m.model_geog]

def get_subgeog1_names_list(m):
    if (m.num_geography_levels_to_use > 1):
        return m.subgeog1_data
    else:
        return [DUMMY_GEOG_STR]

def get_subgeog2_dict_by_key(m, key: str):
    if (m.num_geography_levels_to_use > 2):
        return m.subgeog2_data[key]
    else:
        return [DUMMY_GEOG_STR]

def get_subgeog2_all_values_list(m):
    if (m.num_geography_levels_to_use > 2):
        return [elem for sublist in m.subgeog2_data.values() for elem in sublist]
    else:
        return [DUMMY_GEOG_STR]

def get_subgeog3_dict_by_key(m, key: str):
    if (m.num_geography_levels_to_use > 3):
        return m.subgeog3_data[key]
    else:
        return [DUMMY_GEOG_STR]

def get_subgeog3_all_values_list(m):
    if (m.num_geography_levels_to_use > 3):
        return [elem for sublist in m.subgeog3_data.values() for elem in sublist]
    else:
        return [DUMMY_GEOG_STR]

def init_bal_area_sg2(m):
    if (m.num_geography_levels_to_use == 3):
        return ((mg, sg1, sg2)
                for mg in m.ModelGeography for sg1 in m.SubGeog1
                for sg2 in m.SubGeog2Map[sg1])
    elif (m.num_geography_levels_to_use == 4):
        return ((mg, sg1, sg2, dm)
                for mg in m.ModelGeography for sg1 in m.SubGeog1
                for sg2 in m.SubGeog2Map[sg1] for dm in m.DummyGeog)
    else:
        return None

def init_bal_area_sg3(m):
    return ((mg, sg1, sg2, sg3)
            for mg in m.ModelGeography for sg1 in m.SubGeog1
            for sg2 in m.SubGeog2Map[sg1] for sg3 in m.SubGeog3Map[sg2])

def define_bal_time_sets_for_level1(m):
    return Set(within = m.Year, initialize = m.Year, ordered = True)

def define_bal_time_sets_for_level2(m):
    return Set(within = m.Year * m.DummyTime,
               initialize = m.Year * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonInp,
               initialize = m.Year * m.SeasonInp,
               ordered = True)

def define_bal_time_sets_for_level3(m):
    return Set(within = m.Year * m.DummyTime * m.DummyTime,
               initialize = m.Year * m.DummyTime * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonInp * m.DummyTime,
               initialize = m.Year * m.SeasonInp * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonInp * m.DayTypeInp,
               initialize = m.Year * m.SeasonInp * m.DayTypeInp,
               ordered = True)

def define_bal_time_sets_for_level4(m):
    return Set(within = m.Year * m.DummyTime * m.DummyTime * m.DummyTime,
               initialize = m.Year * m.DummyTime * m.DummyTime * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonInp * m.DummyTime * m.DummyTime,
               initialize = m.Year * m.SeasonInp * m.DummyTime * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonInp * m.DayTypeInp * m.DummyTime,
               initialize = m.Year * m.SeasonInp * m.DayTypeInp * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonInp * m.DayTypeInp * m.DaySliceInp,
               initialize = m.Year * m.SeasonInp * m.DayTypeInp * m.DaySliceInp,
               ordered = True)

def define_bal_area_sets_for_level1(m):
    return Set(within = m.ModelGeography, initialize = m.ModelGeography,
               ordered = True)

def define_bal_area_sets_for_level2(m):
    return Set(within = m.ModelGeography * m.DummyGeog,
               initialize = m.ModelGeography * m.DummyGeog,
               ordered = True), \
           Set(within = m.ModelGeography * m.SubGeog1,
               initialize = m.ModelGeography * m.SubGeog1,
               ordered = True)

def define_bal_area_sets_for_level3(m):
    return Set(within = m.ModelGeography * m.DummyGeog * m.DummyGeog,
               initialize = m.ModelGeography * m.DummyGeog * m.DummyGeog,
               ordered = True), \
           Set(within = m.ModelGeography * m.SubGeog1 * m.DummyGeog,
               initialize = m.ModelGeography * m.SubGeog1 * m.DummyGeog,
               ordered = True), \
           Set(within = m.ModelGeography * m.SubGeog1 * m.SubGeog2AllValues,
               initialize = init_bal_area_sg2,
               ordered = True)

def define_bal_area_sets_for_level4(m):
    return Set(within = m.ModelGeography * m.DummyGeog * 
                        m.DummyGeog * m.DummyGeog,
               initialize = m.ModelGeography * m.DummyGeog * 
                            m.DummyGeog * m.DummyGeog,
               ordered = True), \
           Set(within = m.ModelGeography * m.SubGeog1 * 
                        m.DummyGeog * m.DummyGeog,
               initialize = m.ModelGeography * m.SubGeog1 * 
                            m.DummyGeog * m.DummyGeog,
               ordered = True), \
           Set(within = m.ModelGeography * m.SubGeog1 * 
                        m.SubGeog2AllValues * m.DummyGeog,
               initialize = init_bal_area_sg2,
               ordered = True), \
           Set(within = m.ModelGeography * m.SubGeog1 * 
                        m.SubGeog2AllValues * m.SubGeog3AllValues,
               initialize = init_bal_area_sg3,
               ordered = True)


model.Year = Set(initialize = get_year_range, ordered = Set.SortedOrder)
if (model.num_time_levels_to_use > 1):
    model.SeasonInp = Set(initialize = get_season_names_list, ordered = True)
if (model.num_time_levels_to_use > 2):
    model.DayTypeInp = Set(initialize = get_daytype_names_list, ordered = True)
if (model.num_time_levels_to_use > 3):
    model.DaySliceInp = Set(initialize = get_dayslice_names_list, ordered = True)

model.ModelGeography = Set(initialize = get_model_geog_name, ordered = True)
if (model.num_geography_levels_to_use > 1):
    model.SubGeog1 = Set(initialize = get_subgeog1_names_list, ordered = True)
if (model.num_geography_levels_to_use > 2):
    model.SubGeog2Map = Set(model.SubGeog1,
                            initialize = get_subgeog2_dict_by_key,
                            ordered = True)
    model.SubGeog2AllValues = Set(initialize = get_subgeog2_all_values_list,
                                  ordered = True)
if (model.num_geography_levels_to_use > 3):
    model.SubGeog3Map = Set(model.SubGeog2AllValues,
                            initialize = get_subgeog3_dict_by_key,
                            ordered = True)
    model.SubGeog3AllValues = Set(initialize = get_subgeog3_all_values_list,
                                  ordered = True)

model.TimeLevel = Set(initialize = RangeSet(model.num_time_levels_to_use),
                                            ordered = Set.SortedOrder)
model.GeogLevel = Set(initialize = RangeSet(model.num_geography_levels_to_use),
                                            ordered = Set.SortedOrder)

model.DummyTime = Set(initialize = [DUMMY_TIME_STR])
model.DummyGeog = Set(initialize = [DUMMY_GEOG_STR])

model.InstYear = Set(initialize = get_install_year_range, 
                     ordered = Set.SortedOrder)
model.InstYearECT = Set(initialize = get_install_year_range, 
                        ordered = Set.SortedOrder)
model.InstYearEST = Set(initialize = get_install_year_range, 
                        ordered = Set.SortedOrder)

if (model.num_time_levels_to_use == 1):
    model.BalTimeYr = define_bal_time_sets_for_level1(model)
elif (model.num_time_levels_to_use == 2):
    model.BalTimeYr, model.BalTimeSe = define_bal_time_sets_for_level2(model)
elif (model.num_time_levels_to_use == 3):
    model.BalTimeYr, model.BalTimeSe, model.BalTimeDT = \
                                      define_bal_time_sets_for_level3(model)
elif (model.num_time_levels_to_use == 4):
    model.BalTimeYr, model.BalTimeSe, model.BalTimeDT, model.BalTimeDS = \
                                      define_bal_time_sets_for_level4(model)

if (model.num_geography_levels_to_use == 1):
    model.BalAreaMG = define_bal_area_sets_for_level1(model)
elif (model.num_geography_levels_to_use == 2):
    model.BalAreaMG, model.BalAreaSG1 = define_bal_area_sets_for_level2(model)
elif (model.num_geography_levels_to_use == 3):
    model.BalAreaMG, model.BalAreaSG1, model.BalAreaSG2 = \
                                       define_bal_area_sets_for_level3(model)
elif (model.num_geography_levels_to_use == 4):
    model.BalAreaMG, model.BalAreaSG1, model.BalAreaSG2, model.BalAreaSG3 = \
                                       define_bal_area_sets_for_level4(model)

if (model.num_time_levels_to_use > 1):
    model.Season = model.SeasonInp | model.DummyTime
if (model.num_time_levels_to_use > 2):
    model.DayType = model.DayTypeInp | model.DummyTime
if (model.num_time_levels_to_use > 3):
    model.DaySlice = model.DaySliceInp | model.DummyTime

if (model.num_geography_levels_to_use > 1):
    model.SubGeography1 = model.SubGeog1 | model.DummyGeog
if (model.num_geography_levels_to_use > 2):
    model.SubGeography2 = model.SubGeog2AllValues | model.DummyGeog
if (model.num_geography_levels_to_use > 3):
    model.SubGeography3 = model.SubGeog3AllValues | model.DummyGeog

if (model.num_time_levels_to_use == 1):
    model.BalTime = Set(within = model.Year, initialize = model.BalTimeYr,
                        ordered = True)
elif (model.num_time_levels_to_use == 2):
    model.BalTime = Set(within = model.Year * model.Season,
                        initialize = model.BalTimeYr | model.BalTimeSe,
                        ordered = True)
elif (model.num_time_levels_to_use == 3):
    model.BalTime = Set(within = model.Year * model.Season * model.DayType,
                        initialize = model.BalTimeYr | model.BalTimeSe | 
                                     model.BalTimeDT,
                        ordered = True)
elif (model.num_time_levels_to_use == 4):
    model.BalTime = Set(within = model.Year * model.Season * 
                                 model.DayType * model.DaySlice,
                        initialize = model.BalTimeYr | model.BalTimeSe | 
                                     model.BalTimeDT | model.BalTimeDS,
                        ordered = True)

if (model.num_geography_levels_to_use == 1):
    model.BalArea = Set(within = model.ModelGeography,
                        initialize = model.BalAreaMG, ordered = True)
elif (model.num_geography_levels_to_use == 2):
    model.BalArea = Set(within = model.ModelGeography * model.SubGeography1,
                        initialize = model.BalAreaMG | model.BalAreaSG1,
                        ordered = True)
elif (model.num_geography_levels_to_use == 3):
    model.BalArea = Set(within = model.ModelGeography * model.SubGeography1 * 
                                 model.SubGeography2,
                        initialize = model.BalAreaMG | model.BalAreaSG1 | 
                                     model.BalAreaSG2,
                        ordered = True)
elif (model.num_geography_levels_to_use == 4):
    model.BalArea = Set(within = model.ModelGeography * model.SubGeography1 *
                                 model.SubGeography2 * model.SubGeography3,
                        initialize = model.BalAreaMG | model.BalAreaSG1 | 
                                     model.BalAreaSG2 | model.BalAreaSG3,
                        ordered = True)

#########                   Energy Carriers                        #############

def get_nppec_names_list(m):
    return list(m.nppec_data[ENERGYCARRIER_COLUMN_NAME])

def get_ppec_names_list(m):
    return list(m.ppec_data[ENERGYCARRIER_COLUMN_NAME])

def get_npdec_names_list(m):
    return list(m.npdec_data[ENERGYCARRIER_COLUMN_NAME])

def get_pdec_names_list(m):
    return list(m.pdec_data[ENERGYCARRIER_COLUMN_NAME])

def is_primary(ec):
    return ec in get_nppec_names_list(model) or ec in get_ppec_names_list(model)

def is_derived(ec):
    return ec in get_npdec_names_list(model) or ec in get_pdec_names_list(model)

def is_physical(ec):
    return ec in get_ppec_names_list(model) or ec in get_pdec_names_list(model)

def is_non_physical(ec):
    return ec in get_nppec_names_list(model) or ec in get_npdec_names_list(model)

def get_bal_time_inp(ec):
    return model.ec_bal_time_map.get(ec)

def get_bal_area_inp(ec):
    return model.ec_bal_area_map.get(ec)

def get_bal_time_set(m, ec):
    bal_time_inp = get_bal_time_inp(ec)
    if (bal_time_inp == BALTIME_YR_STR):
        return m.BalTimeYr
    if (bal_time_inp == BALTIME_SE_STR):
        return m.BalTimeSe
    if (bal_time_inp == BALTIME_DT_STR):
        return m.BalTimeDT
    if (bal_time_inp == BALTIME_DS_STR):
        return m.BalTimeDS

def get_bal_area_set(m, ec):
    bal_area_inp = get_bal_area_inp(ec)
    if (bal_area_inp == BALAREA_MG_STR):
        return m.BalAreaMG
    if (bal_area_inp == BALAREA_SG1_STR):
        return m.BalAreaSG1
    if (bal_area_inp == BALAREA_SG2_STR):
        return m.BalAreaSG2
    if (bal_area_inp == BALAREA_SG3_STR):
        return m.BalAreaSG3

def get_upto_bal_time_set(m, ec):
    bal_time_inp = get_bal_time_inp(ec)
    if (bal_time_inp == BALTIME_YR_STR):
        return m.BalTimeYr
    if (bal_time_inp == BALTIME_SE_STR):
        return m.BalTimeYr | m.BalTimeSe
    if (bal_time_inp == BALTIME_DT_STR):
        return m.BalTimeYr | m.BalTimeSe | m.BalTimeDT
    if (bal_time_inp == BALTIME_DS_STR):
        return m.BalTimeYr | m.BalTimeSe | m.BalTimeDT | m.BalTimeDS

def get_upto_bal_area_set(m, ec):
    bal_area_inp = get_bal_area_inp(ec)
    if (bal_area_inp == BALAREA_MG_STR):
        return m.BalAreaMG
    if (bal_area_inp == BALAREA_SG1_STR):
        return m.BalAreaMG | m.BalAreaSG1
    if (bal_area_inp == BALAREA_SG2_STR):
        return m.BalAreaMG | m.BalAreaSG1 | m.BalAreaSG2
    if (bal_area_inp == BALAREA_SG3_STR):
        return m.BalAreaMG | m.BalAreaSG1 | m.BalAreaSG2 | m.BalAreaSG3

def get_bt_ba(m, ec):
    bal_time_set = get_bal_time_set(m, ec)
    bal_area_set = get_bal_area_set(m, ec)
    return bal_time_set * bal_area_set

def get_upto_bt_upto_ba(m, ec):
    upto_bal_time_set = get_upto_bal_time_set(m, ec)
    upto_bal_area_set = get_upto_bal_area_set(m, ec)
    return upto_bal_time_set * upto_bal_area_set

def get_ba1_ba2(m, ec):
    bal_area_set = get_bal_area_set(m, ec)
    return bal_area_set * bal_area_set

def get_bt_ba1_ba2(m, ec):
    bal_time_set = get_bal_time_set(m, ec)
    bal_area_set = get_bal_area_set(m, ec)
    return bal_time_set * bal_area_set * bal_area_set

def init_ec_bt_ba(m):
    return ((ec, bt_ba) 
            for ec in m.EnergyCarrier
            for bt_ba in get_bt_ba(m, ec))

def init_ec_upto_bt_upto_ba(m):
    return ((ec, upto_bt_upto_ba) 
            for ec in m.EnergyCarrier
            for upto_bt_upto_ba in get_upto_bt_upto_ba(m, ec))

def init_ppec_bt_ba(m):
    return ((ec, bt_ba) 
            for ec in m.EnergyCarrierPrimaryPhys
            for bt_ba in get_bt_ba(m, ec))

def init_dec_bt_ba(m):
    return ((ec, bt_ba) 
            for ec in m.EnergyCarrierDerivedNonPhys | m.EnergyCarrierDerivedPhys
            for bt_ba in get_bt_ba(m, ec))

def init_ec_yr_ba1_ba2(m):
    return ((ec, yr, ba1_ba2)
            for ec in m.EnergyCarrier
            for yr in m.Year
            for ba1_ba2 in get_ba1_ba2(m, ec))

def init_ec_bt_ba1_ba2(m):
    return ((ec, bt_ba1_ba2)
            for ec in m.EnergyCarrier
            for bt_ba1_ba2 in get_bt_ba1_ba2(m, ec))


model.EnergyCarrierPrimaryNonPhys = Set(initialize = get_nppec_names_list,
                                        ordered = True)
model.EnergyCarrierPrimaryPhys = Set(initialize = get_ppec_names_list,
                                     ordered = True)
model.EnergyCarrierDerivedNonPhys = Set(initialize = get_npdec_names_list,
                                        ordered = True)
model.EnergyCarrierDerivedPhys = Set(initialize = get_pdec_names_list,
                                     ordered = True)

model.EnergyCarrierAll = model.EnergyCarrierPrimaryNonPhys | \
                         model.EnergyCarrierPrimaryPhys | \
                         model.EnergyCarrierDerivedNonPhys | \
                         model.EnergyCarrierDerivedPhys

model.EnergyCarrier = model.EnergyCarrierPrimaryPhys | \
                      model.EnergyCarrierDerivedNonPhys | \
                      model.EnergyCarrierDerivedPhys

model.EC_BT_BA = Set(within = model.EnergyCarrier * model.BalTime * model.BalArea,
                     initialize = init_ec_bt_ba, ordered = True)

model.EC_UPTOBT_UPTOBA = Set(within = model.EnergyCarrier * model.BalTime * model.BalArea,
                             initialize = init_ec_upto_bt_upto_ba, ordered = True)

model.PPEC_BT_BA = Set(within = model.EnergyCarrier * model.BalTime * model.BalArea,
                       initialize = init_ppec_bt_ba, ordered = True)

model.DEC_BT_BA = Set(within = model.EnergyCarrier * model.BalTime * model.BalArea,
                      initialize = init_dec_bt_ba, ordered = True)

model.EC_YR_BA1_BA2 = Set(within = model.EnergyCarrier * model.Year * 
                                   model.BalArea * model.BalArea,
                          initialize = init_ec_yr_ba1_ba2, ordered = True)

# Unused
'''
model.EC_BT_BA1_BA2 = Set(within = model.EnergyCarrier * model.BalTime * 
                                   model.BalArea * model.BalArea,
                          initialize = init_ec_bt_ba1_ba2, ordered = True)
'''

#########                   Energy Conversion Technologies      #############

def get_ect_names_list(m):
    return list(m.ect_data[ENERGYCONVTECH_COLUMN_NAME])

def get_input_ec(ect):
    return model.ect_data_dict[INPUTEC_COLUMN_NAME].get(ect)

def get_output_dec(ect):
    return model.ect_data_dict[OUTPUTDEC_COLUMN_NAME].get(ect)

def get_ect_filtered_names_list(m):
    return [ect for ect in m.EnergyConvTech 
            if get_input_ec(ect) not in m.EnergyCarrierPrimaryNonPhys]

def get_yr_ba(m, ec):
    bal_area_set = get_bal_area_set(m, ec)
    return m.Year * bal_area_set

def init_ect_yr_ba(m):
    return ((ect, yr_ba)
            for ect in m.EnergyConvTech
            for yr_ba in get_yr_ba(m, get_output_dec(ect)))

def init_ect_filtered_yr_ba(m):
    return ((ect, yr_ba)
            for ect in m.EnergyConvTechFiltered
            for yr_ba in get_yr_ba(m, get_output_dec(ect)))

def init_ect_bt_ba(m):
    return ((ect, bt_ba)
            for ect in m.EnergyConvTech 
            for bt_ba in get_bt_ba(m, get_output_dec(ect)))

def init_ect_ba(m):
    return ((ect, ba)
            for ect in m.EnergyConvTech 
            for ba in get_bal_area_set(m, get_output_dec(ect)))


model.EnergyConvTech = Set(initialize = get_ect_names_list, ordered = True)

model.EnergyConvTechFiltered = Set(initialize = get_ect_filtered_names_list, 
                                   ordered = True)

model.ECT_YR_BA = Set(within = model.EnergyConvTech * model.Year * model.BalArea,
                      initialize = init_ect_yr_ba, ordered = True)

model.ECTFILT_YR_BA = Set(within = model.EnergyConvTechFiltered * model.Year * 
                                   model.BalArea,
                          initialize = init_ect_filtered_yr_ba, ordered = True)

model.ECT_BT_BA = Set(within = model.EnergyConvTech * model.BalTime * model.BalArea,
                      initialize = init_ect_bt_ba, ordered = True)

model.ECT_BA = Set(within = model.EnergyConvTech * model.BalArea,
                   initialize = init_ect_ba, ordered = True)

#########                   Energy Storage Technologies         #############

def get_est_names_list(m):
    return list(m.est_data[ENERGYSTORTECH_COLUMN_NAME])

def get_stored_ec(est):
    return model.est_data_dict[STOREDEC_COLUMN_NAME].get(est)

def get_stor_periodicity(est):
    return model.est_data_dict[STORPERIODICITY_COLUMN_NAME].get(est)

def init_est_yr_ba(m):
    return ((est, yr_ba)
            for est in m.EnergyStorTech
            for yr_ba in get_yr_ba(m, get_stored_ec(est)))

def init_est_ba(m):
    return ((est, ba)
            for est in m.EnergyStorTech 
            for ba in get_bal_area_set(m, get_stored_ec(est)))

model.EnergyStorTech = Set(initialize = get_est_names_list, ordered = True)

model.EST_YR_BA = Set(within = model.EnergyStorTech * model.Year * model.BalArea,
                      initialize = init_est_yr_ba, ordered = True)

model.EST_BA = Set(within = model.EnergyStorTech * model.BalArea,
                   initialize = init_est_ba, ordered = True)

#########                   Time (with day number)              #############

def create_ec_est_list_map(m):
    m.ec_est_list_map = {}

    for ec in get_ppec_names_list(m) + get_npdec_names_list(m) + \
              get_pdec_names_list(m):
        m.ec_est_list_map[ec] = []

    for est in get_est_names_list(m):
        ec = get_stored_ec(est)
        m.ec_est_list_map[ec] = m.ec_est_list_map.get(ec) + [est]

def create_ec_day_no_reqd_map(m):
    m.ec_day_no_reqd_map = {}
    m.day_no_time_elem_reqd = False

    for ec in get_ppec_names_list(m) + get_npdec_names_list(m) + \
              get_pdec_names_list(m):
        day_no_required = False
    
        bal_time_inp = get_bal_time_inp(ec)
        if (bal_time_inp in [BALTIME_DT_STR, BALTIME_DS_STR]):
            est_list = m.ec_est_list_map.get(ec)
            for est in est_list:
                if (get_stor_periodicity(est) != DAILY_RESET_STR):
                    day_no_required = True
                    m.day_no_time_elem_reqd = True
                    break;

        m.ec_day_no_reqd_map[ec] = day_no_required

'''
    for ect in get_ect_names_list(m):
        ec_inp = get_input_ec(ect)
        ec_out = get_output_dec(ect)
        if ((ec_inp not in get_nppec_names_list(m)) and 
            m.ec_day_no_reqd_map[ec_inp]):
            m.ec_day_no_reqd_map[ec_out] = True
'''

def get_day_no_list(m):
    return list(range(1, m.seasons_data[DURATION_COLUMN_NAME].max() + 1))

def get_season_dayno_tuples_list(m):
    season_dayno_tuples_list = []
    
    for se in get_season_names_list(m):
        for day_num in range(1, m.seasons_data_dict[DURATION_COLUMN_NAME][se] + 1):
            season_dayno_tuples_list.append( (se, day_num) )

    return season_dayno_tuples_list

def define_bal_time_concrete_sets_for_level3(m):
    return Set(within = m.Year * m.DummyTime * m.DummyTime * m.DummyTime,
               initialize = m.Year * m.DummyTime * m.DummyTime * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonInp * m.DummyTime * m.DummyTime,
               initialize = m.Year * m.SeasonInp * m.DummyTime * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonInp * m.DummyTime * m.DayTypeInp,
               initialize = m.Year * m.SeasonInp * m.DummyTime * m.DayTypeInp,
               ordered = True), \
           Set(within = m.Year * m.SeasonDayNoTuples * m.DayTypeInp,
               initialize = m.Year * m.SeasonDayNoTuples * m.DayTypeInp,
               ordered = True)

def define_bal_time_concrete_sets_for_level4(m):
    return Set(within = m.Year * m.DummyTime * m.DummyTime * m.DummyTime * m.DummyTime,
               initialize = m.Year * m.DummyTime * m.DummyTime * m.DummyTime * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonInp * m.DummyTime * m.DummyTime * m.DummyTime,
               initialize = m.Year * m.SeasonInp * m.DummyTime * m.DummyTime * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonInp * m.DummyTime * m.DayTypeInp * m.DummyTime,
               initialize = m.Year * m.SeasonInp * m.DummyTime * m.DayTypeInp * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonDayNoTuples * m.DayTypeInp * m.DummyTime,
               initialize = m.Year * m.SeasonDayNoTuples * m.DayTypeInp * m.DummyTime,
               ordered = True), \
           Set(within = m.Year * m.SeasonInp * m.DummyTime * m.DayTypeInp * m.DaySliceInp,
               initialize = m.Year * m.SeasonInp * m.DummyTime * m.DayTypeInp * m.DaySliceInp,
               ordered = True), \
           Set(within = m.Year * m.SeasonDayNoTuples * m.DayTypeInp * m.DaySliceInp,
               initialize = m.Year * m.SeasonDayNoTuples * m.DayTypeInp * m.DaySliceInp,
               ordered = True)

def get_bal_time_conc_set(m, ec):
    bal_time_inp = get_bal_time_inp(ec)
    if (bal_time_inp == BALTIME_YR_STR):
        return m.BalTimeYr if (not m.day_no_time_elem_reqd) else m.BalTimeYrConc
    if (bal_time_inp == BALTIME_SE_STR):
        return m.BalTimeSe if (not m.day_no_time_elem_reqd) else m.BalTimeSeConc
    if (bal_time_inp == BALTIME_DT_STR):
        return m.BalTimeDT if (not m.day_no_time_elem_reqd) else \
                           (m.BalTimeDTConcActualDayNo if m.ec_day_no_reqd_map[ec] 
                                                       else m.BalTimeDTConcDummyDayNo)
    if (bal_time_inp == BALTIME_DS_STR):
        return m.BalTimeDS if (not m.day_no_time_elem_reqd) else \
                           (m.BalTimeDSConcActualDayNo if m.ec_day_no_reqd_map[ec] 
                                                       else m.BalTimeDSConcDummyDayNo)

def get_upto_bal_time_conc_set(m, ec):
    bal_time_inp = get_bal_time_inp(ec)
    if (bal_time_inp == BALTIME_YR_STR):
        return m.BalTimeYr if (not m.day_no_time_elem_reqd) else m.BalTimeYrConc
    if (bal_time_inp == BALTIME_SE_STR):
        return m.BalTimeYr | m.BalTimeSe if (not m.day_no_time_elem_reqd) \
                                         else m.BalTimeYrConc | m.BalTimeSeConc
    if (bal_time_inp == BALTIME_DT_STR):
        return m.BalTimeYr | m.BalTimeSe | m.BalTimeDT \
               if (not m.day_no_time_elem_reqd) else \
               (m.BalTimeYrConc | m.BalTimeSeConc | m.BalTimeDTConcActualDayNo 
                if m.ec_day_no_reqd_map[ec] else 
                m.BalTimeYrConc | m.BalTimeSeConc | m.BalTimeDTConcDummyDayNo)
    if (bal_time_inp == BALTIME_DS_STR):
        return m.BalTimeYr | m.BalTimeSe | m.BalTimeDT | m.BalTimeDS \
               if (not m.day_no_time_elem_reqd) else \
               (m.BalTimeYrConc | m.BalTimeSeConc | m.BalTimeDTConcActualDayNo | m.BalTimeDSConcActualDayNo 
                if m.ec_day_no_reqd_map[ec] else 
                m.BalTimeYrConc | m.BalTimeSeConc | m.BalTimeDTConcDummyDayNo | m.BalTimeDSConcDummyDayNo)

coarser_bal_time_dict = \
{
    (BALTIME_YR_STR, BALTIME_YR_STR): BALTIME_YR_STR,
    (BALTIME_YR_STR, BALTIME_SE_STR): BALTIME_YR_STR,
    (BALTIME_YR_STR, BALTIME_DT_STR): BALTIME_YR_STR,
    (BALTIME_YR_STR, BALTIME_DS_STR): BALTIME_YR_STR,

    (BALTIME_SE_STR, BALTIME_YR_STR): BALTIME_YR_STR,
    (BALTIME_SE_STR, BALTIME_SE_STR): BALTIME_SE_STR,
    (BALTIME_SE_STR, BALTIME_DT_STR): BALTIME_SE_STR,
    (BALTIME_SE_STR, BALTIME_DS_STR): BALTIME_SE_STR,

    (BALTIME_DT_STR, BALTIME_YR_STR): BALTIME_YR_STR,
    (BALTIME_DT_STR, BALTIME_SE_STR): BALTIME_SE_STR,
    (BALTIME_DT_STR, BALTIME_DT_STR): BALTIME_DT_STR,
    (BALTIME_DT_STR, BALTIME_DS_STR): BALTIME_DT_STR,

    (BALTIME_DS_STR, BALTIME_YR_STR): BALTIME_YR_STR,
    (BALTIME_DS_STR, BALTIME_SE_STR): BALTIME_SE_STR,
    (BALTIME_DS_STR, BALTIME_DT_STR): BALTIME_DT_STR,
    (BALTIME_DS_STR, BALTIME_DS_STR): BALTIME_DS_STR
}

def get_coarser_bal_time_conc_set(m, ec1, ec2):
    bal_time_inp1 = get_bal_time_inp(ec1)
    bal_time_inp2 = get_bal_time_inp(ec2)

    coarser_bal_time_inp = coarser_bal_time_dict.get((bal_time_inp1, bal_time_inp2))

    day_no_reqd = True if (m.ec_day_no_reqd_map[ec1] and m.ec_day_no_reqd_map[ec2]) else False

    if (coarser_bal_time_inp == BALTIME_YR_STR):
        return m.BalTimeYr if (not m.day_no_time_elem_reqd) else m.BalTimeYrConc
    if (coarser_bal_time_inp == BALTIME_SE_STR):
        return m.BalTimeSe if (not m.day_no_time_elem_reqd) else m.BalTimeSeConc
    if (coarser_bal_time_inp == BALTIME_DT_STR):
        return m.BalTimeDT if (not m.day_no_time_elem_reqd) else \
                           (m.BalTimeDTConcActualDayNo if day_no_reqd 
                                                       else m.BalTimeDTConcDummyDayNo)
    if (coarser_bal_time_inp == BALTIME_DS_STR):
        return m.BalTimeDS if (not m.day_no_time_elem_reqd) else \
                           (m.BalTimeDSConcActualDayNo if day_no_reqd 
                                                       else m.BalTimeDSConcDummyDayNo)

coarser_bal_area_dict = \
{
    (BALAREA_MG_STR, BALAREA_MG_STR)   : BALAREA_MG_STR,
    (BALAREA_MG_STR, BALAREA_SG1_STR)  : BALAREA_MG_STR,
    (BALAREA_MG_STR, BALAREA_SG2_STR)  : BALAREA_MG_STR,
    (BALAREA_MG_STR, BALAREA_SG3_STR)  : BALAREA_MG_STR,

    (BALAREA_SG1_STR, BALAREA_MG_STR)   : BALAREA_MG_STR,
    (BALAREA_SG1_STR, BALAREA_SG1_STR)  : BALAREA_SG1_STR,
    (BALAREA_SG1_STR, BALAREA_SG2_STR)  : BALAREA_SG1_STR,
    (BALAREA_SG1_STR, BALAREA_SG3_STR)  : BALAREA_SG1_STR,

    (BALAREA_SG2_STR, BALAREA_MG_STR)   : BALAREA_MG_STR,
    (BALAREA_SG2_STR, BALAREA_SG1_STR)  : BALAREA_SG1_STR,
    (BALAREA_SG2_STR, BALAREA_SG2_STR)  : BALAREA_SG2_STR,
    (BALAREA_SG2_STR, BALAREA_SG3_STR)  : BALAREA_SG2_STR,

    (BALAREA_SG3_STR, BALAREA_MG_STR)   : BALAREA_MG_STR,
    (BALAREA_SG3_STR, BALAREA_SG1_STR)  : BALAREA_SG1_STR,
    (BALAREA_SG3_STR, BALAREA_SG2_STR)  : BALAREA_SG2_STR,
    (BALAREA_SG3_STR, BALAREA_SG3_STR)  : BALAREA_SG3_STR
}

def get_coarser_bal_area_set(m, ec1, ec2):
    bal_area_inp1 = get_bal_area_inp(ec1)
    bal_area_inp2 = get_bal_area_inp(ec2)

    coarser_bal_area_inp = coarser_bal_area_dict.get((bal_area_inp1, bal_area_inp2))

    return get_bal_area_set(m, ec1) \
           if (coarser_bal_area_inp == bal_area_inp1) else get_bal_area_set(m, ec2)

def get_btconc_ba(m, ec):
    bal_time_conc_set = get_bal_time_conc_set(m, ec)
    bal_area_set = get_bal_area_set(m, ec)
    return bal_time_conc_set * bal_area_set

def get_upto_btconc_upto_ba(m, ec):
    upto_bal_time_conc_set = get_upto_bal_time_conc_set(m, ec)
    upto_bal_area_set = get_upto_bal_area_set(m, ec)
    return upto_bal_time_conc_set * upto_bal_area_set

def get_time_levels_set(ec):
    return RangeSet(bal_time_str_level_map.get(get_bal_time_inp(ec)))

def get_geog_levels_set(ec):
    return RangeSet(bal_area_str_level_map.get(get_bal_area_inp(ec)))

def get_time_lvl_geog_lvl_btconc_ba(m, ec):
    time_levels_set = get_time_levels_set(ec)
    geog_levels_set = get_geog_levels_set(ec)
    bal_time_conc_set = get_bal_time_conc_set(m, ec)
    bal_area_set = get_bal_area_set(m, ec)
    return time_levels_set * geog_levels_set * bal_time_conc_set * bal_area_set

def get_btconc_ba1_ba2(m, ec):
    bal_time_conc_set = get_bal_time_conc_set(m, ec)
    bal_area_set = get_bal_area_set(m, ec)
    return bal_time_conc_set * bal_area_set * bal_area_set

def init_ec_btconc_ba(m):
    return ((ec, btconc_ba) 
            for ec in m.EnergyCarrier
            for btconc_ba in get_btconc_ba(m, ec))

def init_ec_upto_btconc_upto_ba(m):
    return ((ec, upto_btconc_upto_ba) 
            for ec in m.EnergyCarrier
            for upto_btconc_upto_ba in get_upto_btconc_upto_ba(m, ec))

def init_ec_time_lvl_geog_lvl_btconc_ba(m):
    return ((ec, time_lvl_geog_lvl_btconc_ba) 
            for ec in m.EnergyCarrier
            for time_lvl_geog_lvl_btconc_ba in get_time_lvl_geog_lvl_btconc_ba(m, ec))

def init_ppec_btconc_ba(m):
    return ((ec, btconc_ba) 
            for ec in m.EnergyCarrierPrimaryPhys
            for btconc_ba in get_btconc_ba(m, ec))

def init_dec_btconc_ba(m):
    return ((ec, btconc_ba) 
            for ec in m.EnergyCarrierDerivedNonPhys | m.EnergyCarrierDerivedPhys
            for btconc_ba in get_btconc_ba(m, ec))

def init_ec_btconc_ba1_ba2(m):
    return ((ec, btconc_ba1_ba2)
            for ec in m.EnergyCarrier
            for btconc_ba1_ba2 in get_btconc_ba1_ba2(m, ec))

def init_iy_ect_yr_ba(m):
    return ((iy, ect, yr, ba)
            for iy in m.InstYear for ect in m.EnergyConvTech
            for ec in [get_output_dec(ect)]
            for yr in m.Year if iy <= yr 
            for ba in get_bal_area_set(m, ec))

def init_iy_ect_btconc_ba(m):
    return ((iy, ect, btconc, ba)
            for iy in m.InstYear for ect in m.EnergyConvTech
            for ec in [get_output_dec(ect)]
            for btconc in get_bal_time_conc_set(m, ec) if iy <= btconc[0] 
            for ba in get_bal_area_set(m, ec))

def init_iy_ect_filt_btconc_ba(m):
    return ((iy, ect, btconc, ba)
            for iy in m.InstYear for ect in m.EnergyConvTechFiltered
            for ec in [get_output_dec(ect)]
            for btconc in get_bal_time_conc_set(m, ec) if iy <= btconc[0] 
            for ba in get_bal_area_set(m, ec))

def init_iy_ect_filt_btconc_ba_input_gran(m):
    return ((iy, ect, btconc, ba)
            for iy in m.InstYear for ect in m.EnergyConvTechFiltered
            for ec in [get_input_ec(ect)]
            for btconc in get_bal_time_conc_set(m, ec) if iy <= btconc[0] 
            for ba in get_bal_area_set(m, ec))

def init_iy_ect_filt_btconc_ba_coarser_gran(m):
    return ((iy, ect, btconc, ba)
            for iy in m.InstYear for ect in m.EnergyConvTechFiltered
            for ec_inp in [get_input_ec(ect)] for ec_out in [get_output_dec(ect)]
            for btconc in get_coarser_bal_time_conc_set(m, ec_inp, ec_out) if iy <= btconc[0] 
            for ba in get_coarser_bal_area_set(m, ec_inp, ec_out))

def init_iy_est_btconc_ba(m):
    return ((iy, est, btconc, ba)
            for iy in m.InstYear for est in m.EnergyStorTech
            for ec in [get_stored_ec(est)]
            for btconc in get_bal_time_conc_set(m, ec) if iy <= btconc[0] 
            for ba in get_bal_area_set(m, ec))


create_ec_est_list_map(model)
create_ec_day_no_reqd_map(model)

model.num_conc_time_colms = model.num_time_levels_to_use + \
                            (1 if model.day_no_time_elem_reqd else 0)
model.num_geog_colms = model.num_geography_levels_to_use

if (not model.day_no_time_elem_reqd):
    if (model.num_time_levels_to_use == 1):
        model.BalTimeConc = Set(within = model.Year,
                                initialize = model.BalTimeYr,
                                ordered = True)
    elif (model.num_time_levels_to_use == 2):
        model.BalTimeConc = Set(within = model.Year * model.Season,
                                initialize = model.BalTimeYr | model.BalTimeSe,
                                ordered = True)
    elif (model.num_time_levels_to_use == 3):
        model.BalTimeConc = Set(within = model.Year * model.Season * 
                                         model.DayType,
                                initialize = model.BalTimeYr | model.BalTimeSe | 
                                             model.BalTimeDT,
                                ordered = True)
    elif (model.num_time_levels_to_use == 4):
        model.BalTimeConc = Set(within = model.Year * model.Season * 
                                         model.DayType * model.DaySlice,
                                initialize = model.BalTimeYr | model.BalTimeSe | 
                                             model.BalTimeDT | model.BalTimeDS,
                                ordered = True)
else:
    model.DayNoInp = Set(initialize = get_day_no_list, ordered = True)
    model.DayNo = model.DayNoInp | model.DummyTime
    
    model.SeasonDayNoTuples = Set(within = model.SeasonInp * model.DayNoInp,
                                  initialize = get_season_dayno_tuples_list,
                                  ordered = True)

    if (model.num_time_levels_to_use == 3):
        model.BalTimeYrConc, model.BalTimeSeConc, \
        model.BalTimeDTConcDummyDayNo, model.BalTimeDTConcActualDayNo  = \
                                       define_bal_time_concrete_sets_for_level3(model)

        model.BalTimeConc = Set(within = model.Year * model.Season * 
                                         model.DayNo * model.DayType,
                                initialize = model.BalTimeYrConc | 
                                             model.BalTimeSeConc | 
                                             model.BalTimeDTConcDummyDayNo |
                                             model.BalTimeDTConcActualDayNo,
                                ordered = True)
    elif (model.num_time_levels_to_use == 4):
        model.BalTimeYrConc, model.BalTimeSeConc, \
        model.BalTimeDTConcDummyDayNo, model.BalTimeDTConcActualDayNo, \
        model.BalTimeDSConcDummyDayNo, model.BalTimeDSConcActualDayNo = \
                                       define_bal_time_concrete_sets_for_level4(model)

        model.BalTimeConc = Set(within = model.Year * model.Season * 
                                         model.DayNo * model.DayType * 
                                         model.DaySlice,
                                initialize = model.BalTimeYrConc | 
                                             model.BalTimeSeConc | 
                                             model.BalTimeDTConcDummyDayNo |
                                             model.BalTimeDTConcActualDayNo |
                                             model.BalTimeDSConcDummyDayNo |
                                             model.BalTimeDSConcActualDayNo,
                                ordered = True)

model.EC_BTCONC_BA = Set(within = model.EnergyCarrier * model.BalTimeConc * 
                                  model.BalArea,
                         initialize = init_ec_btconc_ba, ordered = True)

model.EC_UPTOBTCONC_UPTOBA = Set(within = model.EnergyCarrier * model.BalTimeConc * 
                                          model.BalArea,
                                 initialize = init_ec_upto_btconc_upto_ba,
                                 ordered = True)

model.EC_TL_GL_BTCONC_BA = Set(within = model.EnergyCarrier * 
                                        model.TimeLevel * model.GeogLevel * 
                                        model.BalTimeConc * model.BalArea,
                               initialize = init_ec_time_lvl_geog_lvl_btconc_ba,
                               ordered = True)

model.PPEC_BTCONC_BA = Set(within = model.EnergyCarrier * model.BalTimeConc * 
                                    model.BalArea,
                           initialize = init_ppec_btconc_ba, ordered = True)

model.DEC_BTCONC_BA = Set(within = model.EnergyCarrier * model.BalTimeConc * 
                                   model.BalArea,
                          initialize = init_dec_btconc_ba, ordered = True)

model.EC_BTCONC_BA1_BA2 = Set(within = model.EnergyCarrier * model.BalTimeConc * 
                                       model.BalArea * model.BalArea,
                              initialize = init_ec_btconc_ba1_ba2, ordered = True)

model.IY_ECT_YR_BA = Set(within = model.InstYear * model.EnergyConvTech * 
                                  model.Year * model.BalArea, 
                                  initialize = init_iy_ect_yr_ba, ordered = True)

model.IY_ECT_BTCONC_BA = Set(within = model.InstYear * model.EnergyConvTech * 
                                      model.BalTimeConc * model.BalArea, 
                             initialize = init_iy_ect_btconc_ba, ordered = True)

model.IY_ECTFILT_BTCONC_BA = Set(within = model.InstYear * 
                                          model.EnergyConvTechFiltered * 
                                          model.BalTimeConc * model.BalArea, 
                                 initialize = init_iy_ect_filt_btconc_ba, 
                                 ordered = True)

model.IY_ECTFILT_BTCONC_BA_OUTPUT_GRAN = Set(within = model.InstYear * 
                                                      model.EnergyConvTechFiltered * 
                                                      model.BalTimeConc * model.BalArea, 
                                             initialize = model.IY_ECTFILT_BTCONC_BA, 
                                             ordered = True)

model.IY_ECTFILT_BTCONC_BA_INPUT_GRAN = Set(within = model.InstYear * 
                                                     model.EnergyConvTechFiltered * 
                                                     model.BalTimeConc * model.BalArea, 
                                            initialize = init_iy_ect_filt_btconc_ba_input_gran, 
                                            ordered = True)

model.IY_ECTFILT_BTCONC_BA_COARSER_GRAN = Set(within = model.InstYear * 
                                                       model.EnergyConvTechFiltered * 
                                                       model.BalTimeConc * model.BalArea, 
                                              initialize = init_iy_ect_filt_btconc_ba_coarser_gran, 
                                              ordered = True)

model.IY_EST_BTCONC_BA = Set(within = model.InstYear * model.EnergyStorTech * 
                                      model.BalTimeConc * model.BalArea, 
                             initialize = init_iy_est_btconc_ba, ordered = True)


#####################
#    Parameters     #
#####################

get_supply_params_data(model)

#########                   Energy Unit Conversions             #############

def get_eu_conv_dict(m):
    return m.eu_conv_data_dict[ENERGYUNITCONV_COLUMN_NAME]

model.EnergyUnitConv = Param(model.EU, model.EU, 
                             initialize = get_eu_conv_dict,
                             within = PositiveReals)

#########                   Energy Carriers                        #############

def get_dom_energy_density_dict(m):
    return m.ppec_energy_density_data_dict[DOMENERGYDENSITY_COLUMN_NAME]

def get_imp_energy_density_dict(m):
    return m.ppec_energy_density_data_dict[IMPENERGYDENSITY_COLUMN_NAME]

def get_energy_density_dict(m):
    return m.pdec_energy_density_data_dict[ENERGYDENSITY_COLUMN_NAME]

def get_non_energy_share_dict(m):
    return m.pec_info_dict[NONENERGYSHARE_COLUMN_NAME]

def get_domestic_price_dict(m):
    return m.pec_info_dict[DOMESTICPRICE_COLUMN_NAME]

def get_av_tax_oh_dom_dict(m):
    return m.pec_info_dict[AVTAXOHDOM_COLUMN_NAME]

def get_fixed_tax_oh_dom_dict(m):
    return m.pec_info_dict[FIXEDTAXOHDOM_COLUMN_NAME]

def get_import_price_dict(m):
    return m.pec_info_dict[IMPORTPRICE_COLUMN_NAME]

def get_av_tax_oh_imp_dict(m):
    return m.pec_info_dict[AVTAXOHIMP_COLUMN_NAME]

def get_fixed_tax_oh_imp_dict(m):
    return m.pec_info_dict[FIXEDTAXOHIMP_COLUMN_NAME]

def get_max_domestic_prod_dict(m):
    return m.pec_prod_imp_constraints_dict[MAXDOMESTICPRODUCT_COLUMN_NAME]

def get_max_import_dict(m):
    return m.pec_prod_imp_constraints_dict[MAXIMPORT_COLUMN_NAME]

def get_fixed_tax_oh_dec_dict(m):
    return m.dec_taxation_dict[FIXEDTAXOH_COLUMN_NAME]

def get_cost_per_unmet_unit_dict(m):
    return m.unmet_demand_value_data_dict[UNMETDEMANDVALUE_COLUMN_NAME]


model.EnergyUnit = Param(model.EnergyCarrier,
                         initialize = model.ec_energy_unit_map, within = Any)

model.DomEnergyDensity = Param(model.EnergyCarrierPrimaryPhys,
                               model.Year,
                               initialize = get_dom_energy_density_dict,
                               within = PositiveReals)

model.ImpEnergyDensity = Param(model.EnergyCarrierPrimaryPhys,
                               model.Year,
                               initialize = get_imp_energy_density_dict,
                               within = PositiveReals)

model.EnergyDensity = Param(model.EnergyCarrierDerivedPhys,
                            model.Year,
                            initialize = get_energy_density_dict,
                            within = PositiveReals)

model.NonEnergyShare = Param(model.PPEC_BT_BA, 
                             initialize = get_non_energy_share_dict,
                             within = PercentFraction, default = 0.0)

model.DomesticPrice = Param(model.PPEC_BT_BA, 
                            initialize = get_domestic_price_dict,
                            within = NonNegativeReals, default = 0.0)

model.AVTaxOHDom = Param(model.PPEC_BT_BA, 
                         initialize = get_av_tax_oh_dom_dict,
                         within = NonNegativeReals, default = 0.0)

model.FixedTaxOHDom = Param(model.PPEC_BT_BA, 
                            initialize = get_fixed_tax_oh_dom_dict,
                            within = NonNegativeReals, default = 0.0)

model.ImportPrice = Param(model.PPEC_BT_BA, 
                          initialize = get_import_price_dict,
                          within = NonNegativeReals, default = 0.0)

model.AVTaxOHImp = Param(model.PPEC_BT_BA, 
                         initialize = get_av_tax_oh_imp_dict,
                         within = NonNegativeReals, default = 0.0)

model.FixedTaxOHImp = Param(model.PPEC_BT_BA, 
                            initialize = get_fixed_tax_oh_imp_dict,
                            within = NonNegativeReals, default = 0.0)

model.MaxDomesticProd = Param(model.PPEC_BT_BA, 
                              initialize = get_max_domestic_prod_dict,
                              within = NonNegativeReals, default = 0.0)

model.MaxImport = Param(model.PPEC_BT_BA, 
                        initialize = get_max_import_dict,
                        within = NonNegativeReals, default = 0.0)

model.FixedTaxOHDEC = Param(model.DEC_BT_BA, 
                            initialize = get_fixed_tax_oh_dec_dict,
                            within = NonNegativeReals, default = 0.0)

model.CostPerUnmetUnit = Param(model.EnergyCarrier,
                               model.Year,
                               initialize = get_cost_per_unmet_unit_dict,
                               within = NonNegativeReals,
                               default = VERY_LARGE_VALUE)

#########                   Energy Conversion Technologies      #############

def get_annual_output_pu_capacity_dict(m):
    return m.ect_data_dict[ANNUALOUTPUTPERUNITCAPACITY_COLUMN_NAME]

def get_max_capacity_dict(m):
    return m.ect_cap_add_bounds_dict[MAXCAPACITY_COLUMN_NAME]

def get_min_capacity_dict(m):
    return m.ect_cap_add_bounds_dict[MINCAPACITY_COLUMN_NAME]

def get_lifetime_dict(m):
    return m.ect_lifetime_dict[LIFETIME_COLUMN_NAME]

def get_aux_cons_dict(m):
    return m.ect_operational_info_dict[SELFCONS_COLUMN_NAME]

def get_capacity_derating_dict(m):
    return m.ect_operational_info_dict[CAPACITYDERATING_COLUMN_NAME]

def get_max_ramp_up_rate_dict(m):
    return m.ect_operational_info_dict[MAXRAMPUPRATE_COLUMN_NAME]

def get_max_ramp_down_rate_dict(m):
    return m.ect_operational_info_dict[MAXRAMPDOWNRATE_COLUMN_NAME]

def get_conv_eff_dict(m):
    return m.ect_efficiency_cost_max_annual_uf_dict[CONVEFF_COLUMN_NAME]

def get_fixed_cost_dict(m):
    return m.ect_efficiency_cost_max_annual_uf_dict[FIXEDCOST_COLUMN_NAME]

def get_var_cost_dict(m):
    return m.ect_efficiency_cost_max_annual_uf_dict[VARCOST_COLUMN_NAME]

def get_max_annual_uf_dict(m):
    return m.ect_efficiency_cost_max_annual_uf_dict[MAXANNUALUF_COLUMN_NAME]

def get_max_uf_dict(m):
    return m.ect_max_cuf_dict[MAXUF_COLUMN_NAME]

def get_legacy_capacity_dict(m):
    return m.ect_legacy_capacity_dict[LEGACYCAPACITY_COLUMN_NAME]

def get_legacy_retirement_dict(m):
    return m.ect_legacy_retirement_dict[RETCAPACITY_COLUMN_NAME]


model.AnnualOutputPerUnitCapacity = Param(model.EnergyConvTech,
                                          initialize = \
                                          get_annual_output_pu_capacity_dict,
                                          within = PositiveReals)

model.MaxCapacity = Param(model.ECT_YR_BA,
                          initialize = get_max_capacity_dict,
                          within = Reals, default = 0.0)

model.MinCapacity = Param(model.ECT_YR_BA,
                          initialize = get_min_capacity_dict,
                          within = NonNegativeReals, default = 0.0)

model.Lifetime = Param(model.EnergyConvTech, model.Year,
                       initialize = get_lifetime_dict,
                       within = PositiveIntegers)

model.AuxCons = Param(model.EnergyConvTech, model.InstYear,
                      initialize = get_aux_cons_dict,
                      within = PercentFraction, default = 0.0)

model.CapacityDerating = Param(model.EnergyConvTech, model.InstYear,
                               initialize = get_capacity_derating_dict,
                               within = PercentFraction, default = 0.0)

model.MaxRampUpRate = Param(model.EnergyConvTech, model.InstYear,
                            initialize = get_max_ramp_up_rate_dict,
                            within = PercentFraction, default = 1.0)

model.MaxRampDownRate = Param(model.EnergyConvTech, model.InstYear,
                              initialize = get_max_ramp_down_rate_dict,
                              within = PercentFraction, default = 1.0)

model.ConvEff = Param(model.InstYear, model.ECT_YR_BA,
                      initialize = get_conv_eff_dict,
                      within = PercentFraction, default = 1.0)

model.FixedCost = Param(model.InstYear, model.ECT_YR_BA,
                        initialize = get_fixed_cost_dict,
                        within = NonNegativeReals, default = 0.0)

model.VarCost = Param(model.InstYear, model.ECT_YR_BA,
                      initialize = get_var_cost_dict,
                      within = NonNegativeReals, default = 0.0)

model.MaxAnnualUF = Param(model.InstYear, model.ECT_YR_BA,
                          initialize = get_max_annual_uf_dict,
                          within = PercentFraction, default = 1.0)

model.MaxUF = Param(model.InstYear, model.ECT_BT_BA,
                    initialize = get_max_uf_dict,
                    within = PercentFraction, default = 1.0)

model.LegacyCapacity = Param(model.ECT_BA,
                             initialize = get_legacy_capacity_dict,
                             within = NonNegativeReals, default = 0.0)

model.LegacyRetirement = Param(model.ECT_YR_BA,
                               initialize = get_legacy_retirement_dict,
                               within = NonNegativeReals, default = 0.0)

#########                   Energy Storage Technologies         #############

def get_dom_or_imp_dict(m):
    return m.est_data_dict[DOMORIMP_COLUMN_NAME]

def get_max_charge_rate_dict(m):
    return m.est_data_dict[MAXCHARGERATE_COLUMN_NAME]

def get_max_discharge_rate_dict(m):
    return m.est_data_dict[MAXDISCHARGERATE_COLUMN_NAME]

def get_stor_lifetime_years_dict(m):
    return m.est_lifetime_dict[LIFETIMEYEARS_COLUMN_NAME]

def get_stor_lifetime_cycles_dict(m):
    return m.est_lifetime_dict[LIFETIMECYCLES_COLUMN_NAME]

def get_max_stor_capacity_dict(m):
    return m.est_cap_add_bounds_dict[MAXCAP_COLUMN_NAME]

def get_min_stor_capacity_dict(m):
    return m.est_cap_add_bounds_dict[MINCAP_COLUMN_NAME]

def get_stor_capacity_derating_dict(m):
    return m.est_derating_depth_of_discharge_dict[CAPACITYDERATING_COLUMN_NAME]

def get_depth_of_discharge_dict(m):
    return m.est_derating_depth_of_discharge_dict[DEPTHOFDISCHARGE_COLUMN_NAME]

def get_stor_efficiency_dict(m):
    return m.est_efficiency_cost_dict[EFFICIENCY_COLUMN_NAME]

def get_stor_fixed_cost_dict(m):
    return m.est_efficiency_cost_dict[FIXEDCOST_COLUMN_NAME]

def get_legacy_stor_capacity_dict(m):
    return m.est_legacy_details_dict[LEGACYCAPACITY_COLUMN_NAME]

def get_legacy_bal_lifetime_dict(m):
    return m.est_legacy_details_dict[BALLIFETIME_COLUMN_NAME]

def get_legacy_bal_cycles_dict(m):
    return m.est_legacy_details_dict[BALCYCLES_COLUMN_NAME]


model.DomOrImp = Param(model.EnergyStorTech,
                       initialize = get_dom_or_imp_dict,
                       within = {EC_DOM_STR, EC_IMP_STR})

model.MaxChargeRate = Param(model.EnergyStorTech,
                            initialize = get_max_charge_rate_dict,
                            within = PositiveReals)

model.MaxDischargeRate = Param(model.EnergyStorTech,
                               initialize = get_max_discharge_rate_dict,
                               within = PositiveReals)

model.StorLifetimeYears = Param(model.EnergyStorTech, model.Year,
                                initialize = get_stor_lifetime_years_dict,
                                within = PositiveIntegers)

model.StorLifetimeCycles = Param(model.EnergyStorTech, model.Year,
                                 initialize = get_stor_lifetime_cycles_dict,
                                 within = PositiveReals)

model.MaxStorCapacity = Param(model.EST_YR_BA,
                              initialize = get_max_stor_capacity_dict,
                              within = Reals, default = 0.0)

model.MinStorCapacity = Param(model.EST_YR_BA,
                              initialize = get_min_stor_capacity_dict,
                              within = NonNegativeReals, default = 0.0)

model.StorCapacityDerating = Param(model.EnergyStorTech, model.InstYear,
                                   initialize = get_stor_capacity_derating_dict,
                                   within = PercentFraction, default = 0.0)

model.DepthOfDischarge = Param(model.EnergyStorTech, model.InstYear,
                               initialize = get_depth_of_discharge_dict,
                               within = PercentFraction, default = 0.0)

model.StorEfficiency = Param(model.InstYear, model.EST_YR_BA,
                             initialize = get_stor_efficiency_dict,
                             within = PercentFraction, default = 0.0)

model.StorFixedCost = Param(model.InstYear, model.EST_YR_BA,
                            initialize = get_stor_fixed_cost_dict,
                            within = NonNegativeReals, default = 0.0)

model.LegacyStorCapacity = Param(model.EST_BA,
                                 initialize = get_legacy_stor_capacity_dict,
                                 within = NonNegativeReals, default = 0.0)

model.LegacyBalLifetime = Param(model.EST_BA,
                                initialize = get_legacy_bal_lifetime_dict,
                                within = PositiveIntegers)

model.LegacyBalCycles = Param(model.EST_BA,
                              initialize = get_legacy_bal_cycles_dict,
                              within = PositiveReals)

#########                   Inter-geography energy flow         #############

def get_transit_cost_dict(m):
    return m.ec_transfers_dict[TRANSITCOST_COLUMN_NAME]

def get_transit_loss_dict(m):
    return m.ec_transfers_dict[TRANSITLOSS_COLUMN_NAME]

def get_max_transit_dict(m):
    return m.ec_transfers_dict[MAXTRANSIT_COLUMN_NAME]


model.TransitCost = Param(model.EC_YR_BA1_BA2,
                          initialize = get_transit_cost_dict,
                          within = NonNegativeReals, default = 0.0)

model.TransitLoss = Param(model.EC_YR_BA1_BA2,
                          initialize = get_transit_loss_dict,
                          within = PercentFraction, default = 0.0)

model.MaxTransit = Param(model.EC_YR_BA1_BA2,
                         initialize = get_max_transit_dict,
                         within = Reals, default = 0.0)

#########                   End-use Demand                      #############

def get_end_use_demand_energy_dict(m):
    return m.end_use_demand_energy_dict[ENDUSEDEMANDENERGY_COLUMN_NAME]

model.EndUseDemandEnergy = Param(model.EC_UPTOBT_UPTOBA, 
                                 initialize = get_end_use_demand_energy_dict,
                                 within = NonNegativeReals, default = 0.0)


#########################
#   Derived Parameters  #
#########################

def prev_conc_time_elem(m, ec, btconc_elem):
    bal_time_conc_set = get_bal_time_conc_set(m, ec)
    pos = bal_time_conc_set.ord(btconc_elem)
    return None if (pos == 1) else bal_time_conc_set.prevw(btconc_elem)

def num_hours_conc_time_elem(m, ec, btconc_elem):
    bal_time_inp = get_bal_time_inp(ec)
    if (bal_time_inp == BALTIME_YR_STR):
        return HOURS_PER_YEAR
    if (bal_time_inp == BALTIME_SE_STR):
        return m.NumDaysPerSeason[btconc_elem[1]] * HOURS_PER_DAY
    if (bal_time_inp == BALTIME_DT_STR):
        return HOURS_PER_DAY
    if (bal_time_inp == BALTIME_DS_STR):
        ds = btconc_elem[4] if m.day_no_time_elem_reqd else btconc_elem[3]
        return m.NumHoursPerDaySlice[ds]

def share_of_bt_in_year(m, ec, btconc_elem):
    bal_time_inp = get_bal_time_inp(ec)
    if (bal_time_inp == BALTIME_YR_STR):
        return 1
    if (bal_time_inp == BALTIME_SE_STR):
        return m.NumDaysPerSeason[btconc_elem[1]] / DAYS_PER_YEAR
    if (bal_time_inp == BALTIME_DT_STR):
        return 1 / DAYS_PER_YEAR
    if (bal_time_inp == BALTIME_DS_STR):
        ds = btconc_elem[4] if m.day_no_time_elem_reqd else btconc_elem[3]
        return m.NumHoursPerDaySlice[ds] / HOURS_PER_YEAR

def get_season_duration_dict(m):
    return m.seasons_data_dict[DURATION_COLUMN_NAME]

def get_daytype_weight_dict(m):
    return m.daytypes_data_dict[WEIGHT_COLUMN_NAME]

def get_dayslice_duration_dict(m):
    return m.dayslices_data_dict[DURATION_COLUMN_NAME]

def get_end_use_demand_dict(m):
    return m.end_use_demand_energy_dict[ENDUSEDEMAND_COLUMN_NAME]

def get_end_use_demand(m, ec, *args):       #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]
    bt = (btconc[0 : 2] + btconc[3 : ]) if m.day_no_time_elem_reqd else btconc
    yr = btconc[0]
    
    if ec in m.EnergyCarrierPrimaryPhys:
        return m.EndUseDemandEnergy[ec, bt, ba] / m.DomEnergyDensity[ec, yr]
    elif ec in m.EnergyCarrierDerivedPhys:
        return m.EndUseDemandEnergy[ec, bt, ba] / m.EnergyDensity[ec, yr]
    else:
        return m.EndUseDemandEnergy[ec, bt, ba]

def calc_derived_transit_cost(m, ec, yr, *args):    #args = ba1_ba2
    ba1 = args[0 : m.num_geog_colms]
    ba2 = args[m.num_geog_colms : 2 * m.num_geog_colms]

    return (m.TransitCost[ec, yr, args] + 
            (0 if (ba1 == ba2) else 
             m.TransitCost[ec, yr, ba2, ba2] * (1 - m.TransitLoss[ec, yr, args])
            ))

def calc_throughput(m, ec, yr, *args):      #args = ba1_ba2
    ba1 = args[0 : m.num_geog_colms]
    ba2 = args[m.num_geog_colms : 2 * m.num_geog_colms]

    return (1 - m.TransitLoss[ec, yr, ba1, ba1]) if (ba1 == ba2) else \
           (1 - m.TransitLoss[ec, yr, ba1, ba2]) * (1 - m.TransitLoss[ec, yr, ba2, ba2])

def calc_derived_max_transit(m, ec, *args):     #args = btconc_ba1_ba2
    ba1_ba2 = args[m.num_conc_time_colms : m.num_conc_time_colms + 
                                           2 * m.num_geog_colms]
    yr = args[0]

    bal_time_inp = get_bal_time_inp(ec)
    if (bal_time_inp == BALTIME_YR_STR):
        return m.MaxTransit[ec, yr, ba1_ba2]
    if (bal_time_inp == BALTIME_SE_STR):
        return m.MaxTransit[ec, yr, ba1_ba2] * (m.NumDaysPerSeason[args[1]] / DAYS_PER_YEAR)
    if (bal_time_inp == BALTIME_DT_STR):
        return m.MaxTransit[ec, yr, ba1_ba2]
    if (bal_time_inp == BALTIME_DS_STR):
        ds = args[4] if m.day_no_time_elem_reqd else args[3]
        return m.MaxTransit[ec, yr, ba1_ba2] * m.NumHoursPerDaySlice[ds]    

def get_stor_lifetime_years_in_model_dict(m):
    dict1 = {(m.StartYear - 1, est_ba) : value + 1 
             for est_ba, value in get_legacy_bal_lifetime_dict(m).items()}
    dict2 = {(est_yr[1], est_yr[0], ba) : value 
             for est_yr, value in get_stor_lifetime_years_dict(m).items()
             for ba in get_bal_area_set(m, get_stored_ec(est_yr[0]))}
    return {**dict1, **dict2}

def get_stor_lifetime_cycles_in_model_dict(m):
    dict1 = {(m.StartYear - 1, est_ba) : value 
             for est_ba, value in get_legacy_bal_cycles_dict(m).items()}
    dict2 = {(est_yr[1], est_yr[0], ba) : value 
             for est_yr, value in get_stor_lifetime_cycles_dict(m).items()
             for ba in get_bal_area_set(m, get_stored_ec(est_yr[0]))}
    return {**dict1, **dict2}

def get_pyomo_set_value_from_ordered_position(pyomo_set_obj, position):
    if (hasattr(pyomo_set_obj, "at") and callable(pyomo_set_obj.at)):
        return pyomo_set_obj.at(position)
    else:
        return pyomo_set_obj[position]

def get_first_daysliceinp(m):
    return get_pyomo_set_value_from_ordered_position(m.DaySliceInp, 1)

def get_last_daysliceinp(m):
    return get_pyomo_set_value_from_ordered_position(m.DaySliceInp, len(m.DaySliceInp))

def is_time_to_reset_stor(m, est, btconc_elem):
    bal_time_inp = get_bal_time_inp(get_stored_ec(est))
    stor_periodicity_inp = get_stor_periodicity(est)

    if (bal_time_inp == BALTIME_SE_STR):
        if (stor_periodicity_inp == ANNUAL_RESET_STR):
            if (btconc_elem[1] == m.SeasonInp[1]):
                return True
        return False
    if (bal_time_inp == BALTIME_DT_STR):
        if (stor_periodicity_inp == ANNUAL_RESET_STR):
            if ((btconc_elem[1] == m.SeasonInp[1]) and (btconc_elem[2] == 1)):
                return True
            return False
        if ((stor_periodicity_inp == SEASONAL_RESET_STR) and (btconc_elem[2] == 1)):
            return True
        return False
    if (bal_time_inp == BALTIME_DS_STR):
        if (stor_periodicity_inp == ANNUAL_RESET_STR):
            if ((btconc_elem[1] == m.SeasonInp[1]) and (btconc_elem[2] == 1) and 
                (btconc_elem[4] == get_first_daysliceinp(m))):
                return True
            return False
        if (stor_periodicity_inp == SEASONAL_RESET_STR):
            if ((btconc_elem[2] == 1) and (btconc_elem[4] == get_first_daysliceinp(m))):
                return True
            return False
        if (stor_periodicity_inp == DAILY_RESET_STR):
            ds = btconc_elem[4] if m.day_no_time_elem_reqd else btconc_elem[3]
            if (ds == get_first_daysliceinp(m)):
                return True
        return False
    return False

def get_ec_scale_factor(m, ec, btconc_elem):
    bal_time_inp = get_bal_time_inp(ec)
    if ((bal_time_inp == BALTIME_YR_STR) or (bal_time_inp == BALTIME_SE_STR)):
        return 1
    if ((bal_time_inp == BALTIME_DT_STR) or (bal_time_inp == BALTIME_DS_STR)):
        se = btconc_elem[1]
        dt = btconc_elem[3] if m.day_no_time_elem_reqd else btconc_elem[2]
        return 1 if m.ec_day_no_reqd_map[ec] else \
                 m.NumDaysPerSeason[se] * m.WeightPerDayType[dt]

def get_ect_scale_factor(m, ect, btconc_elem_out):
    ec_inp = get_input_ec(ect)
    ec_out = get_output_dec(ect)
    ec_inp_bt_inp = get_bal_time_inp(ec_inp)
    ec_out_bt_inp = get_bal_time_inp(ec_out)
    
    if ((ec_inp_bt_inp in [BALTIME_YR_STR, BALTIME_SE_STR]) and 
        (ec_out_bt_inp in [BALTIME_DT_STR, BALTIME_DS_STR]) and
        (not m.ec_day_no_reqd_map[ec_out])):
        se = btconc_elem_out[1]
        dt = btconc_elem_out[3] if m.day_no_time_elem_reqd else btconc_elem_out[2]
        return m.NumDaysPerSeason[se] * m.WeightPerDayType[dt]
    else:
        return 1

def get_current_age_in_years(m, curr_yr, inst_yr):
    return curr_yr - inst_yr - (1 if (inst_yr == m.StartYear - 1) else 0)

def is_contained_in_time(m, btconc_elem_coarse, btconc_elem_fine):
#   The logic implemented below works in the following cases:
#     1. Neither btconc_elem_coarse nor btconc_elem_fine has an actual day number
#     2. Both btconc_elem_coarse and btconc_elem_fine have an actual day number
#     3. btconc_elem_coarse does not have an actual day number,
#        but btconc_elem_fine has an actual day number
#   The logic does not work if btconc_elem_coarse has an actual day number,
#   but btconc_elem_fine does not have an actual day number
    for i in range(m.num_conc_time_colms):
        if (btconc_elem_coarse[i] == DUMMY_TIME_STR): continue
        if (btconc_elem_coarse[i] == btconc_elem_fine[i]): continue
        return False

    return True

def is_contained_in_geog(m, ba_elem_coarse, ba_elem_fine):
    for i in range(m.num_geog_colms):
        if (ba_elem_coarse[i] == DUMMY_GEOG_STR): continue
        if (ba_elem_coarse[i] == ba_elem_fine[i]): continue
        return False

    return True

def get_ect_unit_io_output_conv_factor(m, dom_or_imp, iy, ect, yr, *args):  # args = ba
    iec = get_input_ec(ect)
    oec = get_output_dec(ect)

    if iec in m.EnergyCarrierPrimaryPhys:
        conv_fact = m.DomEnergyDensity[iec, yr] if (dom_or_imp == EC_DOM_STR) \
                                                else m.ImpEnergyDensity[iec, yr]
    elif iec in m.EnergyCarrierDerivedPhys:
        conv_fact = m.EnergyDensity[iec, yr]
    else:
        conv_fact = 1

    conv_fact *= m.ConvEff[iy, ect, yr, args]

    conv_fact *= m.EnergyUnitConv[m.EnergyUnit[iec], m.EnergyUnit[oec]]

    if oec in m.EnergyCarrierDerivedPhys:
        conv_fact /= m.EnergyDensity[oec, yr]

    # conv_fact *=  (1 - m.AuxCons[ect, iy])

    return conv_fact

def get_time_level(m, btconc_elem):
    time_level = 0

    for i in range(m.num_conc_time_colms):
        if (btconc_elem[i] != DUMMY_TIME_STR):
            time_level += 1

    if (m.day_no_time_elem_reqd and (btconc_elem[2] != DUMMY_TIME_STR)):
        time_level -= 1

    return time_level

def get_geog_level(m, ba_elem):
    geog_level = 0

    for i in range(m.num_geog_colms):
        if (ba_elem[i] != DUMMY_GEOG_STR):
            geog_level += 1

    return geog_level

def get_time_scale_factor(m, btconc_elem_coarse, btconc_elem_fine):
    tl_coarse = get_time_level(m, btconc_elem_coarse)
    tl_fine = get_time_level(m, btconc_elem_fine)

    actual_day_no_in_coarse = True if (m.day_no_time_elem_reqd and 
                                       (btconc_elem_coarse[2] != DUMMY_TIME_STR)) else False
    actual_day_no_in_fine = True if (m.day_no_time_elem_reqd and 
                                     (btconc_elem_fine[2] != DUMMY_TIME_STR)) else False

    if ((tl_coarse <= 2) and                            # coarser granularity is YEAR or SEASON
        (tl_fine > 2) and                               # finer granularity is DAYTYPE or DAYSLICE
        (not actual_day_no_in_fine)):                   # finer granularity does not have an actual day no
        se = btconc_elem_fine[1]
        dt = btconc_elem_fine[3] if m.day_no_time_elem_reqd else btconc_elem_fine[2]
        return m.NumDaysPerSeason[se] * m.WeightPerDayType[dt]
    elif ((tl_coarse > 2) and                           # coarser granularity is DAYTYPE or DAYSLICE
          (not actual_day_no_in_coarse) and             # coarser granularity does not have an actual day no
          (actual_day_no_in_fine)):                     # finer granularity has an actual day no
        se = btconc_elem_fine[1]
        dt = btconc_elem_fine[3] if m.day_no_time_elem_reqd else btconc_elem_fine[2]
        return 1.0 / (m.NumDaysPerSeason[se] * m.WeightPerDayType[dt])
    else:
        return 1

if (model.num_time_levels_to_use > 1):
    model.NumDaysPerSeason = Param(model.SeasonInp,
                                   initialize = get_season_duration_dict,
                                   within = PositiveIntegers)

if (model.num_time_levels_to_use > 2):
    model.WeightPerDayType = Param(model.DayTypeInp,
                                   initialize = get_daytype_weight_dict,
                                   within = PercentFraction)

if (model.num_time_levels_to_use > 3):
    model.NumHoursPerDaySlice = Param(model.DaySliceInp,
                                      initialize = get_dayslice_duration_dict,
                                      within = PositiveIntegers)

model.EndUseDemand = Param(model.EC_UPTOBTCONC_UPTOBA, 
                           initialize = get_end_use_demand,
                           within = NonNegativeReals, default = 0.0)

model.DerivedTransitCost = Param(model.EC_YR_BA1_BA2,
                                 initialize = calc_derived_transit_cost,
                                 within = NonNegativeReals, default = 0.0)

model.Throughput = Param(model.EC_YR_BA1_BA2,
                         initialize = calc_throughput,
                         within = PercentFraction, default = 1.0)

model.DerivedMaxTransit = Param(model.EC_BTCONC_BA1_BA2,
                                initialize = calc_derived_max_transit,
                                within = Reals, default = 0.0)

model.StorLifetimeYearsInModel = Param(model.InstYear, model.EST_BA,
                                       initialize = get_stor_lifetime_years_in_model_dict,
                                       within = NonNegativeIntegers, default = 0)

model.StorLifetimeCyclesInModel = Param(model.InstYear, model.EST_BA,
                                        initialize = get_stor_lifetime_cycles_in_model_dict,
                                        within = NonNegativeReals, default = 0.0)

model.ECTUnitIOOutputConv = Param([EC_DOM_STR, EC_IMP_STR], model.InstYear, 
                                  model.ECTFILT_YR_BA, 
                                  initialize = get_ect_unit_io_output_conv_factor, 
                                  within = NonNegativeReals, default = 0.0)

######################
#   Model Variables  #
######################

#########                   Cost related                        #############

model.TotalModelCost = Var(within = NonNegativeReals)
model.AnnualCost = Var(model.Year, within = NonNegativeReals)
model.AnnualCarrierCost = Var(model.Year, within = NonNegativeReals)
model.AnnualECTCost = Var(model.Year, within = NonNegativeReals)
model.AnnualStorageCost = Var(model.Year, within = NonNegativeReals)
model.CostOfCarrier = Var(model.EnergyCarrier, model.Year, 
                          within = NonNegativeReals)
model.ECTFixedCost = Var(model.EnergyConvTech, model.Year, 
                         within = NonNegativeReals)
model.CostOfStorage = Var(model.EnergyStorTech, model.Year, 
                          within = NonNegativeReals)
model.ECCostInBT_BA = Var(model.EC_BTCONC_BA, within = NonNegativeReals)
model.PECCostInBT_BA = Var(model.PPEC_BTCONC_BA, within = NonNegativeReals)
model.DomPECCostInBT_BA = Var(model.PPEC_BTCONC_BA, within = NonNegativeReals)
model.ImpPECCostInBT_BA = Var(model.PPEC_BTCONC_BA, within = NonNegativeReals)
model.DECCostInBT_BA = Var(model.DEC_BTCONC_BA, within = NonNegativeReals)
model.ECTInputVarCost = Var(model.EnergyCarrier, model.Year, 
                            within = NonNegativeReals)
model.StorDischargeTransitCost = Var(model.EC_BTCONC_BA, 
                                     within = NonNegativeReals)

#########                   Energy supply, flow, demand related #############
model.UnmetDemand = Var(model.EC_BTCONC_BA, within = NonNegativeReals,
                        initialize = 0.0)
model.DomSupplyFromTo = Var(model.EC_BTCONC_BA1_BA2, within = NonNegativeReals)
model.ImpSupplyFromTo = Var(model.EC_BTCONC_BA1_BA2, within = NonNegativeReals)
model.TotalSupplyFromTo = Var(model.EC_BTCONC_BA1_BA2, 
                              within = NonNegativeReals)
model.ECTInputDomOutputGran = Var(model.IY_ECTFILT_BTCONC_BA_OUTPUT_GRAN, 
                                  within = NonNegativeReals)
model.ECTInputImpOutputGran = Var(model.IY_ECTFILT_BTCONC_BA_OUTPUT_GRAN, 
                                  within = NonNegativeReals)
model.ECTInputDomInputGran = Var(model.IY_ECTFILT_BTCONC_BA_INPUT_GRAN, 
                                 within = NonNegativeReals)
model.ECTInputImpInputGran = Var(model.IY_ECTFILT_BTCONC_BA_INPUT_GRAN, 
                                 within = NonNegativeReals)
model.ECTInputDomCoarserGran = Var(model.IY_ECTFILT_BTCONC_BA_COARSER_GRAN, 
                                   within = NonNegativeReals)
model.ECTInputImpCoarserGran = Var(model.IY_ECTFILT_BTCONC_BA_COARSER_GRAN, 
                                   within = NonNegativeReals)
model.DomSupplyFrom = Var(model.EC_BTCONC_BA, within = NonNegativeReals)
model.ImpSupplyFrom = Var(model.EC_BTCONC_BA, within = NonNegativeReals)
model.DomesticProd = Var(model.EC_BTCONC_BA, within = NonNegativeReals)
model.Import = Var(model.EC_BTCONC_BA, within = NonNegativeReals)
model.SupplyFromECTiy = Var(model.IY_ECT_BTCONC_BA, within = NonNegativeReals)
model.DomStorDischargedFromTo = Var(model.EC_BTCONC_BA1_BA2, 
                                    within = NonNegativeReals)
model.ImpStorDischargedFromTo = Var(model.EC_BTCONC_BA1_BA2, 
                                    within = NonNegativeReals)
model.OutputFromECTiyUsingDomInput = Var(model.IY_ECTFILT_BTCONC_BA, 
                                         within = NonNegativeReals)
model.OutputFromECTiyUsingImpInput = Var(model.IY_ECTFILT_BTCONC_BA, 
                                         within = NonNegativeReals)
model.OutputFromECTiy = Var(model.IY_ECT_BTCONC_BA, within = NonNegativeReals)
model.EndUseDemandMetByDom = Var(model.EC_BTCONC_BA, within = NonNegativeReals)
model.EndUseDemandMetByImp = Var(model.EC_BTCONC_BA, within = NonNegativeReals)
model.EndUseDemandComponents = Var(model.EC_TL_GL_BTCONC_BA, within = NonNegativeReals)
model.EndUseDemandVar = Var(model.EC_BTCONC_BA, within = NonNegativeReals)

#########                   ECT capacity related                #############
model.NamePlateCapacity = Var(model.InstYear, model.ECT_YR_BA, 
                              within = NonNegativeReals)
model.EffectiveCapacity = Var(model.InstYear, model.ECT_YR_BA, 
                              within = NonNegativeReals)
model.CapacityInstalledInYear = Var(model.ECT_YR_BA, within = NonNegativeReals, 
                                    initialize = 0.0)
model.RemainingLegacyCapacity = Var(model.ECT_YR_BA, within = NonNegativeReals)
model.EffectiveCapacityExistingInYear = Var(model.ECT_YR_BA, 
                                            within = NonNegativeReals)

#########                   Storage related                     #############
model.StorCapacityInstalledInYear = Var(model.InstYear, model.EST_BA, 
                                        within = NonNegativeReals, 
                                        initialize = 0.0)
model.StorDischarged = Var(model.IY_EST_BTCONC_BA, within = NonNegativeReals)
model.StorCharged = Var(model.IY_EST_BTCONC_BA, within = NonNegativeReals)
model.StorMaxLifetimeCharge = Var(model.InstYear, model.EST_BA, 
                                  within = NonNegativeReals)
model.StorChargeLevel = Var(model.IY_EST_BTCONC_BA, within = NonNegativeReals)
model.EffectiveStorCapacity = Var(model.InstYear, model.EST_YR_BA, 
                                  within = NonNegativeReals)
model.StorLifetimeCharge = Var(model.IY_EST_BTCONC_BA, within = NonNegativeReals)
model.StorCapacityExistingInYear = Var(model.EST_YR_BA, within = NonNegativeReals)


######################
# Objective Function #
######################

def ObjectiveFunction_rule(m):
    return m.TotalModelCost                 # summation(m.Supply)
model.OBJ = Objective(rule = ObjectiveFunction_rule, sense = minimize)


#####################
# Constraints       #
#####################

#########                   Cost related                        #############

def total_model_cost_rule(m):
    return (m.TotalModelCost == summation(m.AnnualCost))

model.total_model_cost_constraint = Constraint(rule = total_model_cost_rule)

def annual_cost_rule(m, y):
    return (m.AnnualCost[y] == m.AnnualCarrierCost[y] + m.AnnualECTCost[y] + 
                               m.AnnualStorageCost[y])

model.annual_cost_constraint = Constraint(model.Year, rule = annual_cost_rule)

def annual_carrier_cost_rule(m, y):
    return (m.AnnualCarrierCost[y] == sum(m.CostOfCarrier[ec, y] 
                                          for ec in m.EnergyCarrier))

model.annual_carrier_cost_constraint = Constraint(model.Year, 
    rule = annual_carrier_cost_rule)

def annual_ect_cost_rule(m, y):
    return (m.AnnualECTCost[y] == sum(m.ECTFixedCost[ect, y] 
                                      for ect in m.EnergyConvTech))

model.annual_ect_cost_constraint = Constraint(model.Year, 
    rule = annual_ect_cost_rule)

def annual_storage_cost_rule(m, y):
    return (m.AnnualStorageCost[y] == sum(m.CostOfStorage[est, y] 
                                          for est in m.EnergyStorTech))

model.annual_storage_cost_constraint = Constraint(model.Year, 
    rule = annual_storage_cost_rule)

def ect_fixed_cost_rule(m, ect, y):
    bal_area_set = get_bal_area_set(m, get_output_dec(ect))

    return (m.ECTFixedCost[ect, y] == 
            sum(m.NamePlateCapacity[iy, ect, y, ba] * m.FixedCost[iy, ect, y, ba] 
                for iy in m.InstYear if (iy <= y) for ba in bal_area_set))

model.ect_fixed_cost_constraint = Constraint(model.EnergyConvTech, model.Year, 
    rule = ect_fixed_cost_rule)

def cost_of_storage_rule(m, est, y):
    bal_area_set = get_bal_area_set(m, get_stored_ec(est))

    return (m.CostOfStorage[est, y] == 
            sum(m.StorCapacityInstalledInYear[iy, est, ba] * 
                m.StorFixedCost[iy, est, y, ba] 
                for iy in m.InstYear if (iy <= y) for ba in bal_area_set 
                if (y < iy + m.StorLifetimeYearsInModel[iy, est, ba])
               ))

model.cost_of_storage_constraint = Constraint(model.EnergyStorTech, model.Year, 
    rule = cost_of_storage_rule)

def cost_of_carrier_rule(m, ec, y):
    bal_time_conc_set = get_bal_time_conc_set(m, ec)
    bal_area_set = get_bal_area_set(m, ec)

    return (m.CostOfCarrier[ec, y] == 
            m.ECTInputVarCost[ec, y] + 
            sum(m.ECCostInBT_BA[ec, btconc, ba] * get_ec_scale_factor(m, ec, btconc) 
                for btconc in bal_time_conc_set if (btconc[0] == y) 
                for ba in bal_area_set))

model.cost_of_carrier_constraint = Constraint(model.EnergyCarrier, model.Year, 
    rule = cost_of_carrier_rule)

def ec_cost_in_bt_ba_rule(m, ec, *args):            #args = btconc_ba
    return (m.ECCostInBT_BA[ec, args] == 
            (m.PECCostInBT_BA[ec, args] if is_primary(ec) else 
             m.DECCostInBT_BA[ec, args]) + 
            m.StorDischargeTransitCost[ec, args] + 
            m.UnmetDemand[ec, args] * m.CostPerUnmetUnit[ec, args[0]])

model.ec_cost_in_bt_ba_constraint = Constraint(model.EC_BTCONC_BA, 
    rule = ec_cost_in_bt_ba_rule)

def pec_cost_in_bt_ba_rule(m, ec, *args):           #args = btconc_ba
    return (m.PECCostInBT_BA[ec, args] == 
            m.DomPECCostInBT_BA[ec, args] + m.ImpPECCostInBT_BA[ec, args])

model.pec_cost_in_bt_ba_constraint = Constraint(model.PPEC_BTCONC_BA, 
    rule = pec_cost_in_bt_ba_rule)

def dom_pec_cost_in_bt_ba_rule(m, ec, *args):       #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]
    bt = (btconc[0 : 2] + btconc[3 : ]) if m.day_no_time_elem_reqd else btconc
    bal_area_set = get_bal_area_set(m, ec)

    return (m.DomPECCostInBT_BA[ec, args] == 
            sum((m.DomSupplyFromTo[ec, btconc, ba_src, ba] - 
                 m.DomStorDischargedFromTo[ec, btconc, ba_src, ba]) * 
                (m.DomesticPrice[ec, bt, ba_src] * 
                 (1 + m.AVTaxOHDom[ec, bt, ba_src]) + 
                 m.FixedTaxOHDom[ec, bt, ba_src] + 
                 m.DerivedTransitCost[ec, args[0], ba_src, ba]) 
                for ba_src in bal_area_set
               ))

model.dom_pec_cost_in_bt_ba_constraint = Constraint(model.PPEC_BTCONC_BA, 
    rule = dom_pec_cost_in_bt_ba_rule)

def imp_pec_cost_in_bt_ba_rule(m, ec, *args):       #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]
    bt = (btconc[0 : 2] + btconc[3 : ]) if m.day_no_time_elem_reqd else btconc
    bal_area_set = get_bal_area_set(m, ec)

    return (m.ImpPECCostInBT_BA[ec, args] == 
            sum((m.ImpSupplyFromTo[ec, btconc, ba_src, ba] - 
                 m.ImpStorDischargedFromTo[ec, btconc, ba_src, ba]) * 
                (m.ImportPrice[ec, bt, ba_src] * 
                 (1 + m.AVTaxOHImp[ec, bt, ba_src]) + 
                 m.FixedTaxOHImp[ec, bt, ba_src] + 
                 m.DerivedTransitCost[ec, args[0], ba_src, ba]) 
                for ba_src in bal_area_set
               ))

model.imp_pec_cost_in_bt_ba_constraint = Constraint(model.PPEC_BTCONC_BA, 
    rule = imp_pec_cost_in_bt_ba_rule)

def dec_cost_in_bt_ba_rule(m, ec, *args):           #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]
    bt = (btconc[0 : 2] + btconc[3 : ]) if m.day_no_time_elem_reqd else btconc
    bal_area_set = get_bal_area_set(m, ec)

    return (m.DECCostInBT_BA[ec, args] == 
            sum((m.TotalSupplyFromTo[ec, btconc, ba_src, ba] - 
                 m.DomStorDischargedFromTo[ec, btconc, ba_src, ba] - 
                 m.ImpStorDischargedFromTo[ec, btconc, ba_src, ba]) * 
                (m.FixedTaxOHDEC[ec, bt, ba_src] + 
                 m.DerivedTransitCost[ec, args[0], ba_src, ba]) 
                for ba_src in bal_area_set
               ))

model.dec_cost_in_bt_ba_constraint = Constraint(model.DEC_BTCONC_BA, 
    rule = dec_cost_in_bt_ba_rule)

def ect_var_cost_rule(m, ec, y):
    year_time_elem_list = [DUMMY_TIME_STR] * m.num_conc_time_colms
    year_time_elem_list[0] = y
    year_time_elem_tuple = tuple(year_time_elem_list)

    return (m.ECTInputVarCost[ec, y] == 
            sum((m.ECTInputDomOutputGran[iy, ect, btconc_out, ba_out] + 
                 m.ECTInputImpOutputGran[iy, ect, btconc_out, ba_out]) * 
                m.VarCost[iy, ect, btconc_out[0], ba_out] *
                get_time_scale_factor(m, year_time_elem_tuple, btconc_out)
                for ect in m.EnergyConvTechFiltered if (get_input_ec(ect) == ec) 
                for btconc_out in get_bal_time_conc_set(m, get_output_dec(ect)) 
                if (btconc_out[0] == y) 
                for ba_out in get_bal_area_set(m, get_output_dec(ect)) 
                for iy in m.InstYear if (iy <= y)
               ))

model.ect_var_cost_constraint = Constraint(model.EnergyCarrier, model.Year, 
    rule = ect_var_cost_rule)

def stor_discharge_transit_cost_rule(m, ec, *args):         #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]
    bal_area_set = get_bal_area_set(m, ec)

    return (m.StorDischargeTransitCost[ec, args] == 
            sum((m.DomStorDischargedFromTo[ec, btconc, ba_src, ba] + 
                 m.ImpStorDischargedFromTo[ec, btconc, ba_src, ba]) * 
                m.DerivedTransitCost[ec, args[0], ba_src, ba]
                for ba_src in bal_area_set
               ))

model.stor_discharge_transit_cost_constraint = Constraint(model.EC_BTCONC_BA, 
    rule = stor_discharge_transit_cost_rule)

#########                   Energy supply, flow, demand related #############

def dom_supply_from_rule1(m, ec, *args):            #args = btconc_ba
    bal_area_set = get_bal_area_set(m, ec)

    return (m.DomSupplyFrom[ec, args] == 
            sum(m.DomSupplyFromTo[ec, args, ba_dest]
                for ba_dest in bal_area_set))

model.dom_supply_from_constraint1 = Constraint(model.EC_BTCONC_BA, 
    rule = dom_supply_from_rule1)

def imp_supply_from_rule1(m, ec, *args):            #args = btconc_ba
    bal_area_set = get_bal_area_set(m, ec)

    return (m.ImpSupplyFrom[ec, args] == 
            sum(m.ImpSupplyFromTo[ec, args, ba_dest]
                for ba_dest in bal_area_set))

model.imp_supply_from_constraint1 = Constraint(model.EC_BTCONC_BA, 
    rule = imp_supply_from_rule1)

def total_supply_from_to_rule1(m, ec, *args):       #args = btconc_ba1_ba2
    return (m.TotalSupplyFromTo[ec, args] == 
            m.DomSupplyFromTo[ec, args] + m.ImpSupplyFromTo[ec, args])

model.total_supply_from_to_constraint1 = Constraint(model.EC_BTCONC_BA1_BA2, 
    rule = total_supply_from_to_rule1)

def domestic_prod_rule(m, ec, *args):               #args = btconc_ba
    if (ec in m.EnergyCarrierPrimaryPhys):
        arg_p = (args[0 : 2] + args[3 : ]) if m.day_no_time_elem_reqd else args
        return (m.DomesticProd[ec, args] <= m.MaxDomesticProd[ec, arg_p] * 
                                            (1 - m.NonEnergyShare[ec, arg_p]))
    else:
        return (m.DomesticProd[ec, args] == 
                sum(m.SupplyFromECTiy[iy, ect, args]
                    for ect in m.EnergyConvTech if (get_output_dec(ect) == ec) 
                    for iy in m.InstYear if iy <= args[0]
                   )
               )

model.domestic_prod_constraint = Constraint(model.EC_BTCONC_BA, 
    rule = domestic_prod_rule)

def import_rule(m, ec, *args):                      #args = btconc_ba
    if (ec in m.EnergyCarrierPrimaryPhys):
        arg_p = (args[0 : 2] + args[3 : ]) if m.day_no_time_elem_reqd else args
        return (m.Import[ec, args] <= m.MaxImport[ec, arg_p])
    else:
        return (m.Import[ec, args] == 0)

model.import_constraint = Constraint(model.EC_BTCONC_BA, rule = import_rule)

def dom_supply_from_rule2(m, ec, *args):            #args = btconc_ba
    bal_area_set = get_bal_area_set(m, ec)

    return (m.DomSupplyFrom[ec, args] == 
            m.DomesticProd[ec, args] + 
            sum(m.DomStorDischargedFromTo[ec, args, ba_dest]
                for ba_dest in bal_area_set))

model.dom_supply_from_constraint2 = Constraint(model.EC_BTCONC_BA, 
    rule = dom_supply_from_rule2)

def imp_supply_from_rule2(m, ec, *args):            #args = btconc_ba
    bal_area_set = get_bal_area_set(m, ec)

    return (m.ImpSupplyFrom[ec, args] == 
            m.Import[ec, args] + 
            sum(m.ImpStorDischargedFromTo[ec, args, ba_dest]
                for ba_dest in bal_area_set))

model.imp_supply_from_constraint2 = Constraint(model.EC_BTCONC_BA, 
    rule = imp_supply_from_rule2)

def total_supply_from_to_rule2(m, ec, *args):       #args = btconc_ba1_ba2
    if (m.DerivedMaxTransit[ec, args] >= 0):
        return (m.TotalSupplyFromTo[ec, args] <= m.DerivedMaxTransit[ec, args])
    else:
        return Constraint.Skip

model.total_supply_from_to_constraint2 = Constraint(model.EC_BTCONC_BA1_BA2, 
    rule = total_supply_from_to_rule2)

def supply_from_ect_iy_rule1(m, iy, ect, *args):    #args = btconc_ba
    return (m.SupplyFromECTiy[iy, ect, args] == 
            (m.OutputFromECTiyUsingDomInput[iy, ect, args] + 
             m.OutputFromECTiyUsingImpInput[iy, ect, args]) * 
            (1 - m.AuxCons[ect, iy]) 
           )

model.supply_from_ect_iy_constraint1 = Constraint(model.IY_ECTFILT_BTCONC_BA, 
    rule = supply_from_ect_iy_rule1)

def output_from_ect_iy_using_dom_input_rule(m, iy, ect, *args):     #args = btconc_ba
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    return (m.OutputFromECTiyUsingDomInput[iy, ect, args] == 
            m.ECTInputDomOutputGran[iy, ect, args] * 
            m.ECTUnitIOOutputConv[EC_DOM_STR, iy, ect, args[0], ba])

model.output_from_ect_iy_using_dom_input_constraint = Constraint(
    model.IY_ECTFILT_BTCONC_BA, rule = output_from_ect_iy_using_dom_input_rule)

def output_from_ect_iy_using_imp_input_rule(m, iy, ect, *args):     #args = btconc_ba
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    return (m.OutputFromECTiyUsingImpInput[iy, ect, args] == 
            m.ECTInputImpOutputGran[iy, ect, args] * 
            m.ECTUnitIOOutputConv[EC_IMP_STR, iy, ect, args[0], ba])

model.output_from_ect_iy_using_imp_input_constraint = Constraint(
    model.IY_ECTFILT_BTCONC_BA, rule = output_from_ect_iy_using_imp_input_rule)

def output_from_ect_iy_rule(m, iy, ect, *args):     #args = btconc_ba
    return (m.OutputFromECTiy[iy, ect, args] == 
            m.SupplyFromECTiy[iy, ect, args] / (1 - m.AuxCons[ect, iy])
           )

model.output_from_ect_iy_constraint = Constraint(model.IY_ECT_BTCONC_BA, 
    rule = output_from_ect_iy_rule)

def max_annual_uf_rule(m, iy, ect, *args):          #args = yr_ba
    return ( sum(m.OutputFromECTiy[iy, ect, btconc, args[1:]]
                 for btconc in get_bal_time_conc_set(m, get_output_dec(ect)) 
                 if (btconc[0] == args[0])
                )
             <= 
             m.MaxAnnualUF[iy, ect, args] * m.EffectiveCapacity[iy, ect, args] * 
             m.AnnualOutputPerUnitCapacity[ect]
           )

model.max_annual_uf_constraint = Constraint(model.IY_ECT_YR_BA, 
    rule = max_annual_uf_rule)

def ect_ramp_down_rule1(m, iy, ect, *args):         #args = btconc_ba
    ec_out = get_output_dec(ect)
    ec_out_bt_inp = get_bal_time_inp(ec_out)

    if ((ec_out_bt_inp == BALTIME_SE_STR) or 
        ((ec_out_bt_inp in [BALTIME_DT_STR, BALTIME_DS_STR]) and 
         (len(m.DayTypeInp) == 1))
       ):
        btconc = args[0 : m.num_conc_time_colms]
        ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

        btconc_prev = prev_conc_time_elem(m, ec_out, btconc)

        if ((btconc_prev is None) or (btconc_prev[0] < iy)):
            return Constraint.Skip
        else:
            return (m.OutputFromECTiy[iy, ect, btconc_prev, ba] / 
                    (m.AnnualOutputPerUnitCapacity[ect] * 
                     share_of_bt_in_year(m, ec_out, btconc_prev)
                    )
                    -
                    m.EffectiveCapacity[iy, ect, args[0], ba] * 
                    (m.MaxRampDownRate[ect, iy] * 
                     (1 if (ec_out_bt_inp != BALTIME_DS_STR) else 
                      num_hours_conc_time_elem(m, ec_out, btconc))
                    )
                    <=
                    m.OutputFromECTiy[iy, ect, args] / 
                    (m.AnnualOutputPerUnitCapacity[ect] * 
                     share_of_bt_in_year(m, ec_out, btconc)
                    )
                   )
    else:
        return Constraint.Skip

model.ect_ramp_down_constraint1 = Constraint(model.IY_ECT_BTCONC_BA, 
    rule = ect_ramp_down_rule1)

def ect_ramp_up_rule1(m, iy, ect, *args):           #args = btconc_ba
    ec_out = get_output_dec(ect)
    ec_out_bt_inp = get_bal_time_inp(ec_out)

    if ((ec_out_bt_inp == BALTIME_SE_STR) or 
        ((ec_out_bt_inp in [BALTIME_DT_STR, BALTIME_DS_STR]) and 
         (len(m.DayTypeInp) == 1))
       ):
        btconc = args[0 : m.num_conc_time_colms]
        ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

        btconc_prev = prev_conc_time_elem(m, ec_out, btconc)

        if ((btconc_prev is None) or (btconc_prev[0] < iy)):
            return Constraint.Skip
        else:
            return (m.OutputFromECTiy[iy, ect, args] / 
                    (m.AnnualOutputPerUnitCapacity[ect] * 
                     share_of_bt_in_year(m, ec_out, btconc)
                    )
                    <=
                    m.OutputFromECTiy[iy, ect, btconc_prev, ba] / 
                    (m.AnnualOutputPerUnitCapacity[ect] * 
                     share_of_bt_in_year(m, ec_out, btconc_prev)
                    )
                    +
                    m.EffectiveCapacity[iy, ect, args[0], ba] * 
                    (m.MaxRampUpRate[ect, iy] * 
                     (1 if (ec_out_bt_inp != BALTIME_DS_STR) else 
                      num_hours_conc_time_elem(m, ec_out, btconc))
                    )
                   )
    else:
        return Constraint.Skip

model.ect_ramp_up_constraint1 = Constraint(model.IY_ECT_BTCONC_BA, 
    rule = ect_ramp_up_rule1)

def ect_ramp_down_rule2(m, iy, ect, *args):         #args = btconc_ba
    ec_out = get_output_dec(ect)
    ec_out_bt_inp = get_bal_time_inp(ec_out)

    if ((ec_out_bt_inp == BALTIME_DS_STR) and (len(m.DayTypeInp) == 1) and 
        (not m.ec_day_no_reqd_map[ec_out])
       ):
        ds = args[4] if m.day_no_time_elem_reqd else args[3]

        if (ds == get_first_daysliceinp(m)):
            btconc = args[0 : m.num_conc_time_colms]
            ba = args[m.num_conc_time_colms : 
                      m.num_conc_time_colms + m.num_geog_colms]

            last_ds = get_last_daysliceinp(m)

            btconc_last_ds = btconc[0 : (4 if m.day_no_time_elem_reqd 
                                         else 3)] + (last_ds, )

            return (m.OutputFromECTiy[iy, ect, btconc_last_ds, ba] / 
                    (m.AnnualOutputPerUnitCapacity[ect] * 
                     share_of_bt_in_year(m, ec_out, btconc_last_ds)
                    )
                    -
                    m.EffectiveCapacity[iy, ect, args[0], ba] * 
                    (m.MaxRampDownRate[ect, iy] * 
                     num_hours_conc_time_elem(m, ec_out, btconc)
                    )
                    <=
                    m.OutputFromECTiy[iy, ect, args] / 
                    (m.AnnualOutputPerUnitCapacity[ect] * 
                     share_of_bt_in_year(m, ec_out, btconc)
                    )
                   )
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip

model.ect_ramp_down_constraint2 = Constraint(model.IY_ECT_BTCONC_BA, 
    rule = ect_ramp_down_rule2)

def ect_ramp_up_rule2(m, iy, ect, *args):           #args = btconc_ba
    ec_out = get_output_dec(ect)
    ec_out_bt_inp = get_bal_time_inp(ec_out)

    if ((ec_out_bt_inp == BALTIME_DS_STR) and (len(m.DayTypeInp) == 1) and 
        (not m.ec_day_no_reqd_map[ec_out])
       ):
        ds = args[4] if m.day_no_time_elem_reqd else args[3]

        if (ds == get_first_daysliceinp(m)):
            btconc = args[0 : m.num_conc_time_colms]
            ba = args[m.num_conc_time_colms : 
                      m.num_conc_time_colms + m.num_geog_colms]

            last_ds = get_last_daysliceinp(m)

            btconc_last_ds = btconc[0 : (4 if m.day_no_time_elem_reqd 
                                         else 3)] + (last_ds, )

            return (m.OutputFromECTiy[iy, ect, args] / 
                    (m.AnnualOutputPerUnitCapacity[ect] * 
                     share_of_bt_in_year(m, ec_out, btconc)
                    )
                    <=
                    m.OutputFromECTiy[iy, ect, btconc_last_ds, ba] / 
                    (m.AnnualOutputPerUnitCapacity[ect] * 
                     share_of_bt_in_year(m, ec_out, btconc_last_ds)
                    )
                    +
                    m.EffectiveCapacity[iy, ect, args[0], ba] * 
                    (m.MaxRampUpRate[ect, iy] * 
                     num_hours_conc_time_elem(m, ec_out, btconc)
                    )
                   )
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip

model.ect_ramp_up_constraint2 = Constraint(model.IY_ECT_BTCONC_BA, 
    rule = ect_ramp_up_rule2)

def supply_from_ect_iy_rule2(m, iy, ect, *args):    #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]
    arg_p = (args[0 : 2] + args[3 : ]) if m.day_no_time_elem_reqd else args

    return (m.SupplyFromECTiy[iy, ect, args] <= 
            m.EffectiveCapacity[iy, ect, args[0], ba] * 
            (m.AnnualOutputPerUnitCapacity[ect] * 
             share_of_bt_in_year(m, get_output_dec(ect), btconc)) * 
            m.MaxUF[iy, ect, arg_p] * (1 - m.AuxCons[ect, iy])
           )

model.supply_from_ect_iy_constraint2 = Constraint(model.IY_ECT_BTCONC_BA, 
    rule = supply_from_ect_iy_rule2)

def end_use_demand_rule(m, ec, *args):              #args = btconc_ba
    yr = args[0]

    return (m.EndUseDemandVar[ec, args] == 
            m.UnmetDemand[ec, args] + m.EndUseDemandMetByDom[ec, args] + 
            m.EndUseDemandMetByImp[ec, args] * 
            ((m.ImpEnergyDensity[ec, yr] / m.DomEnergyDensity[ec, yr]) 
             if (ec in m.EnergyCarrierPrimaryPhys) else 1)
           )

model.end_use_demand_constraint = Constraint(model.EC_BTCONC_BA, 
    rule = end_use_demand_rule)

def dom_consumption_rule(m, ec, *args):             #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]
    bal_area_set = get_bal_area_set(m, ec)
    
    return (
            sum(m.DomSupplyFromTo[ec, btconc, ba_src, ba] * 
                m.Throughput[ec, args[0], ba_src, ba]
                for ba_src in bal_area_set)
            == 
            m.EndUseDemandMetByDom[ec, args] + 
            sum(m.ECTInputDomInputGran[iy, ect, btconc, ba]
                for ect in m.EnergyConvTechFiltered if (get_input_ec(ect) == ec) 
                for iy in m.InstYear if (iy <= btconc[0])
               ) 
            + 
            sum(m.StorCharged[iy, est, args] / 
                m.StorEfficiency[iy, est, args[0], ba] 
                for est in m.EnergyStorTech 
                if ((get_stored_ec(est) == ec) and 
                    (m.DomOrImp[est] == EC_DOM_STR))
                for iy in m.InstYear 
                if ((iy <= args[0]) and 
                    (args[0] < iy + m.StorLifetimeYearsInModel[iy, est, ba])) 
               )
           )

model.dom_consumption_constraint = Constraint(model.EC_BTCONC_BA, 
    rule = dom_consumption_rule)

def imp_consumption_rule(m, ec, *args):             #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]
    bal_area_set = get_bal_area_set(m, ec)
    
    return (
            sum(m.ImpSupplyFromTo[ec, btconc, ba_src, ba] * 
                m.Throughput[ec, args[0], ba_src, ba]
                for ba_src in bal_area_set)
            == 
            m.EndUseDemandMetByImp[ec, args] + 
            sum(m.ECTInputImpInputGran[iy, ect, btconc, ba]
                for ect in m.EnergyConvTechFiltered if (get_input_ec(ect) == ec) 
                for iy in m.InstYear if (iy <= btconc[0])
               ) 
            + 
            sum(m.StorCharged[iy, est, args] / 
                m.StorEfficiency[iy, est, args[0], ba] 
                for est in m.EnergyStorTech 
                if ((get_stored_ec(est) == ec) and 
                    (m.DomOrImp[est] == EC_IMP_STR))
                for iy in m.InstYear 
                if ((iy <= args[0]) and 
                    (args[0] < iy + m.StorLifetimeYearsInModel[iy, est, ba])) 
               )
           )

model.imp_consumption_constraint = Constraint(model.EC_BTCONC_BA, 
    rule = imp_consumption_rule)

#########                   ECT capacity related                #############

def name_plate_capacity_rule(m, iy, ect, *args):            #args = yr_ba
    if (iy == (m.StartYear - 1)):
        return (m.NamePlateCapacity[iy, ect, args] == 
                m.RemainingLegacyCapacity[ect, args])
    else:
        if ((iy <= args[0]) and (args[0] < iy + m.Lifetime[ect, iy])):
            return (m.NamePlateCapacity[iy, ect, args] == 
                    m.CapacityInstalledInYear[ect, iy, args[1:]])
        else:
            return (m.NamePlateCapacity[iy, ect, args] == 0)

model.name_plate_capacity_constraint = Constraint(model.InstYear,
    model.ECT_YR_BA, rule = name_plate_capacity_rule)

def effective_capacity_rule(m, iy, ect, *args):             #args = yr_ba
    return (m.EffectiveCapacity[iy, ect, args] == 
            m.NamePlateCapacity[iy, ect, args] * 
            ((1 - m.CapacityDerating[ect, iy]) ** 
             get_current_age_in_years(m, args[0], iy))
           )

model.effective_capacity_constraint = Constraint(model.InstYear,
    model.ECT_YR_BA, rule = effective_capacity_rule)

def capacity_installed_in_year_rule1(m, ect, *args):        #args = yr_ba
    return (m.CapacityInstalledInYear[ect, args] >= 
            m.MinCapacity[ect, args])

model.capacity_installed_in_year_constraint1 = Constraint(model.ECT_YR_BA, 
    rule = capacity_installed_in_year_rule1)

def capacity_installed_in_year_rule2(m, ect, *args):        #args = yr_ba
    if (m.MaxCapacity[ect, args] >= 0):
        return (m.CapacityInstalledInYear[ect, args] <= 
                m.MaxCapacity[ect, args])
    else:
        return Constraint.Skip

model.capacity_installed_in_year_constraint2 = Constraint(model.ECT_YR_BA, 
    rule = capacity_installed_in_year_rule2)

def remaining_legacy_capacity_rule(m, ect, *args):          #args = yr_ba
    if (args[0] == m.StartYear):
        return (m.RemainingLegacyCapacity[ect, args] == 
                m.LegacyCapacity[ect, args[1:]])
    else:
        return (m.RemainingLegacyCapacity[ect, args] == 
                m.RemainingLegacyCapacity[ect, args[0] - 1, args[1:]] - 
                m.LegacyRetirement[ect, args[0] - 1, args[1:]])

model.remaining_legacy_capacity_constraint = Constraint(model.ECT_YR_BA, 
    rule = remaining_legacy_capacity_rule)

def effective_capacity_existing_in_year_rule(m, ect, *args):    #args = yr_ba
    return (m.EffectiveCapacityExistingInYear[ect, args] == 
            sum(m.EffectiveCapacity[iy, ect, args]
                for iy in m.InstYear if iy <= args[0])
           )

model.effective_capacity_existing_in_year_constraint = Constraint(
    model.ECT_YR_BA, rule = effective_capacity_existing_in_year_rule)

#########                   Storage related                     #############

def stor_max_lifetime_charge_rule(m, iy, est, *args):       #args = ba
    return (m.StorMaxLifetimeCharge[iy, est, args] == 
            m.StorCapacityInstalledInYear[iy, est, args] * 
            m.StorLifetimeCyclesInModel[iy, est, args])

model.stor_max_lifetime_charge_constraint = Constraint(model.InstYear,
    model.EST_BA, rule = stor_max_lifetime_charge_rule)

def stor_charge_level_rule1(m, iy, est, *args):             #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    if (is_time_to_reset_stor(m, est, btconc)):
        return (m.StorChargeLevel[iy, est, args] == 
                m.StorCharged[iy, est, args] - m.StorDischarged[iy, est, args])
    else:
        btconc_prev = prev_conc_time_elem(m, get_stored_ec(est), btconc)

        if ((btconc_prev is None) or (btconc_prev[0] < iy)):
            return (m.StorChargeLevel[iy, est, args] == 
                    m.StorCharged[iy, est, args] - 
                    m.StorDischarged[iy, est, args])
        else:
            return (m.StorChargeLevel[iy, est, args] == 
                    m.StorChargeLevel[iy, est, btconc_prev, ba] + 
                    m.StorCharged[iy, est, args] - 
                    m.StorDischarged[iy, est, args])

model.stor_charge_level_constraint1 = Constraint(model.IY_EST_BTCONC_BA, 
    rule = stor_charge_level_rule1)

def stor_charge_level_rule2(m, iy, est, *args):             #args = btconc_ba
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    return (m.StorChargeLevel[iy, est, args] <= 
            m.EffectiveStorCapacity[iy, est, args[0], ba] * 
            m.DepthOfDischarge[est, iy])

model.stor_charge_level_constraint2 = Constraint(model.IY_EST_BTCONC_BA, 
    rule = stor_charge_level_rule2)

def stor_charged_rule(m, iy, est, *args):                   #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    return (m.StorCharged[iy, est, args] <= 
            m.EffectiveStorCapacity[iy, est, args[0], ba] * 
            m.MaxChargeRate[est] * 
            num_hours_conc_time_elem(m, get_stored_ec(est), btconc))

model.stor_charged_constraint = Constraint(model.IY_EST_BTCONC_BA, 
    rule = stor_charged_rule)

def stor_discharged_rule1(m, iy, est, *args):               #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    return (m.StorDischarged[iy, est, args] <= 
            m.EffectiveStorCapacity[iy, est, args[0], ba] * 
            m.MaxDischargeRate[est] * 
            num_hours_conc_time_elem(m, get_stored_ec(est), btconc))

model.stor_discharged_constraint1 = Constraint(model.IY_EST_BTCONC_BA, 
    rule = stor_discharged_rule1)

def stor_discharged_rule2(m, iy, est, *args):               #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]

    if (is_time_to_reset_stor(m, est, btconc)):
        return (m.StorDischarged[iy, est, args] == 0)
    else:
        btconc_prev = prev_conc_time_elem(m, get_stored_ec(est), btconc)

        if ((btconc_prev is None) or (btconc_prev[0] < iy)):
            return (m.StorDischarged[iy, est, args] == 0)
        else:
            return Constraint.Skip

model.stor_discharged_constraint2 = Constraint(model.IY_EST_BTCONC_BA, 
    rule = stor_discharged_rule2)

def stor_lifetime_charge_rule1(m, iy, est, *args):          #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]
    ec = get_stored_ec(est)

    btconc_prev = prev_conc_time_elem(m, ec, btconc)

    if ((btconc_prev is None) or (btconc_prev[0] < iy)):
        return (m.StorLifetimeCharge[iy, est, args] == 
                m.StorCharged[iy, est, args] * 
                get_ec_scale_factor(m, ec, btconc))
    else:
        return (m.StorLifetimeCharge[iy, est, args] == 
                m.StorLifetimeCharge[iy, est, btconc_prev, ba] + 
                m.StorCharged[iy, est, args] * 
                get_ec_scale_factor(m, ec, btconc))

model.stor_lifetime_charge_constraint1 = Constraint(model.IY_EST_BTCONC_BA, 
    rule = stor_lifetime_charge_rule1)

def stor_lifetime_charge_rule2(m, iy, est, *args):          #args = btconc_ba
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    return (m.StorLifetimeCharge[iy, est, args] <= 
            m.StorMaxLifetimeCharge[iy, est, ba])

model.stor_lifetime_charge_constraint2 = Constraint(model.IY_EST_BTCONC_BA, 
    rule = stor_lifetime_charge_rule2)

def stor_capacity_installed_in_year_rule1(m, y, est, *args):    #args = ba
    return (m.StorCapacityInstalledInYear[y, est, args] >= 
            m.MinStorCapacity[est, y, args])

model.stor_capacity_installed_in_year_constraint1 = Constraint(model.Year, 
    model.EST_BA, rule = stor_capacity_installed_in_year_rule1)

def stor_capacity_installed_in_year_rule2(m, y, est, *args):    #args = ba
    if (m.MaxStorCapacity[est, y, args] >= 0):
        return (m.StorCapacityInstalledInYear[y, est, args] <= 
                m.MaxStorCapacity[est, y, args])
    else:
        return Constraint.Skip

model.stor_capacity_installed_in_year_constraint2 = Constraint(model.Year, 
    model.EST_BA, rule = stor_capacity_installed_in_year_rule2)

def stor_capacity_installed_in_year_rule3(m, y, est, *args):    #args = ba
    return (m.StorCapacityInstalledInYear[y, est, args] == 
            m.LegacyStorCapacity[est, args])

model.stor_capacity_installed_in_year_constraint3 = Constraint(
    [model.StartYear - 1], model.EST_BA,
    rule = stor_capacity_installed_in_year_rule3)

def effective_stor_capacity_rule(m, iy, est, *args):        #args = yr_ba
    if ((iy <= args[0]) and 
        (args[0] < iy + m.StorLifetimeYearsInModel[iy, est, args[1:]])):
        return (m.EffectiveStorCapacity[iy, est, args] == 
                m.StorCapacityInstalledInYear[iy, est, args[1:]] * 
                ((1 - m.StorCapacityDerating[est, iy]) ** 
                 get_current_age_in_years(m, args[0], iy))
               )
    else:
        return (m.EffectiveStorCapacity[iy, est, args] == 0)

model.effective_stor_capacity_constraint = Constraint(model.InstYear,
    model.EST_YR_BA, rule = effective_stor_capacity_rule)

def stor_capacity_existing_in_year_rule(m, est, *args):     #args = yr_ba
    return (m.StorCapacityExistingInYear[est, args] == 
            sum(m.EffectiveStorCapacity[iy, est, args]
                for iy in m.InstYear 
                if ((iy <= args[0]) and 
                    (args[0] < iy + m.StorLifetimeYearsInModel[iy, est, args[1:]])) 
               )
           )

model.stor_capacity_existing_in_year_constraint = Constraint(model.EST_YR_BA, 
    rule = stor_capacity_existing_in_year_rule)

def dom_stor_discharged_from_rule(m, ec, *args):            #args = btconc_ba
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]
    bal_area_set = get_bal_area_set(m, ec)

    return (sum(m.DomStorDischargedFromTo[ec, args, ba_dest]
                for ba_dest in bal_area_set
               )
            ==
            sum(m.StorDischarged[iy, est, args]
                for est in m.EnergyStorTech 
                if ((get_stored_ec(est) == ec) and 
                    (m.DomOrImp[est] == EC_DOM_STR))
                for iy in m.InstYear 
                if ((iy <= args[0]) and 
                    (args[0] < iy + m.StorLifetimeYearsInModel[iy, est, ba])) 
               )
           )

model.dom_stor_discharged_from_constraint = Constraint(model.EC_BTCONC_BA, 
    rule = dom_stor_discharged_from_rule)

def imp_stor_discharged_from_rule(m, ec, *args):            #args = btconc_ba
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]
    bal_area_set = get_bal_area_set(m, ec)

    return (sum(m.ImpStorDischargedFromTo[ec, args, ba_dest]
                for ba_dest in bal_area_set
               )
            ==
            sum(m.StorDischarged[iy, est, args]
                for est in m.EnergyStorTech 
                if ((get_stored_ec(est) == ec) and 
                    (m.DomOrImp[est] == EC_IMP_STR))
                for iy in m.InstYear 
                if ((iy <= args[0]) and 
                    (args[0] < iy + m.StorLifetimeYearsInModel[iy, est, ba])) 
               )
           )

model.imp_stor_discharged_from_constraint = Constraint(model.EC_BTCONC_BA, 
    rule = imp_stor_discharged_from_rule)

#########                   EndUse related                      #############

def end_use_demand_components_rule1(m, ec, *args):          #args = btconc_ba
    btconc = args[0 : m.num_conc_time_colms]
    ba = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    tl = get_time_level(m, btconc)
    gl = get_geog_level(m, ba)
    
    return (m.EndUseDemand[ec, args] ==
            sum((m.EndUseDemandComponents[ec, tl, gl, btconc_component, ba_component] * 
                 (1 if (tl > 2) else get_ec_scale_factor(m, ec, btconc_component)))
                for btconc_component in get_bal_time_conc_set(m, ec) 
                if is_contained_in_time(m, btconc, btconc_component) 
                for ba_component in get_bal_area_set(m, ec) 
                if is_contained_in_geog(m, ba, ba_component)
               )
           )

model.end_use_demand_components_constraint1 = Constraint(model.EC_UPTOBTCONC_UPTOBA, 
    rule = end_use_demand_components_rule1)

def end_use_demand_components_rule2(m, ec, *args):          #args = btconc_ba
    return (m.EndUseDemandVar[ec, args] ==
            sum(m.EndUseDemandComponents[ec, tl, gl, args] 
                for tl in get_time_levels_set(ec)
                for gl in get_geog_levels_set(ec)
               )
           )

model.end_use_demand_var_constraint2 = Constraint(model.EC_BTCONC_BA, 
    rule = end_use_demand_components_rule2)

#########                   ECT related                         #############

def ect_input_dom_rule1(m, iy, ect, *args):                 #args = btconc_ba_coarser_gran
    btconc_coarser_gran = args[0 : m.num_conc_time_colms]
    ba_coarser_gran = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    return (m.ECTInputDomCoarserGran[iy, ect, args] ==
            sum((m.ECTInputDomOutputGran[iy, ect, btconc_output_gran, ba_output_gran] * 
                 get_time_scale_factor(m, btconc_coarser_gran, btconc_output_gran))
                for btconc_output_gran in get_bal_time_conc_set(m, get_output_dec(ect)) 
                if is_contained_in_time(m, btconc_coarser_gran, btconc_output_gran) 
                for ba_output_gran in get_bal_area_set(m, get_output_dec(ect)) 
                if is_contained_in_geog(m, ba_coarser_gran, ba_output_gran)
               )
           )

model.ect_input_dom_constraint1 = Constraint(model.IY_ECTFILT_BTCONC_BA_COARSER_GRAN, 
    rule = ect_input_dom_rule1)

def ect_input_imp_rule1(m, iy, ect, *args):                 #args = btconc_ba_coarser_gran
    btconc_coarser_gran = args[0 : m.num_conc_time_colms]
    ba_coarser_gran = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    return (m.ECTInputImpCoarserGran[iy, ect, args] ==
            sum((m.ECTInputImpOutputGran[iy, ect, btconc_output_gran, ba_output_gran] * 
                 get_time_scale_factor(m, btconc_coarser_gran, btconc_output_gran))
                for btconc_output_gran in get_bal_time_conc_set(m, get_output_dec(ect)) 
                if is_contained_in_time(m, btconc_coarser_gran, btconc_output_gran) 
                for ba_output_gran in get_bal_area_set(m, get_output_dec(ect)) 
                if is_contained_in_geog(m, ba_coarser_gran, ba_output_gran)
               )
           )

model.ect_input_imp_constraint1 = Constraint(model.IY_ECTFILT_BTCONC_BA_COARSER_GRAN, 
    rule = ect_input_imp_rule1)

def ect_input_dom_rule2(m, iy, ect, *args):                 #args = btconc_ba_coarser_gran
    btconc_coarser_gran = args[0 : m.num_conc_time_colms]
    ba_coarser_gran = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    return (m.ECTInputDomCoarserGran[iy, ect, args] ==
            sum((m.ECTInputDomInputGran[iy, ect, btconc_input_gran, ba_input_gran] *
                 get_time_scale_factor(m, btconc_coarser_gran, btconc_input_gran))
                for btconc_input_gran in get_bal_time_conc_set(m, get_input_ec(ect)) 
                if is_contained_in_time(m, btconc_coarser_gran, btconc_input_gran) 
                for ba_input_gran in get_bal_area_set(m, get_input_ec(ect)) 
                if is_contained_in_geog(m, ba_coarser_gran, ba_input_gran)
               )
           )

model.ect_input_dom_constraint2 = Constraint(model.IY_ECTFILT_BTCONC_BA_COARSER_GRAN, 
    rule = ect_input_dom_rule2)

def ect_input_imp_rule2(m, iy, ect, *args):                 #args = btconc_ba_coarser_gran
    btconc_coarser_gran = args[0 : m.num_conc_time_colms]
    ba_coarser_gran = args[m.num_conc_time_colms : m.num_conc_time_colms + m.num_geog_colms]

    return (m.ECTInputImpCoarserGran[iy, ect, args] ==
            sum((m.ECTInputImpInputGran[iy, ect, btconc_input_gran, ba_input_gran] *
                 get_time_scale_factor(m, btconc_coarser_gran, btconc_input_gran))
                for btconc_input_gran in get_bal_time_conc_set(m, get_input_ec(ect)) 
                if is_contained_in_time(m, btconc_coarser_gran, btconc_input_gran) 
                for ba_input_gran in get_bal_area_set(m, get_input_ec(ect)) 
                if is_contained_in_geog(m, ba_coarser_gran, ba_input_gran)
               )
           )

model.ect_input_imp_constraint2 = Constraint(model.IY_ECTFILT_BTCONC_BA_COARSER_GRAN, 
    rule = ect_input_imp_rule2)

#########                   User specified                      #############

constraint_tuples_list = []

def user_constraint_rule(m, num_tuples):
    ret_expr = 0

    for tokens_tuple in constraint_tuples_list[: -1]:
        num_tokens = len(tokens_tuple)

        var = getattr(m, tokens_tuple[0])

        var_index = (tokens_tuple[1:-1] if (num_tokens >= 3) else None)

        ret_expr += tokens_tuple[-1] * var[var_index]

    tokens_tuple = constraint_tuples_list[-1]
    lower_bound = tokens_tuple[0]
    upper_bound = tokens_tuple[1]

    return (lower_bound, ret_expr, upper_bound)

                         #########################################
#####################            End of Model Definition            #############
                         #########################################

if isinstance(model, AbstractModel):
    print("Creating instance ...")
    logger.info("Before create instance")
    instance = model.create_instance()
    logger.info("After create instance")
    print("Instance created")
else:
    logger.info("Equating instance to model")
    instance = model

'''    DEBUG prints to console - hence commented out 
print_common_params_data(instance)
print_supply_params_data(instance)

print("ec_est_list_map: ")
print(model.ec_est_list_map)
print("ec_day_no_reqd_map: ")
print(model.ec_day_no_reqd_map)
print("day_no_time_elem_reqd: ", model.day_no_time_elem_reqd)

instance.EnergyConvTechFiltered.pprint()

#instance.pprint()
'''

if isinstance(model, AbstractModel):
    delete_supply_params_data(model)

delete_supply_params_data(instance)

#### Validate and add user specified constraints, if any
if (supply.validate_param(USERCONSTRAINTS_PARAM_NAME, model = instance) == False):
    print(f"ERROR: {USERCONSTRAINTS_PARAM_NAME} parameter validation failed")
    script_exit(-1)

uc_dict_list = supply.get_filtered_parameter(USERCONSTRAINTS_PARAM_NAME, model = instance)

if (uc_dict_list is not None):
    num_uc = len(uc_dict_list)
    print(f"{num_uc} user constraints specified")

    for index, uc_dict in enumerate(uc_dict_list):
        constraint_tuples_list = uc_dict.get(CONSTRAINT_DICT_VECTORS_KEY)

        bounds_tuple = uc_dict.get(CONSTRAINT_DICT_BOUNDS_KEY)

        constraint_tuples_list.append(bounds_tuple)

        constraint_name = USER_CONSTRAINT_NAME_PREFIX + str(index + 1)
        index_set = {len(constraint_tuples_list)}

        setattr(instance, constraint_name, Constraint(index_set, rule = user_constraint_rule))

        print("Added User Constraint :", constraint_name)
        # getattr(instance, constraint_name).pprint()

#### Create LP file for the instance and call the solver
if (solver_executable is not None):
    opt = SolverFactory(solver_name, executable = solver_executable)
else:
    opt = SolverFactory(solver_name)

for option_name, option_value in solver_options_dict.items():
    opt.options[option_name] = option_value

logger.info("Before calling solver")

if (symbolic_solver_labels):
    print("Creating LP file (with symbolic solver labels) and calling solver")
    results = opt.solve(instance, tee = True, symbolic_solver_labels = True)
else:
    print("Creating LP file and calling solver")
    results = opt.solve(instance, tee = True)

logger.info("After calling solver (solver results summary below)\n%s",
            str(results.solver))

print("\nReturned from solver; solver results summary printed below")
print(str(results.solver))
print("The solver returned the status: " + str(results.solver.status))

#if ((results.solver.status != SolverStatus.ok) or 
#    (results.solver.termination_condition != TerminationCondition.optimal)):
if (results.solver.status != SolverStatus.ok):
    logger.info("The solver returned a status of: " + str(results.solver.status))
    script_exit(1)


# https://github.com/tum-ens/notebooks/blob/master/urbs/pyomoio.py
# https://stackoverflow.com/questions/56904738/how-to-write-pyomo-solutions-including-only-parameters-objective-and-variables
# https://github.com/linmeishang/CDFarm-Decision-Model/blob/master/ConcreteCDFarm6_pandas.py
'''
def GetVarIndexLabels(VarIndex):
    labels = []

    if (VarIndex.dimen > 1):
        if (VarIndex.domain):
            SetTuple = VarIndex.domain.set_tuple
        else:
            SetTuple = VarIndex.set_tuple

        for IndexMember in SetTuple:
            labels.extend(GetVarIndexLabels(IndexMember))

    elif (VarIndex.dimen == 1):
        if (VarIndex.domain):
            labels.append(VarIndex.domain.name)
        else:
            labels.append(VarIndex.name)
    else:
        pass

    return labels
'''

index_labels_cache = {}

def get_index_labels(index):
    labels = index_labels_cache.get(index.name)
    if labels is not None:
        return labels

    labels = []

    if (index.dimen > 1):
        if (index.domain):
            subsets = index.domain.subsets(expand_all_set_operators = False)
        else:
            subsets = index.subsets(expand_all_set_operators = False)

        for index_member in subsets:
            labels.extend(get_index_labels(index_member))
    elif (index.dimen == 1):
        if (index.domain.name == "Any"):
            labels.append(index.name)
        elif (index.domain):
            labels.append(index.domain.name)
        else:
            labels.append(index.name)
    else:
        pass

    index_labels_cache[index.name] = labels
    return labels

def process_columns(columns_list):
    for pos, geog_column_name in enumerate(model.geog_columns_list):
        index_list = [index for index, column_name in enumerate(columns_list) 
                            if column_name == geog_column_name]
        if len(index_list) == 2:
            columns_list[index_list[0]] = model.src_geog_columns_list[pos]
            columns_list[index_list[1]] = model.dest_geog_columns_list[pos]

PARAM_INDEX_ATTRIBUTE_NAME_CHOICES = ["_index", "_index_set"]
VAR_INDEX_ATTRIBUTE_NAME_CHOICES = ["_index", "_index_set"]

def get_param_index_attibute_name(param_obj):
    for choice in PARAM_INDEX_ATTRIBUTE_NAME_CHOICES:
        if hasattr(param_obj, choice):
             return choice

    print("ERROR: Pyomo version does not support any of these attributes for Param:",
          PARAM_INDEX_ATTRIBUTE_NAME_CHOICES)
    logger.info("Exit")
    sys.exit(1)

def get_var_index_attibute_name(var_obj):
    for choice in VAR_INDEX_ATTRIBUTE_NAME_CHOICES:
        if hasattr(var_obj, choice):
             return choice

    print("ERROR: Pyomo version does not support any of these attributes for Var:",
          VAR_INDEX_ATTRIBUTE_NAME_CHOICES)
    logger.info("Exit")
    sys.exit(1)

def CreateVarDF(Var):
    if (Var.dim() > 0):
        columns_list = get_index_labels(getattr(Var, get_var_index_attibute_name(Var)))
        process_columns(columns_list)
        df = pd.DataFrame(list(Var._data.keys()), columns = columns_list)
    else:
        df = pd.DataFrame()

    df[Var.name] = [VarObj.value for VarObj in Var._data.values()]

    return df

def ProcessDF(VarName, df):
    NotWriteVarList = ["LegacyQtyGen"]
    DropZeroValuesVarList = ["QtyGen", "StorChargedByTypeVintage", "StorDischargedByTypeVintage", "StorEnergyCharged", "QtyFuelUsed", ]

    if (VarName in NotWriteVarList):
        WriteDF = False
    else:
        WriteDF = True

    if (VarName in DropZeroValuesVarList ):
        df.drop(df[df[VarName] <= 1e-7].index, inplace = True)

    return WriteDF

ExcelVarList = ["QtyGenFromTech", "ExistingCapacity", "CapacityInstalled", "QtyFuelUsed", "QtyFuelForElec", "QtyGenFromFuel"]

def CreateParamDF(Param):
    if (Param.dim() > 0):
        columns_list = get_index_labels(getattr(Param, get_param_index_attibute_name(Param)))
        process_columns(columns_list)
        df = pd.DataFrame(list(Param._data.keys()), columns = columns_list)
    else:
        df = pd.DataFrame()

    df[Param.name] = list(Param._data.values())

    return df

logger.info("Starting creation of dataframes per parameter")
print("Creating dataframes per parameter and writing to CSV files ...")

for param in instance.component_objects(Param, active = True):
    df = CreateParamDF(param)

    #logger.info("Columns for " + param.name + " : %s", df.columns.values.tolist())

    df.to_csv(output_path_param / (param.name + ".csv"), index = False)
    
    logger.info("DatatFrame written: " + param.name + "%s", df.columns.values.tolist())
 
    del df

logger.info("Starting creation of dataframes per variable")
print("Creating dataframes per variable and writing to CSV files ...")

for var in instance.component_objects(Var, active = True):
    df = CreateVarDF(var)

    WriteDF = ProcessDF(var.name, df)

    #logger.info("Columns for " + var.name + " : %s", df.columns.values.tolist())

    if (WriteDF):
        df.to_csv(output_path_var / (var.name + ".csv"), index = False)

        if (var.name in ExcelVarList):
            df.to_excel(output_path_var / (var.name + ".xlsx"), index = False)

        logger.info("DatatFrame written: " + var.name + "%s", df.columns.values.tolist())
    else:
        logger.info("DatatFrame *NOT* written: " + var.name + "%s", df.columns.values.tolist())

    del df

script_exit(1)

def main():
    return None
