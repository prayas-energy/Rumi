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
import pkg_resources
import pandas as pd
import csv
import os
import pytest
from rumi.io import demand
from rumi.io import loaders
from rumi.io import config
from rumi.io import functionstore as fs
from rumi.io import utilities
from rumi.io import filemanager
from rumi.io import constant
from rumi.io.test_filemanager import clear_filemanager_cache
import yaml


def get_parameter(param_name, **kwargs):
    if 'demand_sector' in kwargs:
        ds = kwargs['demand_sector']
        q = f"DemandSector == '{ds}'"
    if param_name == "ST_EC_Map":
        return [["ST1", "EC", 'EC2'],
                ["ST2", "EC"],
                ["ST3", "EC"],
                ["ST4", 'EC1'],
                ['ST5', 'EC']]
    elif param_name == "DS_List":
        return ['DS1', 'DS2']
    elif param_name == "DS_ES_STC_DemandGranularityMap":
        return pd.DataFrame({'DemandSector': ['DS1']*5,
                             'EnergyService': ['ES1']*5,
                             'ServiceTechCategory': ['ST1', 'ST2', 'ST3', 'ST4', 'ST5'],
                             'ConsumerGranularity': ['CONSUMERALL']*5,
                             'GeographicGranularity': ['MODELGEOGRAPHY']*4 + ['SUBGEOGRAPHY1'],
                             'TimeGranularity': ['YEAR']*4+['SEASON']}).query(q)
    elif param_name == 'DS_ES_EC_Map':
        return pd.DataFrame({'DemandSector': ["DS1", "DS2"]*2,
                             'EnergyService': ["ES2", "ES3", 'ES4', "ES5"],
                             'EnergyCarrier': ["EC", "EC2"]*2,
                             'ConsumerGranularity': ['CONSUMERALL']*4,
                             'GeographicGranularity': ['MODELGEOGRAPHY']*4,
                             'TimeGranularity': ['YEAR']*4}).query(q)
    elif param_name == "DS_ES_Map":
        return pd.DataFrame({"DemandSector": ["DS1", "DS1", "DS2", "DS1", "DS2"],
                             "EnergyService": ["ES1", "ES2", "ES3", "ES4", "ES5"],
                             "InputType": ["BOTTOMUP"]+["EXOGENOUS"]*4}).query(q)
    elif param_name == "DS_Cons1_Map":
        return {'DS1': ['SUBGEOGRAPHY2', 'YEAR', 'URBAN', 'RURAL'],
                'DS2': ['SUBGEOGRAPHY2', 'YEAR', 'URBAN', 'RURAL']}
    elif param_name == "ModelGeography":
        return "INDIA"
    elif param_name == "SubGeography1":
        return ["NR", "ER", "WR", "SR", "NER"]
    elif param_name == "SubGeography2":
        return {"ER": 'BR,JH,OD,WB'.split(","),
                "WR": 'CG,GJ,MP,MH,GA,UT'.split(","),
                "NR": 'DL,HR,HP,JK,PB,RJ,UP,UK'.split(","),
                "SR": 'AP,KA,KL,TN,TS'.split(","),
                "NER": 'AS,NE'.split(",")}
    elif param_name == "ModelPeriod":
        return pd.DataFrame({"StartYear": [2021], "EndYear": [2025]})
    elif param_name == "STC_ST_Map":
        return [['ST1', 'ST1']]
    elif param_name == "STC_ES_Map":
        return [["ST1", "ES1"],
                ["ST2", "ES1"],
                ["ST3", "ES1"],
                ["ST4", "ES1"],
                ["ST5", "ES1"]]
    elif param_name == "Seasons":
        return utilities.make_dataframe("""Season,StartMonth,StartDate
SUMMER,4,1
MONSOON,6,1
AUTUMN,9,1
WINTER,11,1
SPRING,2,1
""")


def test_get_service_techs(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    assert demand.get_service_techs("DS1", "ES1", "EC") == (
        'ST1', 'ST2', 'ST3', 'ST5')


def test_derive_ECs(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    print(demand.get_combined("DS_ES_EC_DemandGranularityMap"))
    assert set(demand.derive_ECs("DS1")) == {'EC', 'EC1', 'EC2'}


def test_ST_to_EC(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    assert demand.ST_to_EC("ST4") == "EC1"


def test_ST_to_ECs(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    assert set(demand.ST_to_ECs("ST1")) == {"EC", 'EC2'}


def test_EC_to_STs(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    assert set(demand.EC_to_STs("EC")) == {"ST1", "ST2", "ST3", 'ST5'}


def test_get_corresponding_sts(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)

    assert set(demand.get_corresponding_sts(
        "DS1", "ES1", "ST2")) == {'ST1', 'ST2', 'ST3', 'ST5'}


def test_is_bottomup(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    assert demand.is_bottomup("DS1", "ES1")
    assert not demand.is_bottomup("RANDOM", "XYZ")


def test_get_all_ds_es_ec(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    assert set(demand.get_all_ds_es_ec()) == set([
        ('DS1', 'ES1', 'EC'),
        ('DS1', 'ES1', 'EC1'),
        ('DS1', 'ES1', 'EC2'),
        ('DS1', 'ES4', 'EC'),
        ('DS2', 'ES3', 'EC2'),
        ('DS2', 'ES5', 'EC2'),
        ('DS1', 'ES2', 'EC')])
    assert fs.unique(demand.get_all_ds_es_ec())


def test_get_combined_granularity_map(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    names = ['DemandSector', 'EnergyService', 'EnergyCarrier']
    assert fs.unique(
        list(zip(*demand.listcols(demand.get_combined_granularity_map()[names]))))


def create_dir_structure(instance_folder):
    def create_substructure(folder):
        common = folder / "Common"
        common.mkdir()
        cparams = common / "Parameters"
        cparams.mkdir()
        demand = folder / "Demand"
        demand.mkdir()
        parameters = demand / "Parameters"
        parameters.mkdir()
        DS1 = parameters / "DS1"
        DS1.mkdir()
        ES1 = DS1 / "ES1"
        ES1.mkdir()
        ES2 = DS1 / "ES2"
        ES2.mkdir()

    global_data = instance_folder / 'Default Data'
    global_data.mkdir()
    create_substructure(global_data)

    scenarios = instance_folder / "Scenarios"
    scenarios.mkdir()
    test_scenario = scenarios / "test_scenario"
    test_scenario.mkdir()
    create_substructure(test_scenario)


def print_dir(path, depth=0):
    print(" "*depth+"+", os.path.basename(path))
    for file_ in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_)):
            print("| "*depth+"-", file_)
        else:
            print_dir(os.path.join(path, file_), depth+1)


def create_common(instance_folder):
    MG = instance_folder/"Scenarios"/"test_scenario" /\
        "Common"/"Parameters"/"ModelGeography.csv"
    MG.write_text("INDIA")
    SG1 = instance_folder/"Scenarios"/"test_scenario" /\
        "Common"/"Parameters"/"SubGeography1.csv"
    SG1.write_text("NR,SR,ER,WR,NER")
    MP = instance_folder/"Scenarios"/"test_scenario" /\
        "Common"/"Parameters"/"ModelPeriod.csv"
    MP.write_text(",".join(["StartYear", "EndYear"]) +
                  "\n"+",".join(["2021", "2031"]))


def create_numinstances(location, ST, val=1.0, finer=False):
    numinstances = location / 'NumInstances.csv'

    if finer:
        geocols = ['ModelGeography', 'SubGeography1']
    else:
        geocols = ['ModelGeography']
    data = utilities.base_dataframe_all(geocols=geocols,
                                        timecols=['Year'],
                                        colname='NumInstances',
                                        val=val).reset_index()
    header = ",".join(['ServiceTech']+list(data.columns))
    text = header + "\n"
    for row in data.values:
        text += ",".join([ST] + [str(r) for r in row]) + "\n"

    numinstances.write_text(text)


def test_get_DS_ES_parameter(clear_filemanager_cache, tmp_path, monkeypatch):
    """clear_filemanager_cache is required because functions in filemanager
    are cached and we are using it here
    """
    instance_folder = tmp_path / "global_test"
    instance_folder.mkdir()

    def get_config_value(name):
        if name == "scenario":
            return "test_scenario"
        elif name == "model_instance_path":
            return instance_folder.absolute()
        elif name == "yaml_location":
            yaml_location = pkg_resources.resource_filename("rumi",
                                                            "Config")
            return yaml_location
        elif name == "config_location":
            yaml_location = pkg_resources.resource_filename("rumi",
                                                            "Config")
            # for this test conf will be taken from package not from model instance
            os.path.join(yaml_location, "Config.yml")

    loaders.get_parameter.cache_clear()
    monkeypatch.setattr(config, "get_config_value", get_config_value)
    monkeypatch.setattr(demand, "ST_to_STC", lambda x: x)
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)

    create_dir_structure(instance_folder)
    create_common(instance_folder)
    # ========================================================
    # Default data from DS folder
    ds = instance_folder/"Scenarios"/"test_scenario"/"Demand"/"Parameters"/"DS1"
    create_numinstances(ds, ST="ST1")
    data = utilities.base_dataframe_all(geocols=['ModelGeography'],
                                        timecols=['Year'],
                                        colname='ServiceTech',
                                        val='ST1').reset_index()
    data['NumInstances'] = 1.0
    df1 = demand.get_DS_ES_parameter("NumInstances", "DS1", "ES1")
    assert set(data.columns) == set(df1.columns)

    data = data[[c for c in df1.columns]]
    assert fs.get_set(data) == fs.get_set(df1)

    # =========================================================
    # Data overridden with data from ES folder
    es = instance_folder/"Scenarios"/"test_scenario"/"Demand"/"Parameters"/"DS1"/"ES1"

    create_numinstances(es, ST="ST1", val=2.0)
    df2 = demand.get_DS_ES_parameter(
        "NumInstances", "DS1", "ES1")
    data['NumInstances'] = 2.0
    assert data.equals(df2)

    # =========================================================
    # Data overridden with data from ES folder, remaining data
    # taken from DS folder
    create_numinstances(es, ST="ST2", val=2.0)
    df3 = demand.get_DS_ES_parameter(
        "NumInstances", "DS1", "ES1")
    data2 = utilities.base_dataframe_all(geocols=['ModelGeography'],
                                         timecols=['Year'],
                                         colname='ServiceTech',
                                         val='ST2').reset_index()
    data2['NumInstances'] = 2.0
    data2 = data2.reindex(
        columns=['ServiceTech', 'ModelGeography', 'Year', 'NumInstances'])
    final_data = pd.concat([df1, data2]).reset_index(drop=True)
    assert final_data.equals(df3)

    # =========================================================
    # Data overridden from ES folder , but default data has coarser granuarity
    # than overridden data. result should be of granuarity same as of data
    # given in ES folder
    es2 = instance_folder/"Scenarios"/"test_scenario"/"Demand"/"Parameters"/"DS1"/"ES1"
    create_numinstances(es2, ST="ST1", val=3.0, finer=True)
    df4 = demand.get_DS_ES_parameter("NumInstances",
                                     "DS1",
                                     "ES1")

    data4 = utilities.base_dataframe_all(geocols=['ModelGeography', 'SubGeography1'],
                                         timecols=['Year'],
                                         demand_sector='DS1',
                                         colname='ServiceTech',
                                         val='ST1').reset_index()
    data4['NumInstances'] = 3.0
    cols = [c for c in df4.columns]
    assert set(cols) == set(data4.columns)
    data4 = data4[cols]
    assert fs.get_set(data4) == fs.get_set(df4)


def test_add_demand_sector_filters(monkeypatch, clear_filemanager_cache):
    def yaml_location(*args):
        return pkg_resources.resource_filename("rumi",
                                               "Config")

    monkeypatch.setattr(config, "get_config_value", yaml_location)
    demand_sector = "D_RES"
    demand.add_demand_sector_filters(demand_sector)
    demand_yaml = filemanager.demand_specs()
    params = [
        k for k in demand_yaml if 'DemandSector' in demand_yaml[k].get('columns', [])]
    dataframe_filter = f"DemandSector == '{demand_sector}'"
    for param in params:
        assert dataframe_filter in demand_yaml[param]['filterqueries']

    assert f" item == '{demand_sector}'" in demand_yaml['DS_List']['filterqueries']


def test_extract_STCs():
    f = "FAN+UsagePenetration.csv"
    assert demand.extract_STCs(f) == ['FAN']
    assert demand.extract_STCs("FAN+AC+UsagePenetration.csv") == ['FAN', 'AC']


def create_penetrations(comb, folder):
    p1 = folder / \
        constant.ST_SEPARATOR_CHAR.join(comb+["UsagePenetration.csv"])
    p1.write_text("\n")


def test_list_dir_usagepenetration(clear_filemanager_cache, tmp_path, monkeypatch):
    instance_folder = tmp_path / "global_test"
    instance_folder.mkdir()

    def get_config_value(name):
        if name == "scenario":
            return "test_scenario"
        elif name == "model_instance_path":
            return instance_folder.absolute()
        elif name == "yaml_location":
            yaml_location = pkg_resources.resource_filename("rumi",
                                                            "Config")
            return yaml_location
        elif name == "config_location":
            yaml_location = pkg_resources.resource_filename("rumi",
                                                            "Config")
            # for this test conf will be taken from package not from model instance
            os.path.join(yaml_location, "Config.yml")

    loaders.get_parameter.cache_clear()
    monkeypatch.setattr(config, "get_config_value", get_config_value)
    create_dir_structure(instance_folder)
    gfolder = instance_folder / "Default Data" / \
        "Demand" / "Parameters" / "DS1" / "ES1"
    sfolder = instance_folder / "Scenarios" / \
        "test_scenario" / "Demand" / "Parameters" / "DS1" / "ES1"

    create_penetrations(['A', 'B'], gfolder)

    l = [str(
        gfolder / constant.ST_SEPARATOR_CHAR.join(["A", "B", "UsagePenetration.csv"]))]
    r = demand.list_dir_usagepenetration(gfolder)
    print(r)
    print(l)
    assert (r == {('A', 'B'): l}) or (r == {('B', 'A'): l})
    create_penetrations(['B', 'A'], gfolder)
    l.insert(0, str(
        gfolder / constant.ST_SEPARATOR_CHAR.join(["B", "A", "UsagePenetration.csv"])))

    r = demand.list_dir_usagepenetration(gfolder)
    print(r)
    print(l)
    assert (r == {('A', 'B'): l}) or (r == {('B', 'A'): l}) or (
        r == {('A', 'B'): l[::-1]}) or (r == {('B', 'A'): l[::-1]})


def test_find_usage_penetrations(clear_filemanager_cache, tmp_path):
    create_penetrations(['A', 'B'], tmp_path)
    create_penetrations(['A', 'B', 'C'], tmp_path)

    assert demand.find_usage_penetrations(tmp_path, ['A', 'B']) == [
        'A+B+UsagePenetration.csv']
    assert demand.find_usage_penetrations(tmp_path, ['B', 'A']) == [
        'A+B+UsagePenetration.csv']
    assert demand.find_usage_penetrations(tmp_path/"test", ["A", "B"]) == []


def test_fill_missing_TechSplitRatio():
    pass


def test_fill_missing_rows_with_zero(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    monkeypatch.setattr(filemanager, "get_specs", lambda x: dict())
    data = utilities.base_dataframe_all(conscols=['ConsumerType1'],
                                        geocols=['ModelGeography',
                                                 'SubGeography1'],
                                        timecols=['Year'],
                                        demand_sector="DS1",
                                        val=1.0).reset_index().set_index('SubGeography1')
    data = data.drop(["NR"]).reset_index()
    assert len(data.query('SubGeography1 == "NR"')) == 0
    d = demand.fill_missing_rows_with_zero(
        "Test_Param", data, demand_sector="DS1")
    assert len(d.query('SubGeography1 == "NR"')) > 0
    assert d['dummy'].sum() == len(data)

    monkeypatch.setattr(filemanager, "get_specs", lambda x: {
                        "entities": ["EnergyCarrier"]})
    data1 = utilities.base_dataframe_all(conscols=['ConsumerType1'],
                                         geocols=['ModelGeography',
                                                  'SubGeography1'],
                                         timecols=['Year'],
                                         demand_sector="DS1",
                                         val=1.0).reset_index().set_index('SubGeography1')
    data1['EnergyCarrier'] = "LPG"
    data2 = utilities.base_dataframe_all(conscols=['ConsumerType1'],
                                         geocols=['ModelGeography',
                                                  'SubGeography1'],
                                         timecols=['Year'],
                                         demand_sector="DS1",
                                         val=1.0).reset_index().set_index('SubGeography1')
    data2['EnergyCarrier'] = "CNG"

    data = pd.concat([data1, data2])
    data = data.drop(["NR"]).reset_index()
    assert len(data.query('SubGeography1 == "NR"')) == 0
    d = demand.fill_missing_rows_with_zero(
        "Test_Param", data, demand_sector="DS1")
    assert len(d.query('SubGeography1 == "NR"')) > 0
    assert d['dummy'].sum() == len(data)


def test_check_coarser_sum(monkeypatch):
    monkeypatch.setattr(loaders, "get_parameter", get_parameter)
    monkeypatch.setattr(demand, "get_demand_granularity",
                        lambda demand_sector, energy_service, energy_carrier, service_tech_category: ("CONSUMERALL", "MODELGEOGRAPHY", "YEAR",))
    GTProfile = utilities.base_dataframe_all(geocols=['ModelGeography', 'SubGeography1'], timecols=[
        'Year'], colname='GTProfile', val=1/55).reset_index()
    assert demand.check_coarser_sum(GTProfile)
    GTProfile = utilities.base_dataframe_all(geocols=['ModelGeography', 'SubGeography1'], timecols=[
        'Year', "Season"], colname='GTProfile', val=1/10).reset_index()
    assert demand.check_coarser_sum(GTProfile)
    GTProfile = utilities.base_dataframe_all(geocols=['ModelGeography', 'SubGeography1'], timecols=[
        'Year', "Season"], colname='GTProfile', val=1/10).reset_index()
    GTProfile = GTProfile.set_index('Year')
    GTProfile.loc[2021, 'GTProfile'] = 0
    assert not demand.check_coarser_sum(GTProfile.reset_index())
    GTProfile = utilities.base_dataframe_all(geocols=['ModelGeography', 'SubGeography1', 'SubGeography2'], timecols=[
        'Year', "Season"], colname='GTProfile', val=1/10).reset_index()
    gy = GTProfile.set_index('Year')
    gy.loc[2021, 'GTProfile'] = 0
    assert not demand.check_coarser_sum(gy.reset_index())
    gs = GTProfile.set_index('Season')
    gs.loc['SUMMER', 'GTProfile'] = 0
    assert demand.check_coarser_sum(gs.reset_index())


def test_get_granularity(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    C, G, T = demand.get_granularity("ExogenousDemand",
                                     demand_sector="DS2",
                                     energy_service="ES5",
                                     energy_carrier="EC2")
    assert C == "CONSUMERALL"
    assert G == "MODELGEOGRAPHY"
    assert T == "YEAR"

    C, G, T = demand.get_granularity("ExogenousDemand",
                                     demand_sector="DS2",
                                     energy_service="ES5",
                                     service_tech_category=None,
                                     energy_carrier="EC2")

    assert C == "CONSUMERALL"
    assert G == "MODELGEOGRAPHY"
    assert T == "YEAR"

    C, G, T = demand.get_granularity("EfficiencyLevelSplit",
                                     demand_sector="DS1",
                                     energy_service="ES1",
                                     service_tech_category='ST2',
                                     energy_carrier=None)
    assert C == 'CONSUMERALL'
    assert G == 'MODELGEOGRAPHY'
    assert T == 'YEAR'

    C, G, T = demand.get_granularity("NumInstances",
                                     demand_sector="DS1",
                                     energy_service="ES1",
                                     service_tech_category='ST1',
                                     energy_carrier=None)
    assert C == 'CONSUMERALL'
    assert G == 'MODELGEOGRAPHY'
    assert T == 'YEAR'

    with pytest.raises(KeyError):
        demand.get_granularity("NumInstances",
                               demand_sector="DS3",
                               energy_service="ES3",
                               service_tech_category='ST3',
                               energy_carrier=None)


def test_expand_to_approp_gran(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    edf = pd.DataFrame({'ServiceTech': ['ST5']})
    data = utilities.base_dataframe_all(conscols=[],
                                        geocols=['ModelGeography'],
                                        timecols=['Year'],
                                        demand_sector='DS1',
                                        colname="EfficiencyLevelSplit",
                                        val=1.0,
                                        extracols_df=edf).reset_index()

    df = demand.expand_to_approp_gran("EfficiencyLevelSplit",
                                      ['ServiceTech'],
                                      demand_sector="DS1",
                                      energy_service="ES1",
                                      data=data)

    df1 = utilities.base_dataframe_all(conscols=[],
                                       geocols=['ModelGeography',
                                                'SubGeography1'],
                                       timecols=['Year', 'Season'],
                                       demand_sector='DS1',
                                       colname="EfficiencyLevelSplit",
                                       val=1.0,
                                       extracols_df=edf).reset_index()

    assert fs.get_set(df) == fs.get_set(df1)

    df = demand.expand_to_approp_gran("EfficiencyLevelSplit",
                                      ['ServiceTech'],
                                      demand_sector="DS1",
                                      energy_service="ES1",
                                      data=pd.DataFrame())
    assert len(df) == 0

    df = demand.expand_to_approp_gran("EfficiencyLevelSplit",
                                      ['ServiceTech'],
                                      demand_sector="DS1",
                                      energy_service="ES1",
                                      data=None)
    assert fs.isnone(df)


def test_override(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    param_name = "PARAM"
    default_df = utilities.base_dataframe_all(conscols=['ConsumerType1'],
                                              geocols=['ModelGeography'],
                                              timecols=['Year'],
                                              demand_sector='DS1',
                                              colname="value",
                                              val=1.0).reset_index()

    over_df = utilities.base_dataframe_all(conscols=['ConsumerType1'],
                                           geocols=['ModelGeography',
                                                    'SubGeography1'],
                                           timecols=['Year', 'Season'],
                                           demand_sector='DS1',
                                           colname="value",
                                           val=2.0).reset_index()
    df = demand.override(param_name, "DS1", "ES1", default_df, over_df)

    assert set(df.itertuples(index=False, name=None)) == set(
        over_df.itertuples(index=False, name=None))

    default_df = utilities.base_dataframe_all(conscols=['ConsumerType1'],
                                              geocols=['ModelGeography',
                                                       'SubGeography1'],
                                              timecols=['Year',
                                                        'Season'],
                                              demand_sector='DS1',
                                              colname="value",
                                              val=1.0).reset_index()

    over_df = utilities.base_dataframe_all(conscols=['ConsumerType1'],
                                           geocols=['ModelGeography',
                                                    'SubGeography1'],
                                           timecols=['Year', 'Season'],
                                           demand_sector='DS1',
                                           colname="value",
                                           val=2.0).reset_index()
    df = demand.override(param_name, "DS1", "ES1", default_df, over_df)

    assert set(df.itertuples(index=False, name=None)) == set(
        over_df.itertuples(index=False, name=None))


def test_baseyear_demand(monkeypatch, clear_filemanager_cache):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    monkeypatch.setattr(filemanager, 'get_specs', lambda x: {
                        'entities': ['EnergyService', 'EnergyCarrier']})

    data = demand.BaseYearDemandBaseData(conscols=['ConsumerType1'],
                                         geocols=['ModelGeography',
                                                  'SubGeography1'],
                                         timecols=['Year',
                                                   'Season'],
                                         demand_sector='DS1',
                                         colname="BaseYearDemand",
                                         val=0.0,
                                         extracols_df=pd.DataFrame({"EnergyService": ['ES2'],
                                                                   'EnergyCarrier': ['EC']})).get_dataframe().reset_index()

    d = demand.fill_missing_rows_with_zero_baseyeardemand(
        "BaseYearDemand", data.query("Season !='WINTER'"), demand_sector="DS1")
    d = d[data.columns]

    assert set(data.itertuples(index=False, name=None)) == set(
        d.itertuples(index=False, name=None))


def compare_dataframes(df1, df2):
    return set(df1.itertuples(index=False, name=None)) == set(df2.itertuples(index=False, name=None))


def test_get_combined(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    combined = pd.DataFrame({'DemandSector': ["DS1", "DS1", "DS2", "DS2"],
                            'EnergyService': ["ES2", 'ES4', "ES3", "ES5"],
                             'EnergyCarrier': ["EC", "EC", "EC2", "EC2"],
                             'ConsumerGranularity': ['CONSUMERALL']*4,
                             'GeographicGranularity': ['MODELGEOGRAPHY']*4,
                             'TimeGranularity': ['YEAR']*4})

    assert compare_dataframes(demand.get_combined('DS_ES_EC_Map'), combined)
    assert compare_dataframes(get_parameter("DS_ES_EC_Map", demand_sector="DS1"),
                              combined.query("DemandSector == 'DS1'"))
