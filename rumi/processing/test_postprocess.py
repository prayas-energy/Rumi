import shutil
import os
import pytest
import pandas as pd
import numpy as np
from rumi.processing import postprocess
from rumi.io import config
from rumi.io import utilities
from rumi.io import constant
from rumi.io import loaders


def get_parameter(name, **kwargs):
    if name == "ModelPeriod":
        return pd.DataFrame({"StartYear": [2021],
                             "EndYear": [2022]})
    elif name == "Seasons":
        return utilities.make_dataframe("""Season,StartMonth,StartDate
SUMMER,4,1
MONSOON,6,1
AUTUMN,9,1
WINTER,11,1
SPRING,2,1""")
    elif name == "DayTypes":
        return utilities.make_dataframe("""DayType,Weight
A,0.4
B,0.6""")
    elif name == "DaySlices":
        return utilities.make_dataframe("""DaySlice,StartHour
EARLY,6
MORN,9
MID,12
AFTERNOON,15
EVENING,18
NIGHT,22"""
                                        )
    elif name == "ModelGeography":
        return "INDIA"
    elif name == "SubGeography1":
        return 'ER,WR,NR,SR,NER'.split(",")
    elif name == "SubGeography2":
        return {"ER": 'BR,JH,OD,WB'.split(","),
                "WR": 'CG,GJ,MP,MH,GA,UT'.split(","),
                "NR": 'DL,HR,HP,JK,PB,RJ,UP,UK'.split(","),
                "SR": 'AP,KA,KL,TN,TS'.split(","),
                "NER": 'AS,NE'.split(",")}


def get_config_value(configname):
    if configname == "scenario":
        return "S1_Ref"
    else:
        return config.get_config_value(configname)


@pytest.fixture()
def dummy_output():
    path = "test_output_supply"
    scenario = "S1_Ref"
    os.mkdir(path)
    os.mkdir(os.path.join(path, scenario))
    os.mkdir(os.path.join(path, scenario, "Supply"))
    os.mkdir(os.path.join(path, scenario, "Supply", "Output"))
    os.mkdir(os.path.join(path, scenario, "Demand"))
    os.mkdir(os.path.join(path, scenario, "Demand", "Output"))
    os.mkdir(os.path.join(path, scenario, "Supply", "Output", "Run-Outputs"))
    files = ["EndUseDemandMetByDom",
             "EndUseDemandMetByImp",
             "ECTInputDom",
             "ECTInputImp"]
    files = [".".join([f, "csv"]) for f in files]
    for f in files:
        with open(os.path.join(path, scenario, "Supply", "Output", "Run-Outputs", f), "w") as fd:
            fd.write("\n")

    demandpath = os.path.join(path, scenario, "Demand", "Output")
    with open(os.path.join(demandpath, "EndUseDemandEnergy.csv"), "w") as f:
        f.write("\n")

    yield path
    shutil.rmtree(path)


def test_demand_filepath(monkeypatch):
    tmp_path = "output"
    monkeypatch.setattr(postprocess,
                        "get_demand_output_path",
                        lambda: tmp_path)

    ds, es, ec, st = "ds", "es", "ec", "st"
    folderpath = os.path.join(tmp_path, "DemandSector", ds, es)
    filename = f"{ds}_{es}_{ec}_Demand.csv"
    path = os.path.join(folderpath, filename)
    assert postprocess.demand_filepath(ds, es, ec) == path
    filename = f"{ds}_{es}_{st}_{ec}_Demand.csv"
    path = os.path.join(folderpath, filename)
    assert postprocess.demand_filepath(ds, es, ec, st) == path

    
def test_season_wise(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)

    tcols = list(constant.TIME_SLICES)
    gcols = list(constant.GEOGRAPHIES)
    entity = "EnergyConvTech"
    df1 = utilities.base_dataframe_all(geocols=gcols[:2],
                                       timecols=tcols,
                                       colname="value",
                                       val=1.0).reset_index()
    df1[entity] = "DF1"

    df2 = utilities.base_dataframe_all(geocols=gcols[:2],
                                       timecols=tcols[:2],
                                       colname="value",
                                       val=1.0).reset_index()
    df2[entity] = "DF2"

    df3 = utilities.base_dataframe_all(geocols=gcols[:2],
                                       timecols=tcols,
                                       colname="value",
                                       val=1.0).reset_index()
    df3[entity] = "DF3"
    df3['DayNo'] = 1

    df4 = utilities.base_dataframe_all(geocols=gcols[:2],
                                       timecols=tcols,
                                       colname="value",
                                       val=1.0).reset_index()
    df4[entity] = "DF4"
    df4['DayNo'] = np.nan

    df5 = utilities.base_dataframe_all(geocols=gcols[:2],
                                       timecols=tcols,
                                       colname="value",
                                       val=1.0).reset_index()
    df5[entity] = "DF5"
    df5['DayNo'] = 1

    df = pd.concat([df1, df2, df3, df4, df5])
    dfs = postprocess.season_wise(df, colname="value")

    for item in ['DF2', 'DF3', 'DF5']:
        d = dfs['value'][dfs[entity] == item]
        assert d.sum() == pytest.approx(len(d))

    for item in ['DF1', 'DF4']:
        d = dfs['value'][(dfs[entity] == item) & (
            dfs.Season == 'SUMMER') & (dfs.DayType == 'A')]
        print(dfs[dfs[entity] == item])

        assert d.sum() == pytest.approx(len(d)*24.4)

        d1 = dfs['value'][(dfs[entity] == item) & (
            dfs.Season == 'SPRING') & (dfs.DayType == 'B')]

        assert d1.sum() == pytest.approx(len(d1)*35.4)


def test_find_order():
    assert postprocess.find_order(list('abcd'), list("dcbe")) == list("abcde")
    assert postprocess.find_order(list('abcd'), list("abcd")) == list("abcd")
    assert postprocess.find_order(list('abcd'), list("abc")) == list("abcd")
    assert postprocess.find_order(list('abcd'), list("abcde")) == list("abcde")
