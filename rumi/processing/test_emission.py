from rumi.processing import emission
from rumi.io import loaders, utilities
import pandas as pd


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


def test_get_EC_ST_ES_data(monkeypatch):
    monkeypatch.setattr(loaders, 'get_parameter', get_parameter)
    df = utilities.make_dataframe("""EnergyCarrier,ServiceTech,EnergyService
EC,ST1,ES1
EC2,ST1,ES1
EC,ST2,ES1
EC,ST3,ES1
EC1,ST4,ES1
EC,ST5,ES1""")
    print(df)
    print(emission.get_EC_ST_ES_data())
    assert df.compare(emission.get_EC_ST_ES_data()).empty
