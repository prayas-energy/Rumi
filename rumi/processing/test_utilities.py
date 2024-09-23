from rumi.processing import utilities
from rumi.io import loaders
from rumi.io import functionstore as fs
from rumi.io import utilities as ioutils
import pandas as pd
import pytest


def get_parameter(param_name):
    if param_name == "ModelPeriod":
        return pd.DataFrame([{"StartYear": 2021, "EndYear": 2022}])
    elif param_name == "DayTypes":
        return pd.DataFrame({"DayType": ['A', 'B'],
                             "Weight": [0.4, 0.6]})
    elif param_name == "Seasons":
        return pd.DataFrame({"Season": ["SUMMER", "MONSOON"],
                             "StartMonth": [4, 5],
                             "StartDate": [1, 31]})
    elif param_name == "ModelGeography":
        return "INDIA"
    elif param_name == "SubGeography1":
        return ["NR", "SR", "NER", "WR", "ER"]
    elif param_name == "DaySlices":
        return ioutils.make_dataframe("""DaySlice,StartHour
H0,0
H1,13
""")


def test_seasonwise_timeslices(monkeypatch):
    monkeypatch.setattr(loaders, "get_parameter", get_parameter)
    data = ioutils.base_dataframe(['Year', 'Season', 'DayType', 'DaySlice', 'ModelGeography'],
                                  colname='Value',
                                  val=1).reset_index()
    data = utilities.seasonwise_timeslices(data, "Value")
    assert data['SeasonValue'].values == pytest.approx(
        fs.flatten([[x, x] for x in [24, 36, 122, 183]])*2)

    data = ioutils.base_dataframe(['Year', 'Season', 'DayType', 'ModelGeography', 'SubGeography1'],
                                  colname='Value',
                                  val=1).reset_index()
    data = utilities.seasonwise_timeslices(data, "Value")
    assert data['SeasonValue'].values == pytest.approx([24, 36, 122, 183]*10)

    data = ioutils.base_dataframe(['Year', 'Season', 'DayType', 'ModelGeography', 'SubGeography1', 'DaySlice'],
                                  colname='Value',
                                  val=1).reset_index()
    data = utilities.seasonwise_timeslices(data, "Value")
    assert data['SeasonValue'].values == pytest.approx(
        fs.flatten([[x, x] for x in [24, 36, 122, 183]]*10))


def test_get_coarsest(monkeypatch):
    monkeypatch.setattr(loaders, "get_parameter", get_parameter)
    d1 = ioutils.base_dataframe(['Year', 'Season', 'DayType', 'DaySlice', 'ModelGeography'],
                                colname='Value',
                                val=1).reset_index()
    
    d2 = ioutils.base_dataframe(['Year', 'ModelGeography'],
                                colname='Value',
                                val=1).reset_index()
    d3 = ioutils.base_dataframe(['Year', 'Season', 'DayType', 'DaySlice', 'ModelGeography', 'SubGeography1'],
                                colname='Value',
                                val=1).reset_index()
    d3['ConsumerType1'] = "C1"
    cols = utilities.get_coarsest([d1, d2, d3], take_cons_cols=True)
    assert cols == ['Year', 'ModelGeography']
    
