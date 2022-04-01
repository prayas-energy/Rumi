from rumi.processing import utilities
from rumi.io import loaders
from rumi.io import common
import pandas as pd
import pytest


def get_parameter(param_name):
    if param_name == "DayTypes":
        return pd.DataFrame({"DayType": ['A', 'B'],
                             "Weight": [0.4, 0.6]})
    elif param_name == "Seasons":
        return pd.DataFrame({"Season": ["SUMMER", "MONSOON"],
                             "StartMonth": [4, 5],
                             "StartDate": [1, 31]})


def test_seasonwise_timeslices(monkeypatch):
    monkeypatch.setattr(loaders, "get_parameter", get_parameter)
    data = pd.DataFrame({"Year": [2021, 2021],
                         "Season": ["SUMMER", "SUMMER"],
                         "DayType": ["A", "B"],
                         "Value": [1, 1]})
    data = utilities.seasonwise_timeslices(data, "Value")
    assert data['SeasonValue'].values == pytest.approx([24.0, 36.0])
