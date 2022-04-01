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
import pandas as pd
from rumi.io import loaders
from rumi.io import constant
from rumi.io.logger import logging
from rumi.io import common
from rumi.io import utilities as commonutils

logger = logging.getLogger(__name__)


def get_geographic_columns(geographic_granularity):
    return constant.GEO_COLUMNS[geographic_granularity]


def get_time_columns(time_granularity):
    return constant.TIME_COLUMNS[time_granularity]


def get_geographic_columns_from_dataframe(df):
    return [c for c in constant.GEOGRAPHIES if c in df.columns]


def group_by_season(data, target):
    """groups target column by season and appends new column to dataset for
    season wise computation of target.
    """
    if "Season" in data.columns:
        colname = "Season" + target
        if colname in data:
            del data[colname]
        rest_cols = [
            c for c in data.columns if c not in constant.TIME_SLICES + (target,)]
        by_day = commonutils.group_daytype(data, rest_cols, target)
        by_season = commonutils.group_season(by_day, rest_cols, target)
        df = by_season.rename(columns={target: colname})
        indexcols = rest_cols + ['Year', 'Season']
        data = data.set_index(indexcols)
        return data.join(df).reset_index()
    else:
        return data


def seasonwise_timeslices(data, target):
    """if data has granularity finer than season, 
    computes new target column  target * DayTypeWeight * NumDaysInSeason
    else same data is returned.

    assumes that dataframe has not empty fields.
    """
    if "DayType" in data.columns:
        colname = "Season" + target
        if colname in data:
            del data[colname]
        indexcols = [
            c for c in data.columns if c != target]
        if 'DayType' in data.columns:
            DayTypes = loaders.get_parameter('DayTypes')
            weights = DayTypes.set_index("DayType")['Weight']
        else:
            weights = 1
        data = data.set_index(indexcols)
        seasons_size = pd.Series(common.seasons_size())
        seasons_size.index.rename('Season', inplace=True)
        starget = data[target]*weights*seasons_size
        return data.join(starget.rename(colname)).reset_index()
    else:
        return data
