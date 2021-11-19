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
"""Functions to be used by common. all functions from this module
are available in Common.yml validation code.
"""
from rumi.io import utilities
import datetime
import functools

from rumi.io import loaders
from rumi.io import constant
from rumi.io.utilities import seasons_size, compute_intervals
from rumi.io.utilities import get_geographic_columns
from rumi.io.utilities import get_time_columns
from rumi.io.utilities import valid_geography
from rumi.io.utilities import balancing_area
from rumi.io.utilities import balancing_time


class CommonValidationError(Exception):
    pass


def drop_columns(tabular, n):
    """drops first n items from each row and returns new tabular data

    >>> drop_columns([[1, 2, 3],
                      [21, 22, 23],
                      [31, 32, 33]],
                     1)
    [[2, 3], [22, 23], [32, 33]]
    """
    return [row[n:] for row in tabular]


def concat(*args):
    return sum([list(item) for item in args], [])


def find_hours(end, start):
    if end < start:
        return 24-start + end
    else:
        return end - start


def dayslice_size():
    DaySlices = loaders.get_parameter("DaySlices")
    dayslices = DaySlices.to_dict(orient='records')
    dayslices.append(dayslices[0])
    return {s['DaySlice']: find_hours(dayslices[i+1]['StartHour'], s['StartHour'])
            for i, s in enumerate(dayslices[:-1])}


def sum_of_durations(Seasons):
    intervals = compute_intervals(Seasons)
    return sum(intervals.values())
