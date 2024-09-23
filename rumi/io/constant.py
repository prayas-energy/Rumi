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
"""constants related to names of geographic, consumer and time columns
"""


def get_columns(X):
    return {item.upper(): list(X[:X.index(
        item)+1]) for item in X}


GEOGRAPHIES = ('ModelGeography', 'SubGeography1',
               'SubGeography2', 'SubGeography3')
GEO_COLUMNS = get_columns(GEOGRAPHIES)
TIME_SLICES = ("Year", "Season", "DayType", "DaySlice")
TIME_COLUMNS = get_columns(TIME_SLICES)

CONSUMER_TYPES = ('ConsumerType1', 'ConsumerType2')
CONSUMER_COLUMNS = get_columns(CONSUMER_TYPES)
CONSUMER_COLUMNS.setdefault("CONSUMERALL", [])

ST_SEPARATOR_CHAR = '+'
