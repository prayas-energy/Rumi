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
"""function repository for loaders and validations. 
this module can depend only on python modules
"""
import pandas as pd
import datetime


def concat(*vectors):
    return sum([list(v) for v in vectors], [])


def x_in_y(x, y):
    if set(x) - set(y):
        print(set(x) - set(y), "Not found!")
    return not set(x) - set(y)


def one_to_one(x, y):
    return x_in_y(x, y) and x_in_y(y, x)


def circular(values, circle):
    values = list(values)
    m = values.index(min(values))
    values = values[m:] + values[:m]
    diff = [v2-v1 for v1, v2 in zip(values, values[1:])]
    return all([d > 0 for d in diff]) and all([v in circle for v in values])


def valid_date(m, d):
    return all(valid_date_(a, b) for a, b in zip(m, d))


def valid_date_(m, d):
    """
    checks validity of date for a non leap year.
    """
    try:
        datetime.datetime(2019, m, d)
    except ValueError as e:
        print("Invalid date:", e)
        return False
    except TypeError as e:
        print("Invalid date:", e)
        return False

    return True


def create_query(names, values):
    return " and ".join([f'({name} == "{value}")' for name, value in zip(names, values)])


def column(tabular, n):
    """get nth column from tabular data
    """
    col = []
    for row in tabular:
        if len(row) < n+1:
            col.append("")
        else:
            col.append(row[n])
    return col


def transpose(data):
    """transpose tabular data
    """
    colcount = len(data[0])
    return [column(data, c) for c in range(colcount)]


def combined_key_subset(cols, df: pd.DataFrame):
    """creates subset of dataframe based on fields in given columns

    It will look for fields in given columns into dataframe.
    and subset the dataframe where the match is found.
       Parameters
       ----------

          cols: list(pd.Series)
            Collection of series of type str 

          df: pd.DataFrame
            DataFrame from which  

    """

    if isinstance(cols[0], pd.Series):
        names = [c.name for c in cols]
    else:  # assume list
        names = [df.columns[i] for i in range(len(cols))]
    rows = zip(*cols)
    r = pd.DataFrame({c: [] for c in df.columns})

    for row in rows:
        r = pd.concat([r, df.query(create_query(names, row))])
    return r


def drop_columns(tabular, n):
    """drops first n items from each row and returns new tabular data

    >>> drop_columns([[1, 2, 3],
                      [21, 22, 23],
                      [31, 32, 33]],
                     1)
    [[2, 3], [22, 23], [32, 33]]
    """
    return [row[n:] for row in tabular]


def take_columns(tabular, n):
    return [row[:n] for row in tabular]


def flatten(list_of_lists):
    return sum(list_of_lists, [])


def zip_columns(tabular):
    return [tuple(row) for row in tabular]


def unique(values):
    """Tell whether values are unique or not
    """
    return len(set(values)) == len(values)


def get_col(data, name):
    if not isinstance(data, pd.DataFrame):
        return []
    else:
        return data[name]
