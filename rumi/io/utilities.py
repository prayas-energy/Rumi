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
"""utilities used by all common, demand and supply.
this should not depend on common,demand or supply modules.
"""
import sys
from io import StringIO
import datetime
import pandas as pd
from rumi.io import loaders
from rumi.io import constant
from rumi.io.logger import logging
from rumi.io import functionstore as fs
import numpy as np
import itertools
import functools
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger(__name__)
print = functools.partial(print, flush=True)


class InValidColumnsError(Exception):
    pass


class InvalidCGTDataError(Exception):
    pass


def debug(f):
    def wrapper(*args, **kwargs):
        print("start", f.__qualname__)
        print("args", args, kwargs)
        r = f(*args, **kwargs)
        print("results", r)
        print("finish", f.__qualname__)
        return r
    return wrapper


def get_all_structure_columns(data, index_cols=None):
    if not index_cols:
        index_cols = []
    c = get_consumer_columns_from_dataframe(data)
    g = get_geographic_columns_from_dataframe(data)
    t = get_time_columns_from_dataframe(data)
    return index_cols+c+g+t


def base_dataframe(structural_cols,
                   demand_sector=None,
                   colname="dummy",
                   val=np.nan,
                   extracols_df=None):
    conscols = [c for c in structural_cols if c in constant.CONSUMER_TYPES]
    geocols = [c for c in structural_cols if c in constant.GEOGRAPHIES]
    timecols = [c for c in structural_cols if c in constant.TIME_SLICES]
    return base_dataframe_all(conscols=conscols,
                              geocols=geocols,
                              timecols=timecols,
                              demand_sector=demand_sector,
                              colname=colname,
                              val=val,
                              extracols_df=extracols_df)


def base_dataframe_of_granularity(CGRAN=None,
                                  GGRAN=None,
                                  TGRAN=None,
                                  demand_sector=None,
                                  colname="dummy",
                                  val=np.nan,
                                  extracols_df=None):
    """return base dataframe for given granuarity
    """
    conscols = get_consumer_columns(CGRAN) if CGRAN else []
    geocols = get_geographic_columns(GGRAN) if GGRAN else []
    timecols = get_time_columns(TGRAN) if TGRAN else []
    return base_dataframe_all(conscols=conscols,
                              geocols=geocols,
                              timecols=timecols,
                              demand_sector=demand_sector,
                              colname=colname,
                              val=val,
                              extracols_df=extracols_df)


def base_dataframe_all(conscols=None,
                       geocols=None,
                       timecols=None,
                       demand_sector=None,
                       colname="dummy",
                       val=np.nan,
                       extracols_df=None):
    return BaseDataFrame(conscols=conscols,
                         geocols=geocols,
                         timecols=timecols,
                         demand_sector=demand_sector,
                         colname=colname,
                         val=val,
                         extracols_df=extracols_df).get_dataframe()


class BaseDataFrame:

    def __init__(self,
                 conscols=None,
                 geocols=None,
                 timecols=None,
                 demand_sector=None,
                 colname="dummy",
                 val=np.nan,
                 extracols_df=None):
        self.conscols = [] if not conscols else conscols
        self.geocols = [] if not geocols else geocols
        self.timecols = [] if not timecols else timecols
        self.demand_sector = demand_sector
        self.colname = colname
        self.val = val
        self.extracols_df = extracols_df

    def product_consumers(self):
        return compute_product_consumers(self.conscols, self.demand_sector)

    def product_geo(self):
        return compute_product_geo(self.geocols)

    def product_time(self):
        return compute_product_time(self.timecols)

    def get_dataframe(self):
        cp = self.product_consumers() if self.conscols else [tuple()]
        gp = self.product_geo() if self.geocols else [tuple()]
        tp = self.product_time() if self.timecols else [tuple()]

        if fs.isnone(self.extracols_df):
            ep = [tuple()]
            extracols_names = []
        else:
            ep = [tuple(row) for row in self.extracols_df.values]
            extracols_names = list(self.extracols_df.columns)

        def tup(x):
            return x if isinstance(x, tuple) else (x,)

        p = [tup(c)+tup(g)+tup(t)+tup(e)
             for c in cp for g in gp for t in tp for e in ep]

        names = [c for c in constant.CONSUMER_TYPES if c in self.conscols] + \
            [c for c in constant.GEOGRAPHIES if c in self.geocols] + \
            [c for c in constant.TIME_SLICES if c in self.timecols] + extracols_names

        if len(names) == 1:
            index = pd.Index([item[0] for item in p], name=names[0])
        else:
            index = pd.MultiIndex.from_tuples(p,
                                              names=names)
        rows = len(p)

        return pd.DataFrame({self.colname: [self.val]*rows},
                            index=index)


def unique_across(data, columns):
    """checks if given columns combination has unique items
    """
    geocols = get_geographic_columns_from_dataframe(data)
    timecols = get_time_columns_from_dataframe(data)
    conscols = get_consumer_columns_from_dataframe(data)
    cols = geocols + timecols + conscols + columns
    return len(data) == len(data.drop_duplicates(subset=cols))


def find_interval(start, end):
    m1, d1 = start
    m2, d2 = end

    y = 2019  # nonleap year
    interval = datetime.datetime(y, m2, d2) - datetime.datetime(y, m1, d1)
    if interval.days < 0:
        return interval.days + 365
    return interval.days


def get_valid_time_levels():
    """Checks files defined in Common and decides what time levels to return
    """
    if isinstance(loaders.get_parameter("DaySlices"), pd.DataFrame):
        levels = [s.upper() for s in constant.TIME_SLICES]
    elif isinstance(loaders.get_parameter("DayTypes"), pd.DataFrame):
        levels = [s.upper() for s in constant.TIME_SLICES[:3]]
    elif isinstance(loaders.get_parameter("Seasons"), pd.DataFrame):
        levels = [s.upper() for s in constant.TIME_SLICES[:2]]
    else:
        levels = [s.upper() for s in constant.TIME_SLICES[:1]]

    return levels


def get_valid_geographic_levels():
    """returns valid geographic levels in system.
    checks actual files defined in common and accordingly
    find valid geographic levels.

    Returns
    -------
    a list containing valid geographic granularities
    """
    if loaders.get_parameter("SubGeography3"):
        levels = [s.upper() for s in constant.GEOGRAPHIES]
    elif loaders.get_parameter("SubGeography2"):
        levels = [s.upper() for s in constant.GEOGRAPHIES[:3]]
    elif loaders.get_parameter("SubGeography1"):
        levels = [s.upper() for s in constant.GEOGRAPHIES[:2]]
    else:
        levels = [s.upper() for s in constant.GEOGRAPHIES[:1]]

    return levels


def get_pair(season):
    """return month and date pair
    """
    return season['StartMonth'], season['StartDate']


def get_consumer_columns(consumer_granularity):
    return constant.CONSUMER_COLUMNS[consumer_granularity]


def get_geographic_columns(geographic_granularity):
    return constant.GEO_COLUMNS[geographic_granularity]


def get_time_columns(time_granularity):
    return constant.TIME_COLUMNS[time_granularity]


def compute_intervals(Seasons):
    seasons = Seasons.to_dict(orient='records')
    seasons.append(seasons[0])
    return {s['Season']: find_interval(get_pair(s), get_pair(seasons[i+1]))
            for i, s in enumerate(seasons[:-1])}


def seasons_size():
    Seasons = loaders.get_parameter("Seasons")
    return compute_intervals(Seasons)


def valid_geography(dataframe):
    """This checks validity of geography columns, this works
    only for datasets in which there is no other repeating entity.
    so this will work only for for data like GDP and DemoGraphics
    """
    geocols = get_geographic_columns_from_dataframe(dataframe)
    timecols = get_time_columns_from_dataframe(dataframe)
    basegeodata = base_dataframe_geography(geocols).reset_index()
    base = basegeodata[geocols].sort_values(geocols).reset_index(drop=True)
    if not timecols:
        return dataframe[geocols].sort_values(geocols).reset_index(drop=True).eq(base).all().all()
    data = dataframe.set_index(timecols)[geocols]
    valid = True

    for t in data.index.unique():
        d = data.loc[t]
        d = d.sort_values(geocols).reset_index(drop=True)
        valid = valid and d.eq(base).all().all()
    return valid


def filter_by(data, param_name, colname):
    param = loaders.get_parameter(param_name)

    def filter_(x):
        return x in param[colname].values

    return data[data[colname].apply(filter_)]


def filter_empty(data, all=True):
    """filters out empty columns from dataframe"""
    rows = len(data)
    empty = [c for c in data.columns if data[c].isnull(
    ).sum() == rows or (data[c] == "").sum() == rows]
    return data[[c for c in data.columns if c not in empty]]


def get_geographic_columns_from_dataframe(data):
    return [c for c in constant.GEOGRAPHIES if c in data.columns]


def get_consumer_columns_from_dataframe(data):
    return [c for c in constant.CONSUMER_TYPES if c in data.columns]


def get_time_columns_from_dataframe(data):
    return [c for c in constant.TIME_SLICES if c in data.columns]


def make_list(target):
    if isinstance(target, list):
        return target
    else:
        return [target]


def groupby_time(data,
                 groupcols,
                 balancing_time,
                 target=None):

    if not target:
        target = [c for c in data.columns if data[c].dtype == np.float64]

    if balancing_time == 'YEAR':
        grouping = groupcols + ['Year']
    elif balancing_time == 'SEASON':
        grouping = groupcols + ['Year', 'Season']
    elif balancing_time == 'DAYTYPE':
        grouping = groupcols + ['Year', 'Season', 'DayType']
    else:
        grouping = groupcols + ['Year', 'Season', 'DayType', 'DaySlice']

    data = groupby(data, grouping, target)
    return data.reset_index()


def groupby(data: pd.DataFrame,
            groupcols: list,
            target):
    targets = make_list(target)
    othercols = [c for c in groupcols if c not in constant.TIME_SLICES]
    destgran = [c for c in constant.TIME_SLICES if c in groupcols][-1]
    sourcegran = get_time_columns_from_dataframe(data)[-1]
    if (sourcegran in ['DayType', 'DaySlice']) and\
       (destgran in ['Season', 'Year']):
        seasons_size_ = pd.Series(seasons_size())
        seasons_size_.index.rename('Season', inplace=True)
        DayTypes = loaders.get_parameter('DayTypes')
        weights = DayTypes.set_index("DayType")['Weight']
        cols = othercols + ['DayType', 'Season', 'Year']
        if 'DaySlice' in data.columns:
            cols.append('DaySlice')
        data = data.set_index(cols)
        for item in targets:
            data[item] = data[item]*seasons_size_*weights
        data = data.reset_index()

    return data.groupby(groupcols,
                        sort=False).sum(numeric_only=True)[target]


def order_by(v):
    flag = True
    if v.name == 'Season':
        order = get_order('Season', 'Seasons')
    elif v.name == 'DayType':
        order = get_order('DayType', 'DayTypes')
    elif v.name == 'DaySlice':
        order = get_order('DaySlice', 'DaySlices')
    elif v.name == 'SubGeography1':
        order = geo_order('SubGeography1')
    elif v.name == 'SubGeography2':
        order = geo_order('SubGeography2')
    elif v.name == 'SubGeography3':
        order = geo_order('SubGeography3')
    else:
        flag = False

    if flag:
        return v.apply(order.index)

    return v


def get_order(col, param):
    parameter = loaders.get_parameter(param)
    return list(parameter[col].values) + [np.NaN]


def geo_order(col):
    parameter = loaders.get_parameter(col)
    if col == 'SubGeography1':
        return parameter + [np.NaN]
    else:
        return sum(parameter.values(), []) + [np.NaN]


def order_columns(columns):
    """order the columns in TIME, GEOGRAPHIES and CONSUMER_TYPES and then REST
    """
    all_cols = constant.TIME_SLICES + constant.GEOGRAPHIES + constant.CONSUMER_TYPES
    TGC_cols = [c for c in all_cols if c in columns]
    remaining_cols = [c for c in columns if c not in TGC_cols]
    return TGC_cols + remaining_cols


def get_ordered_cols(df):
    return order_columns(df.columns)


def order_rows(df):
    """orders dataframe rows by sorting by T*,G*,C* and order to be taken as
    specified in common specs.
    e.g. Order of Seasons will be taken from "Seasons' parameter
    Order of DayType will be taken from "DatTypes' parameter
    Order of  DaySlice will be taken from "DaySlices' parameter
    SubGeography1 order will be taken as given in 'SubGeography1' parameter
    SubGeography2 order will be taken as given in 'SubGeography2' parameter
    SubGeography3 order will be taken as given in 'SubGeography3' parameter
    """
    all_cols = constant.TIME_SLICES + constant.GEOGRAPHIES
    dataset_cols = [c for c in all_cols if c in df.columns]
    return df.sort_values(by=dataset_cols, key=order_by, ignore_index=True)


def get_modelgeography_product():
    data = {c: loaders.get_parameter(c) for c in ['ModelGeography']}
    return (data['ModelGeography'],)


def get_subgeography1_product():
    data = {c: loaders.get_parameter(c)
            for c in ['ModelGeography', 'SubGeography1']}
    return [(data['ModelGeography'], item) for item in data['SubGeography1']]


def get_subgeography2_product():
    data = {c: loaders.get_parameter(c)
            for c in ['ModelGeography', 'SubGeography1', 'SubGeography2']}
    x = [(data['ModelGeography'], item) for item in data['SubGeography1']]
    l = []
    for item1 in x:
        l.extend([item1 + (item2,)
                  for item2 in data['SubGeography2'][item1[-1]]])
    return l


def get_subgeography3_product():
    data = {c: loaders.get_parameter(c)
            for c in ['ModelGeography', 'SubGeography1', 'SubGeography2', 'SubGeography3']}
    x = [(data['ModelGeography'], item) for item in data['SubGeography1']]
    l = []
    for item1 in x:
        l.extend([item1 + (item2,)
                  for item2 in data['SubGeography2'][item1[-1]]])

    m = []
    for item1 in l:
        m.extend([item1 + (item2,)
                  for item2 in data['SubGeography3'][item1[-1]]])
    return m


def compute_product_geo(geocols):
    if set(geocols) == {'ModelGeography'}:
        return get_modelgeography_product()
    elif set(geocols) == {'ModelGeography', 'SubGeography1'}:
        return get_subgeography1_product()
    elif set(geocols) == {'ModelGeography', 'SubGeography1', 'SubGeography2'}:
        return get_subgeography2_product()
    elif set(geocols) == {'ModelGeography',
                          'SubGeography1',
                          'SubGeography2',
                          'SubGeography3'}:
        return get_subgeography3_product()
    else:
        raise InValidColumnsError("Invalid geographies", geocols)


def base_dataframe_geography(geocols, colname="dummy", val=np.nan):
    """Useful for creating dataframes of given gegraphic columns
    """
    return base_dataframe_all(geocols=geocols,
                              colname=colname,
                              val=val)


def get_years():
    modelperiod = loaders.get_parameter('ModelPeriod').iloc[0]
    years = range(modelperiod['StartYear'], modelperiod['EndYear']+1)
    return years


def get_seasons():
    return tuple(loaders.get_parameter('Seasons')['Season'].values)


def get_daytypes():
    return tuple(loaders.get_parameter('DayTypes')['DayType'].values)


def get_dayslices():
    return tuple(loaders.get_parameter('DaySlices')['DaySlice'].values)


def compute_product_time(timecols):
    if set(timecols) == {'Year'}:
        return get_years()
    elif set(timecols) == {'Year', 'Season'}:
        return list(itertools.product(get_years(),
                                      get_seasons()))
    elif set(timecols) == {'Year', 'Season', 'DayType'}:
        return list(itertools.product(get_years(),
                                      get_seasons(),
                                      get_daytypes()))
    elif set(timecols) == {'Year', 'Season', 'DayType', 'DaySlice'}:
        return list(itertools.product(get_years(),
                                      get_seasons(),
                                      get_daytypes(),
                                      get_dayslices()))
    else:
        raise InValidColumnsError("Invalid time slices", timecols)


def base_dataframe_time(timecols, colname='dummy', val=np.nan):
    """Useful for creating datasets of given time columns
    """
    return base_dataframe_all(timecols=timecols,
                              colname=colname,
                              val=val)


def make_dataframe(datastr):
    with StringIO(datastr) as f:
        return pd.read_csv(f)


def check_granularity_per_entity(d,
                                 entity,
                                 GeographicGranularity,
                                 TimeGranularity,
                                 ConsumerGranularity=None):
    """check if the entity specified is exactly equal to granularities specified
    """
    geo_columns, time_columns, cons_columns = [], [], []

    if GeographicGranularity:
        geo_columns = get_geographic_columns(GeographicGranularity)
        dataset_columns = [c for c in d.columns if c in constant.GEOGRAPHIES]
    if TimeGranularity:
        time_columns = get_time_columns(TimeGranularity)
        dataset_columns.extend(
            [c for c in d.columns if c in constant.TIME_SLICES])
    if ConsumerGranularity:
        cons_columns = constant.CONSUMER_COLUMNS[ConsumerGranularity]
        dataset_columns.extend(
            [c for c in d.columns if c in constant.CONSUMER_TYPES])

    diff1 = set(geo_columns + time_columns +
                cons_columns) - set(dataset_columns)
    diff2 = set(dataset_columns) - \
        set(geo_columns + time_columns + cons_columns)

    valid = True
    if diff1:
        logger.error(
            f"{diff1} columns not found in data for {entity}")
        valid = False
    else:
        # redundant!
        nonempty = d[geo_columns+time_columns].isnull().sum().sum()
        valid = nonempty == 0

    if diff2:
        c, r = d[list(diff2)].shape
        empty = d[list(diff2)].isnull().sum().sum() == c*r

        if not empty:
            line1 = f"Granularity is finer than expected for {entity}!"
            line2 = f"It has these columns extra, {diff2}"
            logger.error("\n".join([line1, line2]))

        valid = valid and empty

    return valid


def get_consumertype2_product(demand_sector):
    consumertype1 = get_consumertype1_product(demand_sector)
    Cons1_Cons2_Map = loaders.get_parameter("Cons1_Cons2_Map",
                                            demand_sector=demand_sector)

    return sum([[(t1, t2) for t2 in Cons1_Cons2_Map.get(t1, [])] for t1 in consumertype1], [])


def get_consumertype1_product(demand_sector):
    DS_Cons1_Map = loaders.get_parameter("DS_Cons1_Map",
                                         demand_sector=demand_sector)
    return DS_Cons1_Map[demand_sector][2:]


def compute_product_consumers(conscols, demand_sector):
    if set(conscols) == {'ConsumerType1'}:
        return get_consumertype1_product(demand_sector)
    elif set(conscols) == {'ConsumerType1', 'ConsumerType2'}:
        return get_consumertype2_product(demand_sector)
    else:
        raise InValidColumnsError(f"Invalid consumer columns, {conscols}")


def base_dataframe_consumers(conscols,
                             demand_sector,
                             colname='dummy',
                             val=np.nan):
    """Useful for creating datasets of given consumer columns
    """
    return base_dataframe_all(conscols=conscols,
                              demand_sector=demand_sector,
                              colname=colname,
                              val=val)


def balancing_X(EC, X):
    PhysicalPrimaryCarriers = loaders.get_parameter('PhysicalPrimaryCarriers')
    PhysicalDerivedCarriers = loaders.get_parameter('PhysicalDerivedCarriers')
    NonPhysicalDerivedCarriers = loaders.get_parameter(
        'NonPhysicalDerivedCarriers')

    df = None

    if EC in list(PhysicalPrimaryCarriers.EnergyCarrier):
        df = PhysicalPrimaryCarriers
    elif EC in list(PhysicalDerivedCarriers.EnergyCarrier):
        df = PhysicalDerivedCarriers
    elif EC in list(NonPhysicalDerivedCarriers.EnergyCarrier):
        df = NonPhysicalDerivedCarriers
    else:
        raise Exception(f"No such energy carrier, {EC}!")

    # return df.query(f"EnergyCarrier == '{EC}'")[X].iloc[0]
    xdata = df.set_index('EnergyCarrier')
    return xdata[X][EC]


def balancing_time(EC):
    return balancing_X(EC, "BalancingTime")


def balancing_area(EC):
    return balancing_X(EC, "BalancingArea")


def get_cols_from_dataframe(data, type_):
    f = {"C": [c for c in constant.CONSUMER_TYPES if c in data.columns],
         "G": [c for c in constant.GEOGRAPHIES if c in data.columns],
         "T": [c for c in constant.TIME_SLICES if c in data.columns]}
    return f[type_]


def get_base_dataframe(cols, type_, demand_sector=None):
    f = {"C": base_dataframe_consumers,
         "G": base_dataframe_geography,
         "T": base_dataframe_time}
    if type_ == 'C':
        return f[type_](cols, demand_sector, val=0)
    return f[type_](cols, val=0)


def subset_multi(data, indexnames, items):
    if isinstance(items, str) or isinstance(items, int):
        items = (items,)

    q = []
    for name, item in zip(indexnames, items):
        if isinstance(item, str):
            q.append(f"{name} == '{item}'")
        else:
            q.append(f"{name} == {item}")
    return data.query(" & ".join(q))


def get_set(df):
    return set(df.itertuples(index=False, name=None))


def check_CGT_validity(data,
                       name,
                       entity,
                       type_,
                       demand_sector=None,
                       checkunique=True,
                       exact=False):
    """
    check if consumertype, geographies and time columns have appropriate values
    """

    if isinstance(entity, str):
        entity = [entity]

    valid = True

    def check(subset__, typecols, item):
        if not typecols:
            return True
        else:
            subset__ = subset__[typecols]
            basedf = get_base_dataframe(
                tuple(typecols), type_, demand_sector).reset_index()
            if exact:
                if 'dummy' in basedf.columns:
                    del basedf['dummy']
                v = get_set(subset__) == get_set(basedf)
            else:
                v = subset__.isin(basedf.to_dict(
                    orient='list')).all().all()
            if checkunique and len(subset__.drop_duplicates()) != len(subset__):
                if item:
                    msg = f"{name} parameter for {item} has duplicate rows in {typecols} columns"
                else:
                    msg = f"{name} parameter has duplicate rows in {typecols} columns"
                logger.error(msg)
                v = False
        return v

    def make_iterable(items):
        if isinstance(items, (int, str, float)) or items is None:
            return [items]
        else:
            return items

    def loop_over_rest_structural_cols(subset, item=None):
        timecols = get_time_columns_from_dataframe(subset)
        conscols = get_consumer_columns_from_dataframe(subset)
        geographiccols = get_geographic_columns_from_dataframe(subset)
        allcols = {"C": conscols, "G": geographiccols, "T": timecols}
        typecols = allcols[type_]
        indexcols = fs.flatten([v for k, v in allcols.items() if k != type_])
        if indexcols:
            sdf = subset.set_index(indexcols)
            valid = True
            for items_ in sdf.index.unique():
                subset_ = subset_multi(subset, indexcols, items_)
                v = check(subset_, typecols, item)
                valid = valid and v
                if not v:
                    logger.error(
                        f"{name} parameter for {item} has invalid data for {typecols} columns")
        else:
            valid = check(subset, typecols, None)
            if not valid:
                logger.error(
                    f"{name} parameter  has invalid data for {typecols} columns")
        return valid

    if not entity:
        subset = filter_empty(data)
        return loop_over_rest_structural_cols(subset)
    else:
        df = data.set_index(entity)
        valid = True
        for item in df.index.unique():
            subset = filter_empty(subset_multi(
                data, entity, make_iterable(item)))
            # some times entity has single value! so making it list
            valid = valid and loop_over_rest_structural_cols(subset, item)
        return valid


def check_duplicates(data,
                     cols,
                     param_name,
                     entity_values="",
                     demand_sector="",
                     energy_service=""):
    """Checks if given dataframe has duplicate entries in C*,G*,T* columns
    """
    dups = data.duplicated(subset=cols, keep=False)
    if dups.sum() > 0:
        logger.error(
            f"Following rows have duplicate entries in {param_name} from {demand_sector},{energy_service},{entity_values}")
        for row in data[dups].drop_duplicates().values:
            logger.error(row)
        return True
    return False


def override_dataframe_with_check(dataframe1,
                                  dataframe2,
                                  index_cols,
                                  param_name,
                                  entity_values=None,
                                  demand_sector="",
                                  energy_service=""):
    """override data from dataframe2 in dataframe1 using
    index_cols as a key to compare.  if dataframe2 is not subset
    of dataframe1, error is thrown. Also dataframe2 has duplicate
    entries for any CGT combination, then also error is thwon.
    """
    if fs.is_empty_or_none(dataframe2):
        return dataframe1.copy()
    if entity_values is None:
        entity_values = ""

    if check_duplicates(data=dataframe2,
                        cols=index_cols,
                        param_name=param_name,
                        entity_values=entity_values,
                        demand_sector=demand_sector,
                        energy_service=energy_service):
        raise InvalidCGTDataError(
            f"Duplicate entries for {param_name} from {demand_sector},{energy_service},{entity_values} are found in CGT columns")

    cols = [c for c in dataframe1.columns]
    dataframe2 = dataframe2[cols]
    # order by dataframe1 column order
    indexcols = [c for c in cols if c in index_cols]

    dx = dataframe1.to_dict(orient="records")
    dy = dataframe2.to_dict(orient="records")
    ddx = {tuple(r[c] for c in indexcols): r for r in dx}
    ddy = {tuple(r[c] for c in indexcols): r for r in dy}
    diff = ddy.keys() - ddx.keys()

    if diff:
        logger.error(
            f"{param_name} from {demand_sector},{energy_service},{entity_values} has invalid CGT combination as given below")
        for item in diff:
            logger.error(item)
        raise InvalidCGTDataError(
            f"Invalid CGT data for {param_name} from {demand_sector},{energy_service},{entity_values}")

    ddx.update(ddy)
    return pd.DataFrame(ddx.values())


def check_consumer_validity(data,
                            name,
                            entity,
                            demand_sector=None,
                            checkunique=True,
                            exact=False):
    """
    check if consumertype columns have appropriate values
    """

    return check_CGT_validity(data, name, entity, 'C',
                              demand_sector=demand_sector,
                              checkunique=checkunique,
                              exact=exact)


def check_geographic_validity(data,
                              name,
                              entity,
                              checkunique=True,
                              exact=False):
    """
    check if geographies columns have appropriate values
    """
    return check_CGT_validity(data, name, entity, 'G',
                              checkunique=checkunique,
                              exact=exact)


def check_time_validity(data,
                        name,
                        entity,
                        checkunique=True,
                        exact=False):
    """
    check if time columns have appropriate values
    """
    return check_CGT_validity(data, name, entity, 'T',
                              checkunique=checkunique,
                              exact=exact)


def check_balancing_time_gran(param_name,
                              granmap,
                              entity,
                              comp='coarser',
                              find_EC_=lambda x, y: y[-1]):
    """check if given granularity map specifies granularity appropriately as specified
    by comp parameter.

        Parameters
        ----------
        param_name: str
            parameter name , some granularity map
        granmap: pd.DataFrame
            data for given param_name
        entity: str
            one of EnergyCarrier, EnergyConvTech or EnergyStorTech
        comp: str, default 'coarser'
            one of coarser or finer


        Returns
        -------
        bool
           True if comp is 'finer' and granmap has granularity finer than balancing time
           True if comp is 'coarser' and granmap has granularity coarser than balancing time
           else False

    """
    granmap = granmap.set_index(entity)
    for entity_ in granmap.index:
        ec = find_EC_(entity, entity_)
        balacing_gran = balancing_time(ec)
        data_gran = granmap.loc[entity_]['TimeGranularity']

        if comp == "finer":
            if len(constant.TIME_COLUMNS[balacing_gran]) > len(constant.TIME_COLUMNS[data_gran]):
                logger.error(
                    f"For {param_name} time granularity for {entity},{entity_} is incorrect. It should be finer than balancing time of {ec}")
                return False
        else:
            if len(constant.TIME_COLUMNS[balacing_gran]) < len(constant.TIME_COLUMNS[data_gran]):
                logger.error(
                    f"For {param_name} time granularity for {entity},{entity_} is incorrect. It should be coarser than balancing time of {ec}")
                return False
    return True


def check_balancing_area_gran(param_name,
                              granmap,
                              entity,
                              comp='coarser',
                              find_EC_=lambda x, y: y[-1]):
    """check if given granularity map specifies granularity appropriately as specified
    by comp parameter.

        Parameters
        ----------
        param_name: str
            parameter name , some granularity map
        granmap: pd.DataFrame
            data for given param_name
        entity: str
            one of EnergyCarrier, EnergyConvTech or EnergyStorTech
        comp: str, default 'coarser'
            one of coarser or finer


        Returns
        -------
        bool
           True if comp is 'coarser' and granmap has granularity coarser than balancing area
           True if comp is 'finer' and granmap has granularity finer than balancing area
           else False

    """

    granmap = granmap.set_index(entity)
    for entity_ in granmap.index:
        ec = find_EC_(entity, entity_)
        balacing_gran = balancing_area(ec)
        data_gran = granmap.loc[entity_]['GeographicGranularity']
        if comp == "finer":
            if len(constant.GEO_COLUMNS[balacing_gran]) > len(constant.GEO_COLUMNS[data_gran]):
                logger.error(
                    f"For {param_name} geographic granularity for {entity} is incorrect. It should be finer than balancing area of {ec}")
                return False
        else:
            if len(constant.GEO_COLUMNS[balacing_gran]) < len(constant.GEO_COLUMNS[data_gran]):
                logger.error(
                    f"For {param_name} geographic granularity for {entity}, {entity_} is incorrect. It should be coarser than balancing area of {ec}")
                return False

    return True


if __name__ == "__main__":
    pass
