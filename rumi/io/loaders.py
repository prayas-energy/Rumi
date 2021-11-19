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
"""module which has all loaders for io layer
this module can depend only on python modules and functionstore,
filemanager, config
"""
import os
import csv
import importlib
import yaml
import click
import pandas as pd
from rumi.io import config
from rumi.io import filemanager
from rumi.io.functionstore import transpose, column, unique, concat, x_in_y
from rumi.io.functionstore import circular, valid_date
import functools
import logging
from rumi.io.logger import init_logger

logger = logging.getLogger(__name__)


class LoaderError(Exception):
    pass


def eval_(validation, g=None, l=None):
    statement = validation['code']
    if not g:
        g = {}
    if not l:
        l = {}
    try:
        return eval(statement, g, l)
    except Exception as e:
        logger.error("Failed to evaluate statement " + statement)
        logger.exception(e)


def load_param(param_name: str):
    """load parameter from file in RAW format

        Parameters
        ----------
        param_name : str
           parameter name

        Returns
        -------
        Parameter data from file
    """
    specs = filemanager.get_specs(param_name)
    nested = specs.get('nested')
    if nested and '$' not in nested:
        subfolder = specs.get('nested')
    else:
        subfolder = None

    filepath = filemanager.find_filepath(param_name, subfolder)

    logger.debug(f"Reading {param_name} from file {filepath}")

    if specs.get("optional") and not os.path.exists(filepath):
        logger.warning(
            f"Unable to find file for optional parameter {param_name}")
        return None

    if specs.get("noheader"):
        return read_headerless_csv(param_name, filepath)
    else:
        return read_csv(param_name, filepath)


def load_dataframe(param, filepath, specs):
    with open(filepath) as f:
        csvf = csv.DictReader(f)
        filecols = csvf.fieldnames
        data = {}
        for row in csvf:
            columns = {k: v for k,
                       v in specs['columns'].items() if k in filecols}
            for key, col in columns.items():
                if key not in row:
                    if col.get("optional"):
                        continue
                    else:
                        raise LoaderError(
                            f"column {key} expected in {param}, but not found.")
                else:
                    convert = eval(columns[key]['type'])
                    try:
                        data.setdefault(key, []).append(
                            convert(row[key]))
                    except ValueError as v:
                        logger.error(
                            f"In {param}, could not convert {row[key]} for {key}")

                        data.setdefault(key, []).append(
                            convert(col['default']))
        cols = [c for c in filecols if c in specs['columns']]
        return pd.DataFrame(data)[cols]


def param_env(data):
    env = {c: data[c] for c in data.columns}
    #env['rows'] = data.to_dict(orient='records')

    return env


def validate_each_item(param: str, spec: dict, data):
    """check if min/max boundaries are satisfied

    evaluates whether data is in min/max limits
    in yaml specifications

    Parameters
    ----------
      param : str
        Name of parameter
      spec : dict
        dictinary of specifications for param
      data : pd.DataFrame/[]
        data for the parameters

    Returns
    -------
      True if data is within min/max boubndaries

    """
    if not isinstance(data, pd.DataFrame) and not data:
        if spec.get('optional'):
            logger.warning(f"No data found for optional {param}")
            return True
        else:
            logger.error(f"No data found for {param}")
            return False

    if not spec.get("noheader"):
        for column_, metadata in spec['columns'].items():
            if column_ not in data.columns:
                if metadata.get('optional'):
                    continue
                else:
                    logger.error(
                        f"Expected column {column_} not found in {param}")
                    raise Exception(
                        f"Expected column {column_} not found in {param}")
            c = data[column_]

            if 'min' in spec['columns'].get(column_, {}):
                m = spec['columns'][column_]['min']
                default = spec['columns'][column_].get('default')
                if (c < m).all() and (c[c < m] != default).all():
                    logger.error(
                        f"for {param}, {column_} should be >= {m}")
                    return False

            if 'max' in spec['columns'].get(column_, {}):
                m = spec['columns'][column_]['max']
                if (c > m).all():
                    logger.error(
                        f"For {param}, {column_} should be <= {m}")
                    return False
    return True


def validate_param(param: str, spec: dict, data, module, **kwargs):
    """validate individual parameter data

    evaluates every condition form validations given
    in yaml specifications

    Parameters
    ----------
      param : str
        Name of parameter
      spec : dict
        dictinary of specifications for param
      data : pd.DataFrame/[]
        data for the parameters
      module: string
        definations from this module will be available in validation code
      **kwargs:
        any additional item that should be available in validation code

    """
    logger.info(f"Validating {param}")
    valid = validate_each_item(param, spec, data)

    for validation in spec.get('validation', []):
        if spec.get('noheader'):
            env = {param: data}
        else:
            env = param_env(data)
            env[param] = data

        env.update({p: get_parameter(p) for p in spec.get("dependencies", [])})
        load_module("rumi.io.functionstore", env)
        load_module(module, env)
        env.update(kwargs)

        env.update(globals())
        if not eval_(validation, env):
            logger.error(f"Invalid data for {param}")
            logger.error("{} failed".format(validation['code']))
            # print(validation['message'].format(**env))
            message = validation['message']
            print(eval(f"f'{message}'", env))
            logger.error(eval(f"f'{message}'", env))
            valid = False

    return valid


def load_module(module, env):
    """loads functions from given module in env
    """
    m = importlib.import_module(module)
    for function in dir(m):
        env[function] = getattr(m, function)


def load_namespace(namespace_defs, env):
    for key, value in namespace_defs.items():
        env[key] = eval(value, env)


def get_params(specs):
    params = {}
    for param in [p for p in specs.keys() if p != 'global_validation']:
        try:
            params[param] = get_parameter(param, validation=True)
        except FileNotFoundError as fn:
            logger.exception(fn)
            raise fn
        except filemanager.FolderStructureError as fse:
            logger.exception(fse)
            raise fse
        except TypeError as tpe:
            logger.debug(f"Automatic loading of {param} failed.")
            raise tpe
    return params


def global_validation(data, global_validation_):
    """validations that depend on multiple parameters
    """
    env = {}
    env.update(data)
    env.update(globals())
    load_module(global_validation_['module'], env)
    load_module("rumi.io.functionstore", env)

    include = global_validation_.get("include", [])
    for type_ in include:
        s = filemanager.get_type_specs(type_)
        env.update(get_params(s))
        load_namespace(s['global_validation'].get('namespace', {}), env)

    load_namespace(global_validation_.get('namespace', {}), env)

    validations = global_validation_['validation']
    valid = True
    for validation in validations:
        if not eval_(validation, env):
            print(validation['message'])
            valid = False
            logger.error(f"Global validation failed for {validation['code']}")
            logger.error(validation['message'])

    return valid


def validate_params(param_type):
    """ validate all prameters

         Parameters
         ----------

           param_type: str
              one of Common, Demand, Supply

           specs_file: str
             yaml file path

         Returns
         -------
         returns True if all paramters are valid, else returns False

    """
    logger.info(f"Validating {param_type}")
    print(f"Validating {param_type}")
    allspecs = dict(filemanager.get_type_specs(param_type))
    gvalidation = allspecs['global_validation']
    del allspecs['global_validation']
    data = get_params(allspecs)

    valid = True
    for param, d in data.items():
        try:
            if isinstance(d, type(None)):
                continue
                # for complicated paramters with variable nested folders
                # skip individual validation
            else:
                valid = valid and validate_param(
                    param, allspecs[param], d, gvalidation['module'])
        except Exception as e:
            valid = False
            print(f"Error occured while validating {param}")
            logger.error(f"Error occured while validating {param}")
            logger.exception(e)
            raise e

    return global_validation(data, gvalidation) and valid


def call_loader(loaderstring, **kwargs):
    functionname = loaderstring.split(".")[-1]
    module = ".".join(loaderstring.split(".")[:-1])
    m = importlib.import_module(module)
    loader_function = getattr(m, functionname)
    return loader_function(**kwargs)


def get_config_parameter(param_name):
    path = filemanager.get_config_parameter_path(param_name)
    return pd.read_csv(path)


def call_loader_(specs, param_name, **kwargs):
    try:
        if not specs.get('nested'):
            d = call_loader(specs.get('loader'))
        elif '$' not in specs.get('nested'):
            d = call_loader(specs['loader'],
                            param_name=param_name,
                            subfolder=specs.get('nested'))
        elif "$" in specs.get('nested'):
            if "validation" in kwargs and kwargs['validation'] == True:
                d = None
            else:
                d = call_loader(specs.get('loader'), **kwargs)
    except FileNotFoundError as fne:
        if specs.get('optional'):
            d = None
        else:
            raise fne
    return d


@functools.lru_cache(maxsize=None)
def get_parameter(param_name, **kwargs):
    """ returns data for given parameter. It returns final expanded data.

    except noheader kind of parameter, everything it returns is pandas
    DataFrame.
    for header less parameter, it returns dictionary with first item
    on every row as key and list of rest items as value.

    examples
    --------
    ::

       get_parameter('GDP') -> will return GDP parameter as a DataFrame
       get_parameter('SubGeography1') -> will return SubGeography1 parameter as a list
       get_parameter('SubGeography2') -> will return SubGeography2 parameter as a dictionary, keys are regions and values are list of states
       get_parameter('BaseYearDemand', 
                      demand_sector='D_AGRI') -> BaseYearDemand parameter for 'D_AGRI' as DataFrame 
       get_parameter('NumInstances', 
                      demand_sector='D_RES',
                      energy_service='RES_COOL') -> NumInstances parameter for <'D_RES','RES_COOL'> as DataFrame


    :param: param_name
    :param: `**kwargs` - variable number of named arguments
    :returns: DataFrame or list or dictionary

    """

    #logger.debug("Getting Parameter " + param_name + str(kwargs))
    specs = filemanager.get_specs(param_name)
    if specs.get('loader'):
        d = call_loader_(specs, param_name, **kwargs)
    else:
        d = load_param(param_name)

    if d is None:
        r = d
    elif specs.get("noheader"):
        r = reformat_headerless(param_name, specs, d)
    else:
        r = d
    return filter_param(param_name, r)


def reformat_headerless(param_name, specs, d):
    """Formate headerless data to list/dictionary/string as required
    """
    if specs.get("map"):
        firstcolumn = column(d, 0)
        if not unique(firstcolumn):
            repeating = set(
                [c for c in firstcolumn if firstcolumn.count(c) > 1])
            logger.warning(
                f"First column in {param_name} should not repeat, but repeating rows discovered for {repeating}")
            logger.warning(
                f"For {param_name} last item from repeating rows of {repeating} will be considered")
        r = {key: d[r][1:] for r, key in enumerate(column(d, 0))}
    elif specs.get("list"):
        r = d[0]
        if len(d) > 1:
            logger.warning(
                f"Parameter {param_name} expects only one row but found multiple rows. Only first row will be considered")
    elif len(d) == 1 and len(d[0]) == 1:
        r = d[0][0]
    else:
        r = d
    return r


def filter_param(param_name, param_data):
    """This functions filters parameter based on scheme given
    in yaml specifications.

    caution: this function creates a circular dependency by
    calling get_parameter again. SO IF SELF REFERENCING DEPENDENCIES ARE GIVEN
    IT MIGHT RESULT IN RECURSION ERROR.
    """
    specs = filemanager.get_specs(param_name)
    if specs.get("filterqueries") and isinstance(param_data, pd.DataFrame):
        logger.debug(f"Filtering parameter {param_name}")
        dependencies = specs.get("dependencies")
        dependencies_data = {p: get_parameter(p) for p in dependencies}
        queries = specs.get("filterqueries")
        dependencies_data['param_data'] = param_data
        queries_ = [f"( {q} )" for q in queries]
        statement = "param_data.query(f\"{0}\")".format(" & ".join(queries_))
        param_data = eval(statement, dependencies_data)  # .copy()
        if len(param_data) == 0:
            logger.warning(
                f"Filtering of {param_name} has resulted in empty data")

    return param_data


def find_cols(filepath, columnsdata):
    """
    find columns common between column names provided in specifications
    and those given in file.
    """
    with open(filepath) as f:
        csvf = csv.reader(f)
        columnsf = next(csvf)
        # columns as per order in file
        return [c for c in columnsf if c in columnsdata]


def read_headerless_csv(param_name, filepath):
    try:
        with open(filepath) as f:
            csvf = csv.reader(f)
            return [row for row in csvf]
    except ValueError as v:
        logger.error(f"Unable to parse data for {param_name}")
        logger.exception(v)
        raise v
    except FileNotFoundError as fne:
        logger.error(f"Unable to find file for {param_name}")
        logger.exception(fne)
        raise fne
    except Exception as e:
        logger.error(f"Falied to read parameter {param_name}")
        logger.exception(e)
        raise e


def read_csv(param_name, filepath):
    """read dataframe using pandas.read_csv, but with appropriate types
    """
    specs = filemanager.get_specs(param_name)
    columndata = specs['columns']
    converters = {c: eval(data['type']) for c, data in columndata.items()}
    try:
        cols = find_cols(filepath, columndata)
        return pd.read_csv(filepath,
                           usecols=cols,
                           converters=converters,
                           na_values="")
    except ValueError as v:
        logger.error(f"Unable to parse data for {param_name}")
        logger.exception(v)
        raise v
    except FileNotFoundError as fne:
        if specs.get('optional'):
            logger.warning(
                f"Unable to find file for optional parameter {param_name}")
        else:
            logger.error(f"Unable to find file for {param_name}")
            logger.exception(fne)
        raise fne
    except Exception as e:
        logger.error(f"Falied to read parameter {param_name}")
        logger.exception(e)
        raise e


def rumi_validate(param_type: str,
                  model_instance_path: str,
                  scenario: str,
                  logger_level: str):
    """Function to validate Common or Demand or Supply
    """
    global logger
    config.initialize_config(model_instance_path, scenario)
    init_logger(param_type, logger_level)
    logger = logging.getLogger("rumi.io.loaders")

    print(validate_params(param_type))


@click.command()
@click.option("-p", "--param_type",
              help="Parameter type to validate. can be one of Common, Demand or Supply")
@click.option("-m", "--model_instance_path",
              help="Path where model instance is stored")
@click.option("-s", "--scenario",
              help="Name of Scenario")
@click.option("-l", "--logger_level",
              help="Level for logging,one of INFO,WARN,DEBUG,ERROR.",
              default="INFO")
def main(param_type: str,
         model_instance_path: str,
         scenario: str,
         logger_level: str):
    """Command line interface for data validation.
    """
    rumi_validate(param_type,
                  model_instance_path,
                  scenario,
                  logger_level)


if __name__ == "__main__":
    main()
