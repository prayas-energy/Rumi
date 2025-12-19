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
from rumi.io import supply
from rumi.io import loaders
import pytest
from pyomo.environ import *
from pyomo.core import *
from pyomo.opt import SolverFactory

import string


def create_model():
    DUMMY_GEOG_STR = ""

    ####    Hard-coded data for simulating skeletal supply model    ####
    StartYear = 2021
    EndYear = 2025

    model_geog_name_list = ["INDIA"]
    subgeog1_names_list = ["ER", "NER", "NR", "SR", "WR"]

    BALAREA_MG_STR = "MODELGEOGRAPHY"
    BALAREA_SG1_STR = "SUBGEOGRAPHY1"
    BALAREA_SG2_STR = "SUBGEOGRAPHY2"
    BALAREA_SG3_STR = "SUBGEOGRAPHY3"

    ec_bal_area_map = {
        "ELECTRICITY": "SUBGEOGRAPHY1",
        "ATF": "MODELGEOGRAPHY",
        "HSD": "MODELGEOGRAPHY",
        "LPG": "MODELGEOGRAPHY",
        "MS": "MODELGEOGRAPHY",
        "OTHERPP": "MODELGEOGRAPHY"
    }

    ect_ec_dict = {
        "EG_PHWR": "ELECTRICITY",
        "EG_LH": "ELECTRICITY",
        "EG_SH": "ELECTRICITY",
        "EG_BIOMASS": "ELECTRICITY",
        "EG_SOLARPV": "ELECTRICITY",
        "EG_WIND": "ELECTRICITY",
        "EG_COAL": "ELECTRICITY",
        "EG_CCGT": "ELECTRICITY",
        "EG_OCGT": "ELECTRICITY",
        "RF_ATF": "ATF",
        "RF_HSD": "HSD",
        "RF_LPG": "LPG",
        "RF_MS": "MS",
        "RF_OTHERPP": "OTHERPP"
    }

    #########################################
    #####################            Model Definition                #############
    #########################################

    model = AbstractModel()

    ###############
    #    Sets     #
    ###############

    def get_year_range(m):
        return list(range(StartYear, EndYear + 1))

    def get_ect_names_list(m):
        # returns the keys of the dictionary as a list
        return [*ect_ec_dict]

    def get_model_geog_name(m):
        return model_geog_name_list

    def get_subgeog1_names_list(m):
        return subgeog1_names_list

    model.Year = Set(initialize=get_year_range, ordered=Set.SortedOrder)
    model.EnergyConvTech = Set(initialize=get_ect_names_list, ordered=True)
    model.ModelGeography = Set(initialize=get_model_geog_name, ordered=True)
    model.SubGeog1 = Set(initialize=get_subgeog1_names_list, ordered=True)

    model.DummyGeog = Set(initialize=[DUMMY_GEOG_STR])

    model.SubGeography1 = model.SubGeog1 | model.DummyGeog

    def define_bal_area_sets_for_level2(m):
        return Set(within=m.ModelGeography * m.DummyGeog,
                   initialize=m.ModelGeography * m.DummyGeog,
                   ordered=True), \
            Set(within=m.ModelGeography * m.SubGeog1,
                initialize=m.ModelGeography * m.SubGeog1,
                ordered=True)

    model.BalAreaMG, model.BalAreaSG1 = define_bal_area_sets_for_level2(model)

    model.BalArea = Set(within=model.ModelGeography * model.SubGeography1,
                        initialize=model.BalAreaMG | model.BalAreaSG1,
                        ordered=True)

    def get_bal_area_inp(ec):
        return ec_bal_area_map.get(ec)

    def get_bal_area_set(m, ec):
        bal_area_inp = get_bal_area_inp(ec)
        if (bal_area_inp == BALAREA_MG_STR):
            return m.BalAreaMG
        if (bal_area_inp == BALAREA_SG1_STR):
            return m.BalAreaSG1
        if (bal_area_inp == BALAREA_SG2_STR):
            return m.BalAreaSG2
        if (bal_area_inp == BALAREA_SG3_STR):
            return m.BalAreaSG3

    def get_output_dec(ect):
        return ect_ec_dict.get(ect)

    def get_yr_ba(m, ec):
        bal_area_set = get_bal_area_set(m, ec)
        return m.Year * bal_area_set

    def init_ect_yr_ba(m):
        return ((ect, yr_ba)
                for ect in m.EnergyConvTech
                for yr_ba in get_yr_ba(m, get_output_dec(ect)))

    model.ECT_YR_BA = Set(within=model.EnergyConvTech * model.Year * model.BalArea,
                          initialize=init_ect_yr_ba, ordered=True)

    ######################
    #   Model Variables  #
    ######################

    model.EffectiveCapacityExistingInYear = Var(
        model.ECT_YR_BA, within=NonNegativeReals)
    model.AnnualCost = Var(model.Year, within=NonNegativeReals)
    model.TotalCost = Var(within=NonNegativeReals)

    ######################
    # Objective Function #
    ######################

    def ObjectiveFunction_rule(m):
        return sum(m.EffectiveCapacityExistingInYear["EG_COAL", 2025, ba] for ba in get_bal_area_set(m, "ELECTRICITY"))
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #####################
    # Constraints       #
    #####################

    #########                   Cost related                        #############

    def total_model_cost_rule(m):
        return (m.TotalCost == summation(m.AnnualCost))

    model.total_model_cost_constraint = Constraint(rule=total_model_cost_rule)

    # The model instance gets created after this call
    print("\nBefore create instance")
    instance = model.create_instance()
    print("\nAfter create instance")
    return instance


def test_process_range():
    start, end = 2010, 2015
    element = f"{start}-{end}"
    assert supply.process_range(element) == ",".join(str(i)
                                                     for i in range(2010, 2016))
    assert supply.process_range("text") == "text"

    element = f"{start} - {end}"
    assert supply.process_range(element) == element


def test_process_element():
    start, end = 2010, 2015
    element = f"{start}-{end}"
    assert supply.process_element(element) == list(range(start, end+1))
    assert supply.process_element("1") == [1]
    assert supply.process_element("1.5") == pytest.approx([1.5])
    assert supply.process_element("text") == ["text"]
    start, end = 2015, 2010
    element = f"{start}-{end}"
    assert supply.process_element(element) == [element]


def write_constraints_file(filepath, data):
    with open(filepath, "w") as f:
        f.write(data)


def test_parse_user_constraints(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    filepath = tmp_path / "UserConstraints.csv"

    no_bounds = """
EffectiveCapacityExistingInYear,RF_ATF,2025,INDIA,,1.3
EffectiveCapacityExistingInYear,RF_HSD,2025,INDIA,,1.0
EffectiveCapacityExistingInYear,RF_LPG,2025,INDIA,,1.0
EffectiveCapacityExistingInYear,RF_MS,2025,INDIA,,1.0
EffectiveCapacityExistingInYear,RF_OTHERPP,2025,INDIA,,1.0"""
    write_constraints_file(filepath, no_bounds)
    constraints = supply.parse_user_constraints(filepath)
    assert len(constraints) == 0
    with_bounds1 = """1.07,TotalModelCost
1.05,AnnualCost,2021
BOUNDS,1234,4321"""
    write_constraints_file(filepath, with_bounds1)
    constraints = supply.parse_user_constraints(filepath)
    assert len(constraints) == 1
    assert len(constraints[0]['VECTORS']) == 2
    with_bounds2 = """EffectiveCapacityExistingInYear,"EG_SOLARPV,EG_WIND",2025,INDIA,"ER,NER,NR,WR,SR",1.0
BOUNDS,500,None
AnnualCost,2021,1.05
AnnualCost,2022,1
BOUNDS,None,1000"""
    write_constraints_file(filepath, with_bounds2)
    constraints = supply.parse_user_constraints(filepath)
    assert len(constraints) == 2
    assert len(constraints[0]['VECTORS']) == 10
    assert len(constraints[1]['VECTORS']) == 2
    assert constraints[0]['BOUNDS'] == (500, None)
    assert constraints[1]['BOUNDS'] == (None, 1000)

    with_bounds_both = """TotalModelCost,1.07
AnnualCost,2021,1.23
BOUNDS,1234,4321"""
    write_constraints_file(filepath, with_bounds_both)
    constraints = supply.parse_user_constraints(filepath)
    assert constraints[0]['BOUNDS'] == (1234, 4321)

    no_vector = """AnnualCost,2021,1.05
BOUNDS,1234,4321
BOUNDS,1234,4321"""
    write_constraints_file(filepath, no_vector)
    constraints = supply.parse_user_constraints(filepath)
    assert len(constraints) == 2
    assert len(constraints[0]['VECTORS']) == 1
    assert len(constraints[1]['VECTORS']) == 0

    only_bounds = """BOUNDS,1234,4321
BOUNDS,1234,4321"""
    write_constraints_file(filepath, only_bounds)
    constraints = supply.parse_user_constraints(filepath)
    assert len(constraints) == 2
    assert len(constraints[0]['VECTORS']) == 0
    assert len(constraints[1]['VECTORS']) == 0

    empty = """"""
    write_constraints_file(filepath, empty)
    constraints = supply.parse_user_constraints(filepath)
    assert len(constraints) == 0

    empty_lines = """1.07,TotalModelCost
1.05,AnnualCost,2021


BOUNDS,1234,4321"""
    write_constraints_file(filepath, empty_lines)
    constraints = supply.parse_user_constraints(filepath)
    assert len(constraints) == 1
    assert len(constraints[0]['VECTORS']) == 2
    assert len(constraints[0]['VECTORS'][0]) == 2
    assert len(constraints[0]['VECTORS'][1]) == 3
    assert constraints[0]['BOUNDS'] == (1234, 4321)


def test_parse_user_constraints_with_model(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    filepath = tmp_path / "UserConstraints.csv"

    model = create_model()
    constraints = f"""{supply.COMMENT_KEYWORD}
EffectiveCapacityExistingInYear,EG_PHWR,ALL,INDIA,ER,1.2
{supply.CONSTRAINT_END_KEYWORD},500,None
AnnualCost,2021,2.0
{supply.CONSTRAINT_END_KEYWORD},500,600"""
    write_constraints_file(filepath, constraints)

    constraints = supply.parse_user_constraints(filepath, model)
    assert len(constraints) == 2
    assert len(constraints[0]['VECTORS']) == 5
    assert constraints[0]['BOUNDS'] == (500, None)
    assert len(constraints[1]['VECTORS']) == 1
    assert constraints[1]['BOUNDS'] == (500, 600)

    constraints = f"""{supply.COMMENT_KEYWORD}
EffectiveCapacityExistingInYear,EG_PHWR,ALL,INDIA,"ER,WR",1.2
{supply.CONSTRAINT_END_KEYWORD},500,None
AnnualCost,2021,2.0
{supply.CONSTRAINT_END_KEYWORD},500,600"""
    write_constraints_file(filepath, constraints)

    constraints = supply.parse_user_constraints(filepath, model)
    assert len(constraints) == 2
    assert len(constraints[0]['VECTORS']) == 10
    assert constraints[0]['BOUNDS'] == (500, None)
    assert len(constraints[1]['VECTORS']) == 1
    assert constraints[1]['BOUNDS'] == (500, 600)

    constraints = f"""{supply.COMMENT_KEYWORD}
EffectiveCapacityExistingInYear,EG_PHWR,2021,INDIA,ALL,1.2
{supply.CONSTRAINT_END_KEYWORD},500,None
AnnualCost,2021,2.0
{supply.CONSTRAINT_END_KEYWORD},500,600"""
    write_constraints_file(filepath, constraints)

    constraints = supply.parse_user_constraints(filepath, model)
    assert len(constraints) == 2
    assert len(constraints[0]['VECTORS']) == 5
    assert constraints[0]['BOUNDS'] == (500, None)
    assert len(constraints[1]['VECTORS']) == 1
    assert constraints[1]['BOUNDS'] == (500, 600)

    constraints = f"""{supply.COMMENT_KEYWORD}
EffectiveCapacityExistingInYear,EG_PHWR,2021-2022,INDIA,ALL,1.2
EffectiveCapacityExistingInYear,RF_OTHERPP,2021-2022,INDIA,,1.0
{supply.CONSTRAINT_END_KEYWORD},500,None
AnnualCost,2021,2.0
{supply.CONSTRAINT_END_KEYWORD},500,600"""
    write_constraints_file(filepath, constraints)

    constraints = supply.parse_user_constraints(filepath, model)
    assert len(constraints) == 2
    assert len(constraints[0]['VECTORS']) == 12
    assert constraints[0]['BOUNDS'] == (500, None)
    assert len(constraints[1]['VECTORS']) == 1
    assert constraints[1]['BOUNDS'] == (500, 600)

    constraints = f"""{supply.COMMENT_KEYWORD}
EffectiveCapacityExistingInYear,EG_PHWR,2045,INDIA,ALL,1.2
{supply.CONSTRAINT_END_KEYWORD},500,None
AnnualCost,2021,2.0
{supply.CONSTRAINT_END_KEYWORD},500,600"""
    write_constraints_file(filepath, constraints)

    with pytest.raises(loaders.LoaderError) as e:
        supply.parse_user_constraints(filepath, model)

    constraints = f"""{supply.COMMENT_KEYWORD}
EffectiveCapacityExistingInYear,EG_PHWR,INDIA,ALL,1.2
{supply.CONSTRAINT_END_KEYWORD},500,None
AnnualCost,2021,2.0
{supply.CONSTRAINT_END_KEYWORD},500,600"""
    write_constraints_file(filepath, constraints)

    with pytest.raises(loaders.LoaderError) as e:
        supply.parse_user_constraints(filepath, model)

    constraints = f"""{supply.COMMENT_KEYWORD}
EffectiveCapacityExistingInYear,EG_PHWR,2021,INDIA,ALL,1.2
{supply.CONSTRAINT_END_KEYWORD},500,None
AnnualCost,ALL,2.0
{supply.CONSTRAINT_END_KEYWORD},500,600"""
    write_constraints_file(filepath, constraints)

    constraints = supply.parse_user_constraints(filepath, model)
    assert constraints

    constraints = f"""{supply.COMMENT_KEYWORD}
EffectiveCapacityExistingInYear,ALL,ALL,ALL,ALL,1.2
{supply.CONSTRAINT_END_KEYWORD},500,None
AnnualCost,ALL,2.0
{supply.CONSTRAINT_END_KEYWORD},500,600"""
    write_constraints_file(filepath, constraints)

    constraints = supply.parse_user_constraints(filepath, model)
    assert constraints
    assert len(constraints) == 2
    assert len(constraints[0]['VECTORS']) == 250
    assert constraints[0]['BOUNDS'] == (500, None)
    assert len(constraints[1]['VECTORS']) == 5
    assert constraints[1]['BOUNDS'] == (500, 600)


def test_validate_variable(caplog):
    model = create_model()
    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], [2021], ["INDIA"], ['ER'], [1.3]]
    assert supply.validate_variable(tokens, model, 1)

    tokens = [['EffectiveCapacityExistingInYear1'],
              ['EG_PHWR'], [2021], ["INDIA"], ['ER'], [1.3]]
    with caplog.at_level("ERROR"):
        r = supply.validate_variable(tokens, model, 1)
    assert "line no. 1" in caplog.text
    assert "is not a valid model attribute" in caplog.text
    assert not r

    tokens = [[2.3],
              ['EG_PHWR'], [2021], ["INDIA"], ['ER'], [1.3]]
    with caplog.at_level("ERROR"):
        r = supply.validate_variable(tokens, model, 1)
    assert "line no. 1" in caplog.text
    assert "first item must be text" in caplog.text
    assert not r

    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], [2021], ["INDIA"], ['ER'], ["ere"]]
    with caplog.at_level("ERROR"):
        r = supply.validate_variable(tokens, model, 1)
    assert "line no. 1" in caplog.text
    assert "last item must be numeric value"
    assert not r

    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], [2021], ["INDIA"], [1.2]]
    with caplog.at_level("ERROR"):
        r = supply.validate_variable(tokens, model, 1)
    assert "line no. 1" in caplog.text
    assert "number of items provided for 'EffectiveCapacityExistingInYear' is incorrect" in caplog.text
    assert not r

    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], [2021], ["INDIA"], ['ER'], ['MT'], [1.2]]
    with caplog.at_level("ERROR"):
        r = supply.validate_variable(tokens, model, 1)
    assert "line no. 1" in caplog.text
    assert "number of items provided for 'EffectiveCapacityExistingInYear' is incorrect" in caplog.text
    assert not r

    tokens = [['ECT_YR_BA_domain'],
              ['EG_PHWR'], [2021], ["INDIA"], ['ER'], [1.2]]
    with caplog.at_level("ERROR"):
        r = supply.validate_variable(tokens, model, 1)
    assert "line no. 1" in caplog.text
    assert "is not a valid model output" in caplog.text
    assert not r


def test_validate_variable_ALL():
    model = create_model()
    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], ["ALL"], ["INDIA"], ['ER'], [1.3]]
    assert supply.validate_variable(tokens, model, 1)

    tokens = [['EffectiveCapacityExistingInYear1'],
              ['EG_PHWR'], ["ALL"], ["INDIA"], ['ER'], [1.3]]
    assert not supply.validate_variable(tokens, model, 1)

    tokens = [[2.3],
              ['EG_PHWR'], ["ALL"], ["INDIA"], ['ER'], [1.3]]
    assert not supply.validate_variable(tokens, model, 1)

    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], ["ALL"], ["INDIA"], ['ER'], ["ere"]]
    assert not supply.validate_variable(tokens, model, 1)

    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], ["ALL"], ["INDIA"], [1.2]]
    assert not supply.validate_variable(tokens, model, 1)

    tokens = [['ECT_YR_BA_domain'],
              ['EG_PHWR'], ["ALL"], ["INDIA"], ['ER'], [1.2]]
    assert not supply.validate_variable(tokens, model, 1)


def test_check_valid_index():
    model = create_model()
    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], [2021], ["INDIA"], ['ER'], [1.2]]
    variable_name = tokens[0][0]
    dataframe = pd.DataFrame(getattr(model, variable_name)._data.keys())
    dataframe = dataframe.rename(columns=dict(zip(
        dataframe.columns, string.ascii_uppercase + string.ascii_lowercase)))

    assert supply.check_valid_index(dataframe, tokens[1:-1])
    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], [2021], ["INDIA"], [1.2]]
    assert not supply.check_valid_index(dataframe, tokens[1:-1])

    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], [2021], ["INDIA"], ['ER1'], [1.2]]
    assert not supply.check_valid_index(dataframe, tokens[1:-1])


def test_check_valid_index_ALL():
    model = create_model()
    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], ["ALL"], ["INDIA"], ['ER'], [1.2]]
    variable_name = tokens[0][0]
    dataframe = pd.DataFrame(getattr(model, variable_name)._data.keys())
    dataframe = dataframe.rename(columns=dict(zip(
        dataframe.columns, string.ascii_uppercase + string.ascii_lowercase)))

    assert supply.check_valid_index(dataframe, tokens[1:-1])
    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], ["ALL"], ["INDIA"], [1.2]]
    assert not supply.check_valid_index(dataframe, tokens[1:-1])

    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], ["ALL"], ["INDIA"], ['ER1'], [1.2]]
    assert not supply.check_valid_index(dataframe, tokens[1:-1])


def test_expand_ALL():
    model = create_model()
    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], ["ALL"], ["INDIA"], ['ER'], [1.2]]

    assert supply.expand_ALL(tokens, model, 132)

    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], ["ALL"], ["INDIA"], ['ER1'], [1.2]]

    assert not supply.expand_ALL(tokens, model, 132)

    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], ["ALL"], ["INDIA"], [1.2]]

    assert not supply.expand_ALL(tokens, model, 132)


def test_float_with_spaces():
    from rumi.io import supply
    assert supply.float_("   3.14  ") == 3.14


def test_process_element_comma_list():
    from rumi.io import supply
    assert supply.process_element("1,2,3") == [1, 2, 3]


def test_expand_ALL_no_match(caplog):
    from rumi.io import supply
    model = create_model()
    tokens = [['EffectiveCapacityExistingInYear'],
              ['EG_PHWR'], ["2021"], ["INDIA"], ["ZZ"], [1.2]]
    with caplog.at_level("ERROR"):
        result = supply.expand_ALL(tokens, model, 10)
    assert result is None
    assert "line no. 10 has invalid fields" in caplog.text


def test_check_valid_index_all_tokens():
    from rumi.io import supply
    model = create_model()
    variable_name = "EffectiveCapacityExistingInYear"
    dataframe = pd.DataFrame(getattr(model, variable_name)._data.keys())
    dataframe = dataframe.rename(columns=dict(zip(
        dataframe.columns, string.ascii_uppercase + string.ascii_lowercase)))
    tokens = ["ALL"] * len(dataframe.columns)
    assert supply.check_valid_index(dataframe, tokens)


def test_make_filter_query_all_all():
    from rumi.io import supply
    tokens = [["ALL"], ["ALL"]]
    columns = ["col1", "col2"]
    q = supply.make_filter_query(tokens, columns)
    assert q == "col1 == col1"


def test_parse_user_constraints_last_without_bounds(tmp_path, caplog):
    from rumi.io import supply
    filepath = tmp_path / "UserConstraints.csv"
    data = """AnnualCost,2021,1.05"""
    filepath.write_text(data)
    with caplog.at_level("WARNING"):
        constraints = supply.parse_user_constraints(filepath)
    assert not constraints  # ignored
    assert "last constraint did not end with BOUNDS" in caplog.text


def test_process_range_start_greater_than_end(caplog):
    from rumi.io import supply
    element = "5-3"
    with caplog.at_level("ERROR"):
        result = supply.process_range(element)
    assert result == element
    assert "wrong values" in caplog.text


def test_float_():
    from rumi.io import supply
    assert supply.float_("1.23") == 1.23
    assert supply.float_("None") is None
    assert supply.float_(" none ") is None
    assert supply.float_("abc") == "abc"


def test_make_filter_query():
    from rumi.io import supply
    tokens = [[1, 2], ["A", "B"], ["X"]]
    columns = ["col1", "col2", "col3"]
    q = supply.make_filter_query(tokens, columns)
    # check expected structure
    assert "(col1 == 1 | col1 == 2)" in q
    assert "(col2 == 'A' | col2 == 'B')" in q
    assert "(col3 == 'X')" in q

    # when all tokens contain "ALL"
    tokens = [["ALL"], ["ALL"], ["ALL"]]
    q = supply.make_filter_query(tokens, columns)
    assert q == f"{columns[0]} == {columns[0]}"
