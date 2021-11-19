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
import os
import yaml
import pytest
from rumi.io import customyaml


@pytest.fixture()
def yamlfile():
    with open("test_load_yaml.yml", "w") as f:
        d = """a: 1
b: 2
c: !include test_load_yaml_include.yml"""
        f.write(d)

    with open("test_load_yaml_include.yml", "w") as f:
        d = """x: 1
y: y"""
        f.write(d)

    yield "test_load_yaml.yml"

    os.remove("test_load_yaml.yml")
    os.remove("test_load_yaml_include.yml")


def test_load_yaml(yamlfile):
    d = customyaml.load_yaml(yamlfile)
    assert 'x' in d['c']
    assert 'y' in d['c']
    assert d['c']['x'] == 1
    assert d['c']['y'] == 'y'
