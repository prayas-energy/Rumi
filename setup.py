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
from distutils.core import setup


def requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f]


setup(
    name="rumi",
    version="1.1.0",
    description="Rumi, An Open Source Energy Modelling Platform",
    long_description="README.md",
    long_description_content_type="text/markdown",
    url="https://www.prayaspune.org/peg/publications/item/512",
    maintainer="Prayas Energy Group",
    maintainer_email="energy.model@prayaspune.org",
    license="Apache License 2.0",
    classifiers=[
        "License :: Apache License 2.0 ",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=["rumi",
              "rumi.io",
              "rumi.processing"],
    include_package_data=True,
    package_data={'': ['Config/*.yml']},
    install_requires=requirements(),
    entry_points={"console_scripts": ["rumi_demand=rumi.processing.demand:main",
                                      "rumi_validate=rumi.io.validate:main",
                                      "rumi_postprocess=rumi.processing.postprocess:main",
                                      "rumi_supply=rumi.processing.supply:main"]},
)
