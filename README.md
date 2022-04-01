# Rumi - An open-source energy systems modelling platform
<a href="#rumi---an-open-source-energy-systems-modelling-platform"><img src="./Docs/graphics/Rumi-Logo-75dpi.png" width="250"></a>

Rumi is a generic, open-source energy systems modelling platform developed by
Prayas (Energy Group) to aid policy-relevant analysis. Its design enables
spatially and temporally disaggregated modelling, and modelling of energy demand
in detail.

This documentation refers to the Rumi platform. The platform consists of two
components, one for energy demand estimation and the other for optimisation of
supply sources, given suitable inputs. These two components can be run
independently, as explained later in this document.

The input data used to run the demand and supply modules of Rumi is not included
in the platform, and hence needs to be provided when running the modules.

The Rumi platform is licensed under the Apache License Version 2.0. For the
complete license text, refer to the [`LICENSE`](/LICENSE) file in the root directory of the
Rumi platform repository.

Please contact Prayas (Energy Group) at energy.model@prayaspune.org for any
queries regarding Rumi.

This short guide is a walk through of the commands to access Rumi functionality.

### Downloading Rumi 

Users need to download the latest Rumi source repository from release [tag](https://github.com/prayas-energy/rumi/releases/latest), or git clone the repository at a desired location. If you downloaded it from the release tag, please unzip to the desired location. 

### Creating a Virtual environment

1. Python version greater than or equal to 3.7 is needed.
2. Use of a virtual environment is recommended so that the changes made for
   installing Rumi do not affect other programs.

   The Anaconda distribution of Python provides simple tools to create virtual
   environments. Specifically, the following commands can be used to create
   and activate a virtual environment:
   ```
    conda create -n VENVNAME
    conda activate VENVNAME
   ```
   where `VENVNAME` is the name of the virtual environment.
   For more info, refer to the Anaconda documentation.

   If not using Anaconda, virtual environments can be created using virtualenv.

   a. Install virtualenv using:

   ```
       python -m pip install virtualenv
   ```

   b. Create a virtual environment with following command:

   ```
       python -m venv VIRTUALENVPATH
   ```
      where `VIRTUALENVPATH` is a location where the files specific to the virtual
      environment will be stored. Refer documentation of `virtualenv` for details.

   c. Activate the environment by excecuting following command on command prompt:

   ```
       source VIRTUALENVPATH/bin/activate
   ```

      on Linux, MacOS and Unix derivatives. On Windows, use the following command:

   ```
       VIRTUALENVPATH\Scripts\activate.bat
   ```

### Installing Rumi

1. In the virtual environment (created as per above instructions), traverse to
   the folder where the Rumi source repository is located. Run the following
   command in the repository base folder. This will install the rumi package and
   all dependencies in the virtual environment:

   ```
    pip install -e .
   ```
      Note:  If you encounter "pip" errors while running it through Anaconda, please try the following: 

      a)  `conda deactivate`

      b)  `conda create -n VENVNAME python=x.y.z`

      where `x.y.z` is the latest Python version (ex: `3.7.4`)
      
      c)  `conda activate VENVNAME`



2. To test the installation, run the following command in the repository
   base directory:

   ```
    pytest
   ```

   If all tests pass successfully, the installation is ready to use!

3. Running the supply module involves solving a linear program using a solver
   supported by Pyomo, such as CBC, CPLEX or Gurobi. Hence, a solver needs to be
   installed, and the solver name (`solver_name`) and, if needed, the path to the
   solver executable (`solver_executable`) need to be mentioned in the
   [`rumi/Config/Config.yml`](/rumi/Config/Config.yml) file.

4. For more details regarding configuring the Rumi installation, refer to the
   rumi-overview.pdf document in the [`Docs`](/Docs) folder.

### Checking data validity

A model developed for the Rumi platform (also referred to as a Rumi model instance)
consists of the input data as per Rumi specifications, which are detailed in the
documents in the [`Docs`](/Docs) folder. This instance data can be validated without
actually running the model using the following command in the virtual
environment where Rumi is installed:

```
 rumi_validate -m INSTANCEPATH -s SCENARIONAME -p PARAM 

```

where `INSTANCEPATH` is the folder where the Rumi instance data is located,
      `SCENARIONAME` is the name of the scenario to be validated
	  `PARAM` is the keyword Common, Demand or Supply (case sensitive)
	  

Following are the different arguments accepted by the validation module:

```
 rumi_validate --help
 Usage: rumi_validate [OPTIONS]
 
   Command line interface for data validation.
 
  -m/--model_instance_path, -s/--scenario and -p/--param_type are compulsory
   named arguments. While others are optional.
 
 Options:
   -p, --param_type TEXT           Parameter type to validate. Can be one of
                                   Common, Demand or Supply. 
   -m, --model_instance_path TEXT  Path where the model instance is located
   -s, --scenario TEXT             Name of the scenario
   -l, --logger_level TEXT         Level for logging: one of INFO, WARN, DEBUG or
                                   ERROR (default: INFO)
   --help                          Show this message and exit
```

For example, run the following command to test validity of Demand parameters
for `Scenario1`:

```
rumi_validate -p Demand -m "../PIER/" -s "Scenario1"
```

### Demand Estimation

To estimate demand based on the inputs provided in a Rumi instance, the
`rumi_demand` command needs to be run in the environment in which Rumi is
installed. This command takes two mandatory inputs for (a) path to the model
instance, and (b) the name of the scenario. The rest of the inputs are optional.

Following help message lists the entire set of arguments:

```
 rumi_demand --help
 Usage: rumi_demand [OPTIONS]
 
   Command line interface for processing demand inputs. 
   
   -m/--model_instance_path and -s/--scenario are mandatory arguments, while the
   others are optional.
   
    if demand_sector, energy_service, energy_carrier options are provided, demand
    will be computed only for given demand_sector, energy_service, energy_carrier
    combinations. If these parameters are not given (none of them),
    then demand will be processed for all demand_sector, energy_service and
    energy_carrier combinations.

 
 Options:
   -m, --model_instance_path TEXT  Path of the model instance root folder
   -s, --scenario TEXT             Name of the scenario within specified model
   -o, --output TEXT               Path of the output folder
   -D, --demand_sector TEXT        Name of demand sector
   -E, --energy_service TEXT       Name of energy service
   -C, --energy_carrier TEXT       Name of energy carrier
   -l, --logger_level TEXT         Level for logging,one of
                                   INFO,WARN,DEBUG,ERROR (default: INFO)
   -t, --numthreads INTEGER        Number of threads/processes (default: 2)
   --validation / --no-validation  Enable/disable validation (default: Enabled)
   --help                          Show this message and exit
```

To run the demand module for all provided demand_sector, energy_service and
energy_carrier combinations, the following command can be run:

```
 rumi_demand -m <INSTANCEPATH> -s <SCENARIONAME>
 e.g., rumi_demand -m "../PIER" -s "Scenario1"
```

where `INSTANCEPATH` is the path to the model instance (`../PIER`), and
      `SCENARIONAME` is the name of the scenario to be run (`Scenario1`)

By default, the output of the demand module is written to the
`INSTANCEPATH/Scenarios/SCENARIONAME/Demand/Output` folder. The output folder
can be changed using the -o option, as follows:

```
 rumi_demand -m "../PIER" -s "Scenario1" -o "../PIER/Output"
```

Note that, depending on the input data, validation can take a long time, even
longer than the actual processing. Validation is enabled by default, and can be
suppressed with the `--no-validation` option.

### Supply Processing

In the supply module, energy supply sources are optimised based on the demand
to be met, and the various supply-side inputs provided. The command to run the
supply module is `rumi_supply`. The following help message lists the arguments
for the `rumi_supply` command, which is to be run in the environment in which
Rumi is installed:

```
 rumi_supply --help
 usage: rumi_supply [-h] [-o OUTPUT_FOLDER] -m
                    MODEL_INSTANCE_PATH -s SCENARIO
 
 Supply processing for the given model
 
 mandatory arguments:
   -m, --model_instance_path TEXT   Path of the model instance top-level folder
   -s, --scenario TEXT              Name of the scenario within specified model
 
 optional arguments:
   -h, --help                       Show this help message and exit
   -o, --output_folder TEXT         Path of the output folder
```

For example, the supply module for `Scenario1` of the PIER model instance located
one level up can be run using the following command:

```
 rumi_supply -m "../PIER" -s "Scenario1"
```

By default, the output of the supply module is written to the
`INSTANCEPATH/Scenarios/SCENARIONAME/Supply/Output` folder.


### Post Processing

In addition to the above commands to run the demand and supply components of Rumi, there is an additional command which processes the outputs produced by the demand and supply components to produce some useful aggregate results. Currently, the only aggregation supported is to compute the various emissions that are specified as part of the input. In future, other types of post-processing may also be added.

The following help message lists the arguments for the `rumi_postprocess` command, which is to be run in the environment in which Rumi is installed:
```
 rumi_postprocess --help
 Usage: rumi_postprocess [OPTIONS]

  Post processing script. Supports computation of ECT emission  and EndUse
  emission.

Options:
  -m, --model_instance_path PATH  Path of the model instance root folder
  -s, --scenario TEXT             Name of the scenario within specified
                                  model
  -o, --output TEXT               Path of the output folder
  -D, --demand_output PATH        Path of Demand processing output folder
  -S, --supply_output PATH        Path of Supply processing output folder
  -l, --logger_level TEXT         Level for logging: one of INFO, WARN,
                                  DEBUG or ERROR. (default: INFO)
  --help                          Show this message and exit.
```

For example, the postprocess module for `Scenario1` of the PIER model instance located one level up can be run using the following command:

```
 rumi_postprocess -m "../PIER" -s "Scenario1"
```

By default, the output of the postprocess module is written to the
`INSTANCEPATH/Scenarios/SCENARIONAME/PostProcess/Output` folder.



## PIER 

PIER, a fully functional model of the Indian energy system built on Rumi, is available for download and use at https://github.com/prayas-energy/PIER 
