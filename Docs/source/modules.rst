Module Level API
================
Rumi is an open source platform for energy systems modelling, developed using Python.
This means that the platform code is available to a user for embedding inside
his or her own software. To use any of the provided functionality as a library,
one has to initialize the system by calling the following function at the start
of one's Python code as indicated below::

   from rumi.io import config
   config.initialize_config(model_instance_path, scenario)


After this you can use the desired functionality directly in your code.
Here are some user interface functions which may be useful.

The following function can be used to get the data corresponding to a specific parameter::

   from rumi.io import loaders 

   loaders.get_parameter("GDP") # this returns the data for the requested parameter from the model.


Computation of demand for a particular DemandSector, EnergyService, 
EnergyCarrier combination can be accomplished using the following function::
   
   from rumi.processing import demand

   demand.compute_demand("D_RES", # DemandSector
                         "RES_COOL", # EnergyService
                         "ELECTRICITY") # EnergyCarrier


For supply parameter access, the following library function can be used.
This function will return data for the requested supply parameter at a 
geographic and time granularity equal to the corresponding energy carrier's 
balancing area and balancing time::


  from rumi.io import supply
  supply.get_filtered_parameter("DEC_Taxation")


   
Following are some useful APIs and their description in brief.

Data Validation
---------------
`rumi.io.validate.rumi_validate`

.. automodule:: rumi.io.validate.rumi_validate


Demand Processing
-----------------
`rumi.processing.demand.rumi_demand`

.. automodule:: rumi.processing.demand.rumi_demand


`rumi.processing.demand.compute_demand`

.. automodule:: rumi.processing.demand.compute_demand


Supply Processing
-----------------
`rumi.processing.supplyaux.rumi_supply`

.. automodule:: rumi.processing.supplyaux.rumi_supply

		
Parameter Access
----------------
`rumi.io.loader.get_parameter`

.. automodule:: rumi.io.loaders.get_parameter


Supply Parameter Access
-----------------------
`rumi.io.supply.get_filtered_parameter`:

Supply parameters are passed to `Pyomo` always at the corresponding energy carrier's 
balancing area and balancing time granularity.
This function gets supply parameter data at a geographic and time granularity equal 
to the corresponding energy carrier's balancing area and balancing time.

.. automodule:: rumi.io.supply.get_filtered_parameter


Other
-----
`rumi.io.utilities.base_dataframe_all`:

This is a useful function to automatically generate time, geography and consumer 
column data by making use of common parameters.

.. automodule:: rumi.io.utilities.base_dataframe_all
