import time
from rumi.io import loaders
from rumi.io import supply
from rumi.io import filemanager


def timeit(f, *args, **kwargs):
    start = time.time()
    r = f(*args, **kwargs)
    end = time.time()
    print(f"Time taken to execute {f.__qualname__}:", end-start)
    return r


def validate_param(param_name, validation_index=None, **kwargs):
    common_ = filemanager.common_specs()
    demand_ = filemanager.demand_specs()
    supply_ = filemanager.supply_specs()

    if param_name in common_:
        specs = dict(common_[param_name])
        module = "rumi.io.common"
        data = loaders.get_parameter(param_name,
                                     **kwargs)
    elif param_name in demand_:
        specs = dict(demand_[param_name])
        module = "rumi.io.demand"
        data = loaders.get_parameter(param_name,
                                     **kwargs)
    else:
        specs = dict(supply_[param_name])
        module = "rumi.io.supply"
        data = supply.get_filtered_parameter(param_name, **kwargs)

    if validation_index:
        v = specs['validation']
        specs['validation'] = [v[validation_index]]

    print(specs['validation'])
    return loaders.validate_param(param_name,
                                  specs,
                                  data,
                                  module,
                                  **kwargs)
