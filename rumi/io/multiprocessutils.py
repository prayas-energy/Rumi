import logging
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import platform
from rumi.io import config
from rumi.io import logger as rumilogger

logger = logging.getLogger(__name__)


def get_numthreads():
    n = config.get_config_value("numthreads")
    if n:
        return int(n)
    else:
        return 1


def init_pool_loggers(queue):
    nprocess = get_numthreads()
    if nprocess and nprocess > 1 and platform.system() == "Windows":
        logger.debug("Configuring pool for logging")
        rumilogger.worker_configurer(queue)


def execute_in_process_pool(f, args):
    nprocess = get_numthreads()
    if nprocess and nprocess > 1:
        queue = rumilogger.get_queue()
        with Pool(processes=nprocess) as pool:
            logger.debug(
                f"Starting process pool of size {nprocess} for {f.__qualname__}")
            pool.starmap(init_pool_loggers, [(queue,)]*nprocess)
            return pool.starmap(f, args)
    else:
        return [f(*a) for a in args]


def execute_in_thread_pool(f, args):
    nthreads = get_numthreads()
    if nthreads and nthreads > 1:
        with ThreadPoolExecutor(nthreads) as pool:
            logger.debug(
                f"Starting thread pool of size {nthreads} for {f.__qualname__}")
            return pool.map(f, args)
    else:
        return [f(a) for a in args]
