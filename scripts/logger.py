import json

from scripts.params import get_all_run_params
from splintering.utils.logger import initialize_logger as _init, get_logger


def initialize_logger():
    _init()
    logger = get_logger()

    # log run params
    logger.info("********** RUN PARAMETERS **********")
    run_params_str = json.dumps(get_all_run_params(), indent='\t')
    logger.info(f"\n{run_params_str}\n\n\n")
