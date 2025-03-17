import json
import logging

from src.params import get_all_run_params
from src.utils.path_utils import get_logs_dir

logger = None

def initialize_logger():
    global logger
    logging.basicConfig(
        filename=f'{get_logs_dir()}/log.log',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # log run params
    logger.info("********** RUN PARAMETERS **********")
    run_params_str = json.dumps(get_all_run_params(), indent='\t')
    logger.info(f"\n{run_params_str}\n\n\n")

def get_logger():
    global logger
    return logger
