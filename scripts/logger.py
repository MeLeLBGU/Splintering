from scripts.params import SplinterExperiment
import json
from dataclasses import asdict

from splintering.utils.logger import initialize_logger as _init, get_logger


def initialize_logger(experiment: SplinterExperiment):
    _init(experiment.PATHS)
    logger = get_logger()

    # log run params
    logger.info("********** RUN PARAMETERS **********")
    run_params_str = json.dumps(asdict(experiment), indent='\t')
    logger.info(f"\n{run_params_str}\n\n\n")
