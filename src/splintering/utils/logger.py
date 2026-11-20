import logging

from splintering.utils.path_utils import get_logs_dir

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


def get_logger():
    global logger
    return logger
