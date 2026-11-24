import logging

from splintering.utils.path_utils import SplinterPaths

logger = None

def initialize_logger(home: SplinterPaths):
    global logger
    logging.basicConfig(
        filename=home.get_logs_dir() / "log.log",
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
