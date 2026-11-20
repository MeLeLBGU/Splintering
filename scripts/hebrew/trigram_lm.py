from scripts.params import get_dummy_experiment, set_run_params
from scripts.logger import get_logger, initialize_logger

from splintering.utils.path_utils import get_trigram_models_dir, get_corpora_tokenized_dir, get_logs_dir
from splintering.experiments.trigram_lm import test_perplexity


experiment = get_dummy_experiment('2025-01-01-trigram_lm_perplexity')
set_run_params(experiment)

import os
os.makedirs(get_logs_dir(), exist_ok=True)
os.makedirs(get_trigram_models_dir(), exist_ok=True)
initialize_logger()
perplexity = test_perplexity(corpus_name='hebrew_diacritized_pre_modern_unigram_10000', train_model=True)
