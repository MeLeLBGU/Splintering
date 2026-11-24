from scripts.params import get_dummy_experiment
from scripts.logger import get_logger, initialize_logger

from splintering.experiments.trigram_lm import test_perplexity


experiment = get_dummy_experiment('trigram_lm_perplexity')  # TODO: dummy experiment probably indicates bad code design.
initialize_logger(experiment)
perplexity = test_perplexity(experiment.PATHS, corpus_name='hebrew_diacritized_pre_modern_unigram_10000', train_model=True)
