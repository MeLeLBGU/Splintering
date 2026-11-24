from scripts.experiments import create_all_experiments
from scripts.logger import get_logger, initialize_logger

from splintering.utils.misc import add_static_result_to_file
from splintering.experiments.cognitive_evaluation import tokenize_cognitive_evaluation_file, run_cognitive_evaluation
from splintering.utils.path_utils import SplinterPaths

experiment = create_all_experiments(SplinterPaths("cognitive_eval"))[0]
experiment.PATHS.create_experiment_dirs()
initialize_logger(experiment)
get_logger().info(f'Experiment started')
try:
    for tokenizer_type in experiment.TOKENIZERS_TYPES:
        for vocab_size in experiment.TOKENIZERS_VOCAB_SIZES:
            tokenizer_path = experiment.PATHS.get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
            tokenize_cognitive_evaluation_file(experiment.PATHS, tokenizer_path, tokenizer_type, vocab_size)
            cognitive_evaluation = run_cognitive_evaluation(experiment.PATHS, tokenizer_type, vocab_size)
            add_static_result_to_file(experiment.PATHS, cognitive_evaluation)
except Exception as e:
    get_logger().exception(f'Experiment failed with error: \n{e}')
    exit(1)
get_logger().info(f'Experiment finished')