from scripts.params import experiments, set_run_params, get_run_params
from scripts.logger import get_logger, initialize_logger

from splintering.utils.path_utils import create_experiment_dirs, get_tokenizer_path
from splintering.utils.misc import add_static_result_to_file
from splintering.experiments.cognitive_evaluation import tokenize_cognitive_evaluation_file, run_cognitive_evaluation

task_id = 0
experiment = experiments[task_id]
experiment["TASK_ID"] = task_id
set_run_params(experiment)
create_experiment_dirs()
initialize_logger()
get_logger().info(f'Experiment started')
try:
    for tokenizer_type in get_run_params("TOKENIZERS_TYPES"):
        for vocab_size in get_run_params("TOKENIZERS_VOCAB_SIZES"):
            tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
            tokenize_cognitive_evaluation_file(tokenizer_path, tokenizer_type, vocab_size)
            cognitive_evaluation = run_cognitive_evaluation(tokenizer_type, vocab_size)
            add_static_result_to_file(cognitive_evaluation)
except Exception as e:
    get_logger().exception(f'Experiment failed with error: \n{e}')
    exit(1)
get_logger().info(f'Experiment finished')