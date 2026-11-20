from scripts.params import experiments, get_run_params, set_run_params
from scripts.logger import get_logger, initialize_logger

from splintering.utils.corpus_tools import CorpusTokenizer
from splintering.trainer import SplinterTrainer
from splintering.text_processors import TextProcessorBaseline, TextProcessorWithEncoding
from splintering.utils.languages import LanguageUtilsFactory
from splintering.utils.static_checks import run_static_checks, run_types_length_distribution
from scripts.train_tokenizer import train_tokenizer
from splintering.utils.path_utils import create_experiment_dirs, get_tokenizer_path, get_corpus_path, get_tokenized_corpus_path
from splintering.utils.misc import add_static_result_to_file, get_corpus_name


def run():
    language_utils = LanguageUtilsFactory.get_by_language(get_run_params("LANGUAGE"))
    train_dataset_path = get_run_params("SPLINTER_TRAINING_CORPUS_PATH")
    train_dataset_name = get_run_params("SPLINTER_TRAINING_CORPUS_NAME")
    letters_subset = get_run_params("SPLINTER_LETTERS_SUBSET")

    if get_run_params("SAVE_CORPORA_INTO_FILE"):
        # this is a splinter experiment
        if get_run_params("IS_ENCODED"):
            splinter_trainer = SplinterTrainer(language_utils)
            reductions_map, new_unicode_chars_map, _ = splinter_trainer.train(train_dataset_path, train_dataset_name, letters_subset)
            text_processor = TextProcessorWithEncoding(language_utils, reductions_map, new_unicode_chars_map)

        # this is a baseline experiment - no splinter
        else:
            text_processor = TextProcessorBaseline(language_utils)

        # save entire corpora as text files, encoded or not
        text_processor.save_corpus_as_text_file(train_dataset_path, train_dataset_name)
        language_utils.save_additional_corpora_for_evaluation(text_processor)

    # train tokenizers of different type (unigram / bpe), resulting in different vocab size
    if get_run_params("TRAIN_TOKENIZERS"):
        tokenizer_corpus_path = get_corpus_path(get_corpus_name(train_dataset_path, train_dataset_name))
        for tokenizer_type in get_run_params("TOKENIZERS_TYPES"):
            for vocab_size in get_run_params("TOKENIZERS_VOCAB_SIZES"):
                tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
                train_tokenizer(tokenizer_type=tokenizer_type, vocab_size=vocab_size, input_path=tokenizer_corpus_path, output_path=tokenizer_path)

    # for each trained tokenizer
    for tokenizer_type in get_run_params("TOKENIZERS_TYPES"):
        for vocab_size in get_run_params("TOKENIZERS_VOCAB_SIZES"):
            # tokenize corpora into files
            if get_run_params("TOKENIZE_CORPORA"):
                tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
                for corpus_name in get_run_params("STATIC_CHECKS_CORPORA"):
                    corpus_path = get_corpus_path(corpus_name)
                    tokenized_corpus_path = get_tokenized_corpus_path(corpus_name, tokenizer_type, vocab_size)
                    CorpusTokenizer().tokenize_corpus_into_file(tokenizer_path, corpus_path, tokenized_corpus_path)

            # run static checks on tokenized corpora
            if get_run_params("RUN_STATIC_CHECKS"):
                types_length_distribution = run_types_length_distribution(tokenizer_type, vocab_size)
                add_static_result_to_file(types_length_distribution)

                for corpus_name in get_run_params("STATIC_CHECKS_CORPORA"):
                    corpus_path = get_corpus_path(corpus_name)
                    tokenized_corpus_path = get_tokenized_corpus_path(corpus_name, tokenizer_type, vocab_size)
                    tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
                    results = run_static_checks(corpus_path, tokenized_corpus_path, tokenizer_path, vocab_size)
                    add_static_result_to_file(results)


if __name__ == '__main__':
    import os
    slurm_array_task_id = os.getenv('SLURM_ARRAY_TASK_ID')
    task_id = (int(slurm_array_task_id) - 1) if slurm_array_task_id else 0
    experiment = experiments[task_id]
    experiment["TASK_ID"] = task_id
    set_run_params(experiment)
    create_experiment_dirs()
    initialize_logger()
    get_logger().info(f'Experiment {task_id + 1} started')
    try:
        run()
    except Exception as e:
        get_logger().exception(f'Experiment {task_id + 1} failed with error: \n{e}')
        exit(1)
    get_logger().info(f'Experiment {task_id + 1} finished')
