import os
import shutil

from src.params import get_run_params


def get_raw_data_dir():
    return './raw_data'


def get_words_dict_dir():
    return './data/word_dict'


def get_logs_dir():
    return f'./experiments/{get_run_params("EXPERIMENT_NAME")}/logs/log-{get_run_params("TASK_ID")}-{get_run_params("TIMESTAMP")}'


def get_corpora_dir():
    return f'./experiments/{get_run_params("EXPERIMENT_NAME")}/corpora'


def get_corpora_tokenized_dir():
    return f'./experiments/{get_run_params("EXPERIMENT_NAME")}/corpora_tokenized'


def get_tokenizers_dir():
    return f'./experiments/{get_run_params("EXPERIMENT_NAME")}/results/tokenizers'


def get_splinter_dir():
    return f'./experiments/{get_run_params("EXPERIMENT_NAME")}/results/splinter'


def get_trigram_models_dir():
    return f'./experiments/{get_run_params("EXPERIMENT_NAME")}/results/trigram_models'


def get_tokenizer_path(tokenizer_type, vocab_size) -> str:
    return f'{get_tokenizers_dir()}/{tokenizer_type}_{vocab_size}'


def get_corpus_path(corpus_name) -> str:
    return f'{get_corpora_dir()}/{corpus_name}.txt'


def get_tokenized_corpus_path(corpus_name, tokenizer_type, vocab_size) -> str:
    return f'{get_corpora_tokenized_dir()}/{corpus_name}_{tokenizer_type}_{vocab_size}.txt'


def create_experiment_dirs():
    os.makedirs(get_logs_dir(), exist_ok=True)
    os.makedirs(get_corpora_dir(), exist_ok=True)
    os.makedirs(get_corpora_tokenized_dir(), exist_ok=True)
    os.makedirs(get_tokenizers_dir(), exist_ok=True)
    os.makedirs(get_splinter_dir(), exist_ok=True)


def delete_corpus_file_if_exists(corpus_name):
    corpus_path = get_corpus_path(corpus_name)
    if os.path.exists(corpus_path):
        os.remove(corpus_path)
