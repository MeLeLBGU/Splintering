import os
import pickle

import numpy as np
from nltk.lm.models import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.logger import get_logger, initialize_logger
from src.params import get_dummy_experiment, set_run_params
from src.utils.path_utils import get_trigram_models_dir, get_corpora_tokenized_dir, get_logs_dir


def test_perplexity(corpus_name, train_model: bool):
    tokenized_corpus_path = f'{get_corpora_tokenized_dir()}/{corpus_name}.txt'
    with open(tokenized_corpus_path, 'r', encoding='utf-8') as file:
        tokenized_text = file.read().split('\n')
    train_data, test_data = train_test_split(tokenized_text, test_size=0.1, random_state=42)

    n = 3  # For trigram model
    if train_model:
        train_data = [data.split(' ') for data in train_data]
        train_data, padded_vocab = padded_everygram_pipeline(n, train_data)
        model = KneserNeyInterpolated(order=n)
        get_logger().info(f"Start model fit for corpus '{corpus_name}'")
        model.fit(train_data, padded_vocab)
        with open(f'{get_trigram_models_dir()}/trigram_model_{corpus_name}.pkl', 'wb') as file:
            pickle.dump(model, file)
    else:
        get_logger().info(f"Start model load for corpus '{corpus_name}'")
        with open(f'{get_trigram_models_dir()}/trigram_model_{corpus_name}.pkl', 'rb') as file:
            model = pickle.load(file)

    test_data = [data.split(' ') for data in test_data]
    test_data, _ = padded_everygram_pipeline(n, test_data)
    test_data = [[i for i in item] for item in test_data]
    test_data = [[i for i in item if len(i) == n] for item in test_data]
    perplexity_list = list()
    get_logger().info(f"Start perplexity calculations for {len(test_data)} sentences in '{corpus_name}'.")
    inf_count = 0
    with open(f'{get_logs_dir()}/trigram_model_{corpus_name}_perplexity.txt', 'a', encoding='utf-8') as file:
        for sentence in tqdm(test_data):
            sentence_perplexity = model.perplexity(sentence)
            file.write(f'{sentence_perplexity}\n')
            if np.isinf(sentence_perplexity):
                inf_count += 1
            else:
                perplexity_list.append(sentence_perplexity)
    get_logger().info(f"Finished perplexity calculations for {len(test_data)} sentences in '{corpus_name}'.")
    get_logger().info(f'number of inf: {inf_count}')
    get_logger().info(f'Average perplexity: {sum(perplexity_list) / len(perplexity_list)}')
    return sum(perplexity_list) / len(perplexity_list)


if __name__ == '__main__':
    experiment = get_dummy_experiment('2025-01-01-trigram_lm_perplexity')
    set_run_params(experiment)
    os.makedirs(get_logs_dir(), exist_ok=True)
    os.makedirs(get_trigram_models_dir(), exist_ok=True)
    initialize_logger()
    perplexity = test_perplexity(corpus_name='hebrew_diacritized_pre_modern_unigram_10000', train_model=True)
