import gc
from collections import Counter

import numpy as np
import scipy.sparse as sp
import sentencepiece as spm
import tokenization_scorer
from tqdm import tqdm

from src.logger import get_logger
from src.utils.path_utils import get_tokenizer_path


# ------------------------------------------------------------------------
# taken from https://github.com/MeLeLBGU/tokenizers_intrinsic_benchmark/
# ------------------------------------------------------------------------


def run_static_checks(corpus_path, tokenized_corpus_path, tokenizer_path, vocab_size):
    lines_number = get_lines_number(corpus_path)
    results = dict()
    results['corpus'] = tokenized_corpus_path
    results['fertility'] = fertility_score(corpus_path, tokenized_corpus_path, lines_number)
    results['renyi_efficiency'] = renyi_efficiency_score(tokenized_corpus_path)
    results['tokens_per_word_distribution'] = tokens_per_word_distribution(tokenizer_path, tokenized_corpus_path, lines_number)
    results['token_length_distribution'] = token_length_distribution(tokenizer_path, tokenized_corpus_path, lines_number)
    results['token_distinct_neighbors_threshold'] = token_distinct_neighbors_number(tokenized_corpus_path, lines_number, vocab_size)
    return results


def get_lines_number(corpus_path):
    get_logger().info('get_lines_number - START')
    with open(corpus_path, 'r', encoding='utf-8') as file:
        lines_number = sum(1 for _ in file)
    get_logger().info('get_lines_number - END')
    get_logger().info(f'Lines number: {lines_number}')
    return lines_number


def fertility_score(corpus_path, tokenized_corpus_path, lines_number):
    get_logger().info('fertility_score - START')

    with open(tokenized_corpus_path, 'r', encoding='utf-8') as file:
        num_of_tokens = sum([len(text.split(" ")) for text in tqdm(file, total=lines_number)])
    get_logger().info(f'num_of_tokens: {num_of_tokens}')

    with open(corpus_path, 'r', encoding='utf-8') as file:
        num_of_words = sum([len(sentence.split(" ")) for sentence in tqdm(file, total=lines_number)])
    get_logger().info(f'num_of_words: {num_of_words}')

    fertility = num_of_tokens / num_of_words
    get_logger().info('fertility_score - END')
    get_logger().info(f'Fertility score: {fertility}')
    return fertility


def renyi_efficiency_score(tokenized_corpus_path):
    get_logger().info('renyi_efficiency_score - START')
    with open(tokenized_corpus_path, 'r', encoding='utf-8') as file:
        file_generator = (line for line in file)
        score = tokenization_scorer.score(file_generator, power=2.5)
    get_logger().info('renyi_efficiency_score - END')
    get_logger().info(f'Renyi efficiency score: {score}')
    return score


def tokens_per_word_distribution(tokenizer_path, tokenized_corpus_path, lines_number):
    get_logger().info('tokens_per_word_distribution - START')
    tokenizer = spm.SentencePieceProcessor(model_file=f'{tokenizer_path}.model')

    distribution = dict()
    with open(tokenized_corpus_path, 'r', encoding='utf-8') as file:
        for text in tqdm(file, total=lines_number):
            if text == '\n':
                continue
            proto = tokenizer.decode([int(token_id) for token_id in text.split()], out_type='immutable_proto')
            tokenized_text = " ".join([piece.piece for piece in proto.pieces]) + " "
            text_split_by_words = tokenized_text.split("▁")
            tokens_per_word_in_text = [word.count(' ') for word in text_split_by_words]
            for tokens_per_word in tokens_per_word_in_text:
                if tokens_per_word not in distribution:
                    distribution[tokens_per_word] = 0
                distribution[tokens_per_word] += 1
    cleaned_distribution = merge_keys_above_ten(distribution)
    get_logger().info('tokens_per_word_distribution - END')
    get_logger().info(f'Tokens per word distribution: {cleaned_distribution}')
    return cleaned_distribution


def token_length_distribution(tokenizer_path, tokenized_corpus_path, lines_number):
    get_logger().info('token_length_distribution - START')
    tokenizer = spm.SentencePieceProcessor(model_file=f'{tokenizer_path}.model')
    token_occurrence = get_token_occurrence_in_corpus(tokenized_corpus_path, lines_number)
    distribution = dict()
    for (token_id, count) in token_occurrence.items():
        proto = tokenizer.decode([int(token_id)], out_type='immutable_proto')
        token = proto.pieces[0].piece
        token_length = get_token_length(token)
        if token_length not in distribution:
            distribution[token_length] = 0
        distribution[token_length] += count
    cleaned_distribution = merge_keys_above_ten(distribution)
    get_logger().info('token_length_distribution - END')
    get_logger().info(f'Token length distribution: {cleaned_distribution}')
    return cleaned_distribution


def get_token_length(token: str):
    return len(token.replace('▁', ''))


def get_token_occurrence_in_corpus(tokenized_corpus_path, lines_number):
    token_occurrence = dict()
    with open(tokenized_corpus_path, 'r', encoding='utf-8') as file:
        for text in tqdm(file, total=lines_number):
            token_occurrence_in_text = get_token_occurrence_in_text(text)
            for token, count in token_occurrence_in_text.items():
                if token not in token_occurrence:
                    token_occurrence[token] = 0
                token_occurrence[token] = token_occurrence[token] + count
    return token_occurrence


def get_token_occurrence_in_text(text: str):
    text_tokens = text.split()
    return Counter(text_tokens)


# sum all the 11+ keys together in a special "11+" key
def merge_keys_above_ten(distribution):
    keys_above_ten_counter = sum(v for (k, v) in distribution.items() if int(k) > 10)
    clean_distribution = {k: v for (k, v) in distribution.items() if int(k) <= 10}
    clean_distribution.pop(0, None)
    clean_distribution = dict(sorted(clean_distribution.items()))
    clean_distribution = {str(k): v for (k, v) in clean_distribution.items()}
    clean_distribution["11+"] = keys_above_ten_counter
    return clean_distribution


def run_types_length_distribution(tokenizer_type, vocab_size):
    get_logger().info('types_length_distribution - START')
    tokens = get_vocab_tokens(tokenizer_type, vocab_size)
    token_lengths = [get_token_length(token) for token in tokens]
    distribution = dict(sorted(Counter(token_lengths).items()))
    get_logger().info('types_length_distribution - END')
    get_logger().info(f'Types length distribution: {distribution}')

    return {
        'vocab': get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size),
        'types_length_distribution': distribution
    }


def get_vocab_tokens(tokenizer_type, vocab_size):
    vocab_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
    with open(f'{vocab_path}.vocab', 'r', encoding='utf-8') as file:
        tokens = [token.split('\t')[0] for token in tqdm(file)]
    return tokens


def token_distinct_neighbors_number(tokenized_corpus_path, lines_number, vocab_size:int, window_size=2, threshold=1):
    get_logger().info('distinct_neighbors - START')

    distinct_neighbors = sp.lil_matrix((vocab_size, vocab_size), dtype=int)
    with open(tokenized_corpus_path, 'r', encoding='utf-8') as file:
        for text in tqdm(file, total=lines_number):
            add_distinct_neighbors_from_text(text, distinct_neighbors, window_size)

    distinct_neighbors_bool = sp.lil_matrix((vocab_size, vocab_size), dtype=bool)
    distinct_neighbors_bool[distinct_neighbors >= threshold] = True

    del distinct_neighbors
    gc.collect()

    distinct_neighbors_bool = distinct_neighbors_bool.tocsr()
    distinct_neighbors_bool = (distinct_neighbors_bool + distinct_neighbors_bool.T).astype(bool).tocsr()

    neighbors_number = np.array(distinct_neighbors_bool.sum(axis=1)).flatten()
    neighbors_number = {str(i): int(neighbors_number[i]) for i in range(vocab_size) if neighbors_number[i] != 0}
    neighbors_number = dict(sorted(neighbors_number.items(), key=lambda item: item[1], reverse=True))

    get_logger().info('distinct_neighbors - END')
    return neighbors_number


def add_distinct_neighbors_from_text(text: str, distinct_neighbors, window_size: int):
    token_ids = [int(token_id) for token_id in text.split()]
    length = len(token_ids)
    for i, token_id in enumerate(token_ids[:-1]):
        neighbors = token_ids[(i + 1):min(length,(i + window_size + 1))]
        for neighbor in neighbors:
            if neighbor >= token_id:
                distinct_neighbors[token_id, neighbor] += 1
            else:
                distinct_neighbors[neighbor, token_id] += 1
    return distinct_neighbors
