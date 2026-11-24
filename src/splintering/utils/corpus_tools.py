from abc import ABC, abstractmethod

import json
import os
import re
from tqdm import tqdm

import sentencepiece as spm
from datasets import load_dataset

from splintering.utils.misc import get_corpus_name
from splintering.utils.paths import SplinterPaths
from splintering.utils.logger import get_logger
from splintering.utils.languages import LanguageUtilsInterface


class CorpusTokenizer:

    @staticmethod
    def tokenize_corpus_into_file(tokenizer_path, corpus_path, output_path):
        if os.path.exists(output_path):
            os.remove(output_path)

        get_logger().info(f'start tokenizing corpus {output_path}')
        sp_tokenizer = spm.SentencePieceProcessor(model_file=f'{tokenizer_path}.model')
        with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
            with open(output_path, 'a', encoding='utf-8') as output_file:
                for article_text in tqdm(corpus_file):
                    if article_text is not None:
                        encoded_article = " ".join([str(token_id) for token_id in sp_tokenizer.encode(article_text)])
                        output_file.write(encoded_article + '\n')
        get_logger().info(f'finished tokenizing corpus {output_path}')


class CorpusWordsExtractor:

    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils

    def convert_corpus_to_words_dict_file(self, home: SplinterPaths, corpus, output_filename):
        words = self.get_words_from_corpus(corpus["text"])
        with open(home.get_words_dict_dir() / f"{output_filename}.json", 'w') as file:
            json.dump(words, file, indent='\t')

    def get_words_from_corpus(self, articles_text):
        words = {}
        for index, article_text in enumerate(articles_text):
            article_text = self.language_utils.remove_diacritics(article_text)
            article_words = re.split(r'\.|\s|\n|-|,|:|"|\(|\)', article_text)

            for word in article_words:
                if word not in words:
                    words[word] = 0
                words[word] += 1

            if (index + 1) % 10000 == 0:
                get_logger().info(f'Finished extracting words from {index + 1} articles out of {len(articles_text)}')
        get_logger().info(f'Finished extracting words from {len(articles_text)} articles out of {len(articles_text)}')
        return words


class TextProcessorInterface(ABC):

    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils

    @abstractmethod
    def process(self, text: str) -> str:
        pass

    def save_corpus_as_text_file(self, home: SplinterPaths, dataset_path: str, dataset_name: str, batch_size=10000):
        corpus_name = get_corpus_name(dataset_path, dataset_name)
        get_logger().info(f'Start saving {corpus_name} corpus as text file with {self.__class__.__name__}')
        dataset = load_dataset(dataset_path, dataset_name, split="train", cache_dir=home.get_raw_data_dir().as_posix())
        home.delete_corpus_file_if_exists(corpus_name)
        for i, j in enumerate(tqdm(range(0, len(dataset), batch_size))):
            dataset_batch = dataset[j:j + batch_size]
            self.process_batch(home, dataset_batch, corpus_name)
        get_logger().info(f'Finished saving {corpus_name} corpus')

    def save_hedc4_corpus(self, home: SplinterPaths, batch_size: int=10_000):
        corpus_name = 'HeNLP_HeDC4'
        get_logger().info(f'Start saving {corpus_name} corpus as text file with {self.__class__.__name__}')
        corpus_path = home.get_raw_data_dir() / "HeNLP-he_dc4-10percent" / "HeDC4-10percent-seed42-sample.csv"
        if not os.path.exists(corpus_path):
            get_logger().info(f'HeDC4-10percent file was not found - creating it from full corpus')
            full_corpus = load_dataset('HeNLP/HeDC4', split="train", cache_dir=home.get_raw_data_dir().as_posix())
            sample_lines_from_corpus_to_csv(full_corpus, corpus_path)

        dataset = load_dataset("csv", data_files=corpus_path.as_posix())
        home.delete_corpus_file_if_exists(corpus_name)
        for i, j in enumerate(tqdm(range(0, len(dataset['train']), batch_size))):
            dataset_batch = dataset['train'][j:j + batch_size]
            self.process_batch(home, dataset_batch, corpus_name)
        get_logger().info(f'Finished saving {corpus_name} corpus')

    # hebrew_diacritized repository needs to be git cloned as a prerequisite from
    # https://github.com/elazarg/hebrew_diacritized.git
    # into Splinter/raw_data/hebrew_diacritized folder
    def save_pre_modern_corpus(self, home: SplinterPaths):
        corpus_name = 'hebrew_diacritized_pre_modern'
        get_logger().info(f'Start saving {corpus_name} corpus as text file with {self.__class__.__name__}')
        path = home.get_raw_data_dir() / "hebrew_diacritized" / "pre_modern"
        files = get_all_files_in_dir(path)
        dataset_batch = {'text': list()}
        for filename in files:
            with open(f'{filename}', 'r', encoding='utf-8') as file:
                dataset_batch['text'].append("\n".join(file.readlines()))
        home.delete_corpus_file_if_exists(corpus_name)
        self.process_batch(home, dataset_batch, corpus_name)
        get_logger().info(f'Finished saving {corpus_name} corpus')

    def process_batch(self, home: SplinterPaths, dataset_batch, corpus_name):
        encoded_articles_list = list()
        for article_text in dataset_batch["text"]:
            encoded_article = self.process(article_text)
            encoded_articles_list.append(encoded_article)
        with open(home.get_corpus_path(corpus_name), 'a', encoding='utf-8') as file:
            file.write("\n".join(encoded_articles_list))


def get_all_files_in_dir(directory):
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files]


def sample_lines_from_corpus_to_csv(full_corpus, output_file):
    get_logger().info(f'Start sample lines from corpus')
    random_seed = 42
    sample_size = int(0.1 * len(full_corpus))
    sampled_dataset = full_corpus.shuffle(seed=random_seed).select(range(sample_size))
    sampled_dataset.to_csv(output_file, index=False)
    get_logger().info(f'Finish sample lines from corpus')
