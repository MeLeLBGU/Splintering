import json
import os

import numpy as np
from datasets import load_dataset

from src.CorpusWordsExtractor import CorpusWordsExtractor
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.logger import get_logger
from src.utils.path_utils import get_logs_dir, get_raw_data_dir, get_splinter_dir, get_words_dict_dir
from src.utils.utils import get_words_dict_by_length, get_permutation, get_corpus_name


class SplinterTrainer:
    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils

    def train(self, dataset_path: str, dataset_name: str, letters_for_reductions: [str] = None):
        words_dict = self.get_word_dict(dataset_path, dataset_name)
        pre_process_words_dict = self.pre_process_words(words_dict)
        words_dict_by_length = get_words_dict_by_length(pre_process_words_dict)
        max_length = sorted(words_dict_by_length.keys(), reverse=True)[0]
    
        get_logger().info(f"Start first iteration of reductions:")
        reductions = self.initialize_reductions_dict(max_length)
        for word_length in range(4, max_length + 1):
            current_length_words = words_dict_by_length[word_length].keys()
            previous_length_words = words_dict_by_length[word_length - 1].keys()
            for word in current_length_words:
                for position in self.get_position_range_including_negative_indexes(word):
                    if letters_for_reductions is None or word[position] in letters_for_reductions:
                        permutation = get_permutation(word, position, word_length)
                        if permutation in previous_length_words:
                            reduction_key = f"{position}:{word[position]}"
                            self.increment_value(reductions[word_length], reduction_key)
    
            reductions[word_length] = self.normalize_values(self.sort_dictionary(reductions[word_length]))
            get_logger().info(f"Finished reductions for word length {word_length}")
    
        with open(f'{get_logs_dir()}/reductions_map_first_iteration.json', 'w') as file:
            json.dump(reductions, file, indent='\t')
    
        get_logger().info(f"Start updating reductions:")
        updated_reductions = self.initialize_reductions_dict(max_length)
        for word_length in range(4, max_length + 1):
            previous_length_words = words_dict_by_length[word_length - 1]
            current_length_words = words_dict_by_length[word_length]
            for word, score in list(current_length_words.items()):
                reduction_with_score = self.get_reduction(word, reductions, previous_length_words, 3)
                if reduction_with_score is not None:
                    self.increment_value(updated_reductions[word_length], reduction_with_score['reduction'])
                    current_length_words[word] = score * reduction_with_score['score']
    
            # sort by frequency
            updated_reductions[word_length] = self.normalize_values(self.sort_dictionary(updated_reductions[word_length]))
            # remove word lengths that doesn't have possible reductions
            if len(updated_reductions[word_length]) == 0:
                del updated_reductions[word_length]
            get_logger().info(f"Finished updating reductions for word length {word_length}")
    
        with open(f'{get_logs_dir()}/reductions_map_with_frequencies.json', 'w') as file:
            json.dump(updated_reductions, file, indent='\t')
    
        self.save_result_file("reductions_map", updated_reductions)
        reduction_to_new_chars_map = self.map_reductions_to_new_chars(updated_reductions)
        self.save_result_file("new_unicode_chars", reduction_to_new_chars_map)
        new_chars_to_reductions_map = {value: key for key, value in reduction_to_new_chars_map.items()}
        self.save_result_file("new_unicode_chars_inverted", new_chars_to_reductions_map)
        return updated_reductions, reduction_to_new_chars_map, new_chars_to_reductions_map
    
    def get_word_dict(self, dataset_path, dataset_name):
        corpus_name = get_corpus_name(dataset_path, dataset_name)
        if not os.path.exists(f'{get_words_dict_dir()}/{corpus_name}.json'):
            get_logger().info(f'word dict file was not found - creating it from corpus')
            corpus = load_dataset(dataset_path, dataset_name, split="train", cache_dir=get_raw_data_dir())
            corpus_word_extractor = CorpusWordsExtractor(self.language_utils)
            corpus_word_extractor.convert_corpus_to_words_dict_file(corpus, corpus_name)
    
        with open(f'{get_words_dict_dir()}/{corpus_name}.json', 'r') as file:
            words_dict = json.load(file)
        return words_dict

    def pre_process_words(self, word_counters):
        # Remove words that appeared less than 10 times in the entire corpus
        words = {k: int(v) for k, v in word_counters.items() if int(v) >= 10}

        words = {self.language_utils.replace_final_letters(k): v for k, v in words.items()}

        # Remove empty words and single characters words
        words = {k: int(v) for k, v in words.items() if len(k) > 1}

        # remove words containing characters from other languages.
        words = {k: v for k, v in words.items() if not self.language_utils.is_word_contains_letters_from_other_languages(k)}

        # normalize values
        max_counter = max(words.values())
        words = {k: (v / max_counter) for k, v in words.items()}
        return words

    def initialize_reductions_dict(self, max_length):
        # adding single chars to the map, so "root" letters will also be converted when encoding the text
        single_chars = self.language_utils.get_language_alphabet()
        reductions = {1: {single_char: 1 for single_char in single_chars}}
    
        for i in range(4, max_length + 1):
            reductions[i] = {}
        return reductions
    
    def increment_value(self, dictionary, key):
        if key in dictionary.keys():
            dictionary[key] += 1
        else:
            dictionary[key] = 1
    
    def map_reductions_to_new_chars(self, reductions):
        reductions_set = set()
        for word_length, reductions in reductions.items():
            reductions_set.update(reductions.keys())
        reduction_to_new_chars_map = dict()
        unicode_pua_start = 0x5000
        for i, reduction in enumerate(sorted(reductions_set)):
            new_char = chr(unicode_pua_start + i)
            reduction_to_new_chars_map[reduction] = new_char
        return reduction_to_new_chars_map
    
    def sort_dictionary(self, dictionary):
        return dict(sorted(dictionary.items(), key=lambda item: (-item[1], item[0])))
    
    # change to percents
    def normalize_values(self, dictionary):
        values = np.array(list(dictionary.values()))
        percent_dictionary = {key: value for key, value in zip(dictionary.keys(), (values / np.sum(values)))}
        return percent_dictionary
    
    def get_position_range_including_negative_indexes(self, word):
        half_length = len(word) // 2
        return range(-half_length, half_length + len(word) % 2)
    
    def get_reduction(self, word, reductions, previous_length_words, max_number_of_candidates):
        word_length = len(word)
        possible_reductions = list()
        if word_length > 3:
            for reduction, reduction_score in reductions[word_length].items():
                position, letter = reduction.split(':')
                position = int(position)
                if word[position] == letter:
                    permutation = get_permutation(word, position, word_length)
                    if permutation in previous_length_words:
                        permutation_score = previous_length_words[permutation]
                        possible_reductions.append({"reduction": reduction, "score": reduction_score * permutation_score})
                        if len(possible_reductions) >= max_number_of_candidates:
                            break
    
        max_score_reduction = None
        if len(possible_reductions) > 0:
            max_score_reduction = max(possible_reductions, key=lambda x: x["score"])
        return max_score_reduction
    
    def save_result_file(self, file_name, data):
        if not os.path.exists(get_splinter_dir()):
            os.makedirs(get_splinter_dir())
    
        with open(f'{get_splinter_dir()}/{file_name}.json', 'w') as file:
            json.dump(data, file, indent='\t')
    
        # keep a copy of results in the log dir
        with open(f'{get_logs_dir()}/{file_name}.json', 'w') as file:
            json.dump(data, file, indent='\t')
