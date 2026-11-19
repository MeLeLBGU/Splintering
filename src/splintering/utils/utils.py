import json
import re
from collections import Counter

from src.utils.path_utils import get_splinter_dir, get_logs_dir


def add_static_result_to_file(result):
    with open(f'{get_logs_dir()}/static_checks_results.json', 'a', encoding='utf-8') as file:
        file.write('\n')
        json.dump(result, file, indent='\t')


def get_reductions_map_from_file():
    with open(f'{get_splinter_dir()}/reductions_map.json', 'r') as file:
        data = json.load(file)
    data = {int(key): value for key, value in data.items()}
    return data


def get_new_unicode_chars_map_from_file():
    with open(f'{get_splinter_dir()}/new_unicode_chars.json', 'r') as file:
        data = json.load(file)
    return data


def get_new_unicode_chars_inverted_map_from_file():
    with open(f'{get_splinter_dir()}/new_unicode_chars_inverted.json', 'r') as file:
        data = json.load(file)
    return data


def get_permutation(word, position, word_length):
    if position < 0:
        permutation = word[:word_length + position] + word[(word_length + position + 1):]
    else:
        permutation = word[:position] + word[(position + 1):]
    return permutation


def get_words_dict_by_length(words):
    words_dict_by_length = {}
    max_length = len(max(words.keys(), key=len))

    # initialize the dict with keys from 2 to max_length
    for i in range(2, max_length + 1):
        words_dict_by_length[i] = dict()
    # fill the dict with words
    for word in words.keys():
        words_dict_by_length[len(word)][word] = words[word]
    return words_dict_by_length


def get_letters_frequency(words):
    words_letters_frequency = [dict(Counter(word)) for word, _ in words.items()]
    letters_frequency = dict()
    for word_letter_frequency in words_letters_frequency:
        for letter in word_letter_frequency.keys():
            letters_frequency[letter] = letters_frequency.get(letter, 0) + word_letter_frequency[letter]
    letters_frequency = dict(sorted(letters_frequency.items(), key=lambda item: item[1], reverse=True))
    return letters_frequency


def decode_tokens_vocab_file(tokens_vocab_file):
    with open(f'{tokens_vocab_file}.vocab', 'r', encoding='utf-8') as file:
        encoded_text = file.read()

    new_unicode_chars_inverted_map = get_new_unicode_chars_inverted_map_from_file()
    encoded_tokens = encoded_text.splitlines()
    decoded_tokens = list()
    for encoded_token in encoded_tokens:
        decoded_words_list = decode_sentence(encoded_token, new_unicode_chars_inverted_map)
        decoded_tokens.append("\t".join(decoded_words_list))
    with open(f'{tokens_vocab_file}_decoded.vocab', 'a', encoding='utf-8') as file:
        file.write("\n".join(decoded_tokens))


def decode_sentence(text, new_unicode_chars_inverted_map):
    decoded_words_list = list()
    encoded_words = text.split()
    for encoded_word in encoded_words:
        decoded_word = [new_unicode_chars_inverted_map.get(char, char) for char in encoded_word]
        decoded_words_list.append("".join(decoded_word))
    return decoded_words_list


def get_corpus_name(dataset_path, dataset_name):
    return re.sub(r'\W', '_', f'{dataset_path}_{dataset_name}')
