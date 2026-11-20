import os

from splintering.utils.languages import LanguageUtilsInterface
from splintering.utils.misc import get_new_unicode_chars_map_from_file


def compare_vocabs(baseline_vocab: list[str], splinter_decoded_vocab: list[str]):
    baseline_vocab_set = set(baseline_vocab)
    decoded_vocab = [line.split("\t\t") for line in splinter_decoded_vocab]
    possible_full_vocab = [line[0] for line in decoded_vocab if len(line) == 3 and line[0] != line[1]]  # BERT differ

    intersection = {token for token in possible_full_vocab if token in baseline_vocab_set}
    non_complete_words = len(decoded_vocab) - len(possible_full_vocab)
    splinter_unique = len(possible_full_vocab) - len(intersection)
    return [
        f"Full vocab: \t\t\t{len(decoded_vocab)} \t(100.00%)",
        f"Non-complete words: \t{non_complete_words}  \t({(non_complete_words / len(decoded_vocab) * 100):.2f}%)",
        f"Intersection: \t\t\t{len(intersection)}  \t({(len(intersection) / len(decoded_vocab) * 100):.2f}%)",
        f"Splinter unique: \t\t{splinter_unique} \t({(splinter_unique / len(decoded_vocab) * 100):.2f}%)",
    ]


def decode_splintered_vocab(encoded_text, language_utils: LanguageUtilsInterface):
    new_unicode_chars_map = get_new_unicode_chars_map_from_file()
    new_unicode_chars_inverted_map = {value: key for key, value in new_unicode_chars_map.items()}
    decoded_tokens = list()
    for encoded_token in encoded_text:
        encoded_token = encoded_token.split('\t')[0]
        decoded_token = [new_unicode_chars_inverted_map.get(char, char) for char in encoded_token]
        rebuilt_token = rebuild_reduced_word(decoded_token, language_utils)
        decoded_tokens.append(rebuilt_token + "\t\t" + decoded_token_to_str(decoded_token) + "\t\t" + encoded_token)
    return decoded_tokens


def rebuild_reduced_word(decoded_token, language_utils: LanguageUtilsInterface):
    decoded_token_concat = "".join(decoded_token)

    if len(decoded_token) == 1 and len(decoded_token[0]) > 1:
        return decoded_token_to_str(decoded_token)

    if decoded_token_concat.startswith('##') and (len(decoded_token) != 3 or ':' in decoded_token[2]):
        return decoded_token_to_str(decoded_token)

    is_word_start = False
    if decoded_token[0] == '▁':
        is_word_start = True
        decoded_token = decoded_token[1:]  # remove the '▁', and add it later

    original_word = ""
    for reduction in decoded_token:
        if ':' in reduction and decoded_token_concat not in [':', '##:']:
            position, letter = reduction.split(':')
            position = int(position)
            if position < 0:
                position = len(original_word) + position + 1
            if len(original_word) == position - 1:
                original_word += letter
            else:
                original_word = original_word[:position] + letter + original_word[position:]
        else:
            original_word += reduction

    original_word = language_utils.replace_final_letters(original_word)
    if is_word_start:
        original_word = '▁' + original_word

    return original_word


def decoded_token_to_str(decoded_token):
    return "[" + ",".join([f"'{reduction}'" for reduction in decoded_token]) + "]"


def list_all_vocab_files(folder_path):
    all_files = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.vocab') and not file.endswith('_decoded.vocab'):
                all_files[os.path.splitext(file)[0]] = f'{root}/{file}'
    sorted_files = dict(sorted(all_files.items(), key=lambda item: f"{(item[0].split('_')[0])}_{int(item[0].split('_')[1]):06}"))
    return sorted_files
