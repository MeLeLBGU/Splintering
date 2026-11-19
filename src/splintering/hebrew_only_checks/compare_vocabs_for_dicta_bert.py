import os

from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.params import set_run_params, get_all_letters_template
from src.language_utils.HebrewUtils import HebrewUtils
from src.utils.path_utils import get_logs_dir
from src.utils.utils import get_new_unicode_chars_map_from_file


def compare_vocabs(baseline_vocab: [str], splinter_decoded_vocab: [str]):
    baseline_vocab_set = set(baseline_vocab)
    decoded_vocab = [line.split("\t\t") for line in splinter_decoded_vocab]
    possible_full_vocab = [line[0] for line in decoded_vocab if len(line) == 2 and line[0] != line[1]]  # BERT differ

    intersection = {token for token in possible_full_vocab if token in baseline_vocab_set}
    non_complete_words = len(decoded_vocab) - len(possible_full_vocab)
    splinter_unique = len(possible_full_vocab) - len(intersection)
    return [
        f"Full vocab: \t\t\t{len(decoded_vocab)} \t(100.00%)",
        f"Non-complete words: \t{non_complete_words}  \t({(non_complete_words / len(decoded_vocab) * 100):.2f}%)",
        f"Intersection: \t\t\t{len(intersection)}  \t({(len(intersection) / len(decoded_vocab) * 100):.2f}%)",
        f"Splinter unique: \t\t{splinter_unique} \t({(splinter_unique / len(decoded_vocab) * 100):.2f}%)",
    ]


def decode_dicta_bert_splintered_vocab(encoded_text, language_utils: LanguageUtilsInterface):
    new_unicode_chars_map = get_new_unicode_chars_map_from_file()
    new_unicode_chars_inverted_map = {value: key for key, value in new_unicode_chars_map.items()}
    decoded_tokens = list()
    i = 0
    for encoded_token in encoded_text:
        i += 1
        decoded_token = [new_unicode_chars_inverted_map.get(char, char) for char in encoded_token]
        rebuilt_token = rebuild_reduced_word(decoded_token, language_utils)
        decoded_tokens.append(rebuilt_token + "\t\t" + decoded_token_to_str(decoded_token))
    return decoded_tokens


def rebuild_reduced_word(decoded_token, language_utils: LanguageUtilsInterface):
    decoded_token_concat = "".join(decoded_token)

    is_middle_of_word = False
    if decoded_token_concat.startswith('##'):
        is_middle_of_word = True
        decoded_token = decoded_token[2:]  # remove the '##', and add it later

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
    if is_middle_of_word:
        original_word = '##' + original_word

    return original_word


def decoded_token_to_str(decoded_token):
    return "[" + ",".join([f"'{reduction}'" for reduction in decoded_token]) + "]"


def get_dicta_bert_tokens(language_utils: LanguageUtilsInterface):
    baseline_vocab_file = "./experiments/2025-01-06-compare_vocabs_dicta_bert/DictaBert - Vocab.txt"
    with open(baseline_vocab_file, 'r', encoding='utf-8') as file:
        baseline_vocab = file.read().splitlines()

    splinter_vocab_file = "./experiments/2025-01-06-compare_vocabs_dicta_bert/SplinterDictaBert - Vocab.txt"
    with open(splinter_vocab_file, 'r', encoding='utf-8') as file:
        splinter_vocab = [line.rstrip("\n") for line in file]
    splinter_decoded_vocab = decode_dicta_bert_splintered_vocab(splinter_vocab, language_utils)

    return baseline_vocab, splinter_decoded_vocab


def main():
    experiment = get_all_letters_template()
    experiment['EXPERIMENT_NAME'] = '2025-01-06-compare_vocabs_dicta_bert'  # all letters
    experiment["TASK_ID"] = '1000000'
    experiment['TOKENIZERS_VOCAB_SIZES'] = [128000, 64000, 32000, 10000, 2000, 1000, 800]
    set_run_params(experiment)
    os.makedirs(get_logs_dir(), exist_ok=True)

    language_utils = HebrewUtils()
    output_path = f'{get_logs_dir()}/vocabs_comparison.txt'
    baseline_vocab, splinter_decoded_vocab = get_dicta_bert_tokens(language_utils)

    with open(f'{get_logs_dir()}/SplinterDictaBert - Vocab - decoded.txt', 'w', encoding='utf-8') as file:
        file.write("\n".join(splinter_decoded_vocab))

    results = compare_vocabs(baseline_vocab, splinter_decoded_vocab)
    with open(output_path, 'a') as file:
        file.write("\n".join(results))


if __name__ == '__main__':
    main()
