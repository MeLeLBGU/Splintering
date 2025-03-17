import re

from src.TextProcessorWithEncoding import TextProcessorWithEncoding
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface


class TextProcessorForDemo(TextProcessorWithEncoding):

    def __init__(self, language_utils: LanguageUtilsInterface, reductions_map, new_unicode_chars_map, new_unicode_chars_inverted_map):
        super().__init__(language_utils, reductions_map, new_unicode_chars_map)
        self.new_unicode_chars_inverted_map = new_unicode_chars_inverted_map

    def process(self, text):
        if text is None:
            return ''

        clean_text = self.language_utils.remove_diacritics(text)
        sentences = re.split(r'[.\n]', clean_text)
        reduced_sentences_list = list()
        encoded_sentences_list = list()
        for sentence in sentences:
            word_reductions_list = list()
            encoded_words_list = list()
            words = re.split(r'\s|-|,|:|"|\(|\)', sentence)
            for word in words:
                if len(word) == 0:
                    continue
                word = self.language_utils.replace_final_letters(word)
                # if a word contains letters from other languages, convert only the letters from our language.
                if self.language_utils.is_word_contains_letters_from_other_languages(word):
                    word_reductions = self.get_single_chars_reductions(word)
                    encoded_word = self.get_reduction_for_word_with_letters_from_other_languages(word)
                else:
                    word_reductions = self.get_word_reductions(word)
                    encoded_word = ''.join([self.new_unicode_chars_map[reduction] for reduction in word_reductions])
                word_reductions_list.append(f'{word_reductions}')
                encoded_words_list.append(encoded_word)
            reduced_sentence = " ".join(word_reductions_list)
            encoded_sentence = " ".join(encoded_words_list)
            if len(encoded_sentence) == 0:
                continue
            reduced_sentences_list.append(reduced_sentence)
            encoded_sentences_list.append(encoded_sentence)
        return (
            "\n".join(reduced_sentences_list),
            "\n".join(encoded_sentences_list)
        )

    def undo_process(self, encoded_text):
        if encoded_text is None:
            return ''

        encoded_sentences = re.split(r'[.\n]', encoded_text)
        decoded_sentences_list = list()
        original_sentences_list = list()
        for encoded_sentence in encoded_sentences:
            decoded_words_list = list()
            original_words_list = list()
            encoded_words = encoded_sentence.split()
            for encoded_word in encoded_words:
                if len(encoded_word) == 0:
                    continue
                decoded_word = [self.new_unicode_chars_inverted_map.get(char, char) for char in encoded_word]
                original_word = self.rebuild_reduced_word(decoded_word)
                decoded_words_list.append(f'{decoded_word}')
                original_words_list.append(original_word)

            reduced_sentence = " ".join(decoded_words_list)
            original_sentence = " ".join(original_words_list)
            if len(original_sentence) == 0:
                continue
            decoded_sentences_list.append(reduced_sentence)
            original_sentences_list.append(original_sentence)
        return (
            "\n".join(decoded_sentences_list),
            "\n".join(original_sentences_list)
        )

    def rebuild_reduced_word(self, decoded_word):
        original_word = ""
        for reduction in decoded_word:
            if ':' in reduction:
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

        original_word = self.language_utils.replace_final_letters(original_word)
        return original_word