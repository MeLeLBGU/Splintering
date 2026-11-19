import re

from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface


class ArabicUtils(LanguageUtilsInterface):
    def remove_diacritics(self, text: str) -> str:
        # text = text.replace('\u0622', '\u0627')
        # text = text.replace('\u0623', '\u0627')
        # text = text.replace('\u0624', '\u0648')
        # text = text.replace('\u0625', '\u0627')
        # text = text.replace('\u0626', '\u064A')
        return re.sub(r'[\u064B-\u065F]', '', text)

    def is_letter_in_language(self, char: str) -> bool:
        return '\u0621' <= char <= '\u064A'

    def is_word_contains_letters_from_other_languages(self, word: str) -> bool:
        return re.search(r'[^\u0621-\u064A]', word) is not None

    def get_language_alphabet(self) -> [str]:
        return [chr(char) for char in range(0x0621, 0x064A + 1)]

    def replace_final_letters(self, text: str) -> str:
        return text

    def save_additional_corpora_for_evaluation(self, text_processor) -> None:
        pass
