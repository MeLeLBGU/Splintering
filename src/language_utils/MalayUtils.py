import re

from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface


class MalayUtils(LanguageUtilsInterface):
    def remove_diacritics(self, text: str) -> str:
        return text

    def is_letter_in_language(self, char: str) -> bool:
        return ('A' <= char <= 'Z') or ('a' <= char <= 'z')

    def is_word_contains_letters_from_other_languages(self, word: str) -> bool:
        return re.search(r'[^a-zA-Z]', word) is not None

    def get_language_alphabet(self) -> [str]:
        return [chr(i) for i in range(97, 123)] + [chr(i) for i in range(65, 91)]

    def replace_final_letters(self, text: str) -> str:
        return text

    def save_additional_corpora_for_evaluation(self, text_processor) -> None:
        pass
