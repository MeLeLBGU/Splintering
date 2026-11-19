from abc import ABC, abstractmethod


class LanguageUtilsInterface(ABC):

    @abstractmethod
    def remove_diacritics(self, text: str) -> str:
        pass

    @abstractmethod
    def is_letter_in_language(self, char: str) -> bool:
        pass

    @abstractmethod
    def is_word_contains_letters_from_other_languages(self, word: str) -> bool:
        pass

    @abstractmethod
    def get_language_alphabet(self) -> range:
        pass

    @abstractmethod
    def replace_final_letters(self, text: str) -> str:
        pass

    @abstractmethod
    def save_additional_corpora_for_evaluation(self, text_processor) -> None:
        pass

    @staticmethod
    def replace_last_letter(text, replacement):
        return text[:-1] + replacement