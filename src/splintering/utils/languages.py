from abc import ABC, abstractmethod

import re

from splintering.experiments.cognitive_evaluation import save_cognitive_evaluation_corpus
from splintering.utils.corpus_tools import TextProcessorInterface


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
    def save_additional_corpora_for_evaluation(self, text_processor: TextProcessorInterface) -> None:
        pass

    @staticmethod
    def replace_last_letter(text, replacement):
        return text[:-1] + replacement


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

    def get_language_alphabet(self) -> list[str]:
        return [chr(char) for char in range(0x0621, 0x064A + 1)]

    def replace_final_letters(self, text: str) -> str:
        return text

    def save_additional_corpora_for_evaluation(self, text_processor) -> None:
        pass


class HebrewUtils(LanguageUtilsInterface):
    def remove_diacritics(self, text: str) -> str:
        return re.sub(r'[\u0590-\u05CF]', '', text)

    def is_letter_in_language(self, char: str) -> bool:
        return '\u05D0' <= char <= '\u05EA'

    def is_word_contains_letters_from_other_languages(self, word: str) -> bool:
        return re.search(r'[^\u05D0-\u05EA]', word) is not None

    def get_language_alphabet(self) -> list[str]:
        return [chr(char) for char in range(0x05D0, 0x05EA + 1)]

    def replace_final_letters(self, text: str) -> str:
        if text == '':
            return text
        elif text[-1] == 'ך':
            return self.replace_last_letter(text, 'כ')
        elif text[-1] == 'כ':
            return self.replace_last_letter(text, 'ך')
        elif text[-1] == 'ם':
            return self.replace_last_letter(text, 'מ')
        elif text[-1] == 'מ':
            return self.replace_last_letter(text, 'ם')
        elif text[-1] == 'ן':
            return self.replace_last_letter(text, 'נ')
        elif text[-1] == 'נ':
            return self.replace_last_letter(text, 'ן')
        elif text[-1] == 'ף':
            return self.replace_last_letter(text, 'פ')
        elif text[-1] == 'פ':
            return self.replace_last_letter(text, 'ף')
        elif text[-1] == 'ץ':
            return self.replace_last_letter(text, 'צ')
        elif text[-1] == 'צ':
            return self.replace_last_letter(text, 'ץ')
        else:
            return text

    def save_additional_corpora_for_evaluation(self, text_processor: TextProcessorInterface) -> None:
        text_processor.save_hedc4_corpus()
        text_processor.save_pre_modern_corpus()
        save_cognitive_evaluation_corpus(text_processor=text_processor)


class MalayUtils(LanguageUtilsInterface):
    def remove_diacritics(self, text: str) -> str:
        return text

    def is_letter_in_language(self, char: str) -> bool:
        return ('A' <= char <= 'Z') or ('a' <= char <= 'z')

    def is_word_contains_letters_from_other_languages(self, word: str) -> bool:
        return re.search(r'[^a-zA-Z]', word) is not None

    def get_language_alphabet(self) -> list[str]:
        return [chr(i) for i in range(97, 123)] + [chr(i) for i in range(65, 91)]

    def replace_final_letters(self, text: str) -> str:
        return text

    def save_additional_corpora_for_evaluation(self, text_processor) -> None:
        pass


########################################################################################################################


class LanguageUtilsFactory:
    @staticmethod
    def get_by_language(language: str) -> LanguageUtilsInterface:
        switch = {
            'he': HebrewUtils(),
            'ar': ArabicUtils(),
            'ms': MalayUtils(),
        }
        return switch.get(language)
