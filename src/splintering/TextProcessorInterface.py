from abc import ABC, abstractmethod

from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface


class TextProcessorInterface(ABC):

    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils

    @abstractmethod
    def process(self, text: str) -> str:
        pass
