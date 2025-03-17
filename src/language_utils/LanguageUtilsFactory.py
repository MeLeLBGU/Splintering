from src.language_utils.ArabicUtils import ArabicUtils
from src.language_utils.HebrewUtils import HebrewUtils
from src.language_utils.MalayUtils import MalayUtils


class LanguageUtilsFactory:

    @staticmethod
    def get_by_language(language: str):
        switch = {
            'he': HebrewUtils(),
            'ar': ArabicUtils(),
            'ms': MalayUtils(),
        }
        return switch.get(language)