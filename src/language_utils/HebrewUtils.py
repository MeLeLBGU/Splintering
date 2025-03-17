import re

from src.hebrew_only_checks.cognitive_evaluation import save_cognitive_evaluation_corpus
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.save_dataset_as_text_file import save_hedc4_corpus, save_pre_modern_corpus


class HebrewUtils(LanguageUtilsInterface):
    def remove_diacritics(self, text: str) -> str:
        return re.sub(r'[\u0590-\u05CF]', '', text)

    def is_letter_in_language(self, char: str) -> bool:
        return '\u05D0' <= char <= '\u05EA'

    def is_word_contains_letters_from_other_languages(self, word: str) -> bool:
        return re.search(r'[^\u05D0-\u05EA]', word) is not None

    def get_language_alphabet(self) -> [str]:
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

    def save_additional_corpora_for_evaluation(self, text_processor) -> None:
        save_hedc4_corpus(text_processor=text_processor)
        save_pre_modern_corpus(text_processor=text_processor)
        save_cognitive_evaluation_corpus(text_processor=text_processor)



