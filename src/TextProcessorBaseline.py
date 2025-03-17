import re

from src.TextProcessorInterface import TextProcessorInterface
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface


class TextProcessorBaseline(TextProcessorInterface):

    def __init__(self, language_utils: LanguageUtilsInterface):
        super().__init__(language_utils)

    def process(self, text):
        if text is None:
            return ''

        clean_text = self.language_utils.remove_diacritics(text)
        sentences = re.split(r'[.\n]', clean_text)
        processed_sentences = [self.process_sentence(sentence) for sentence in sentences]
        processed_sentences = [sentence for sentence in processed_sentences if sentence != '']
        return "\n".join(processed_sentences)

    def process_sentence(self, sentence):
        words = re.split(r'\s|-|,|:|"|\(|\)', sentence)
        filtered_words = [word for word in words if word != '']
        processed_words = [self.language_utils.replace_final_letters(word) for word in filtered_words]
        filtered_sentence = " ".join(processed_words)
        return filtered_sentence
