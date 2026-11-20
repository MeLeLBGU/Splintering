import re

from splintering.utils.corpus_tools import TextProcessorInterface
from splintering.utils.misc import get_permutation
from splintering.utils.languages import LanguageUtilsInterface


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


class TextProcessorWithEncoding(TextProcessorInterface):

    def __init__(self, language_utils: LanguageUtilsInterface, reductions_map, new_unicode_chars_map):
        super().__init__(language_utils)
        self.reductions_map = reductions_map
        self.new_unicode_chars_map = new_unicode_chars_map
        self.word_reductions_cache = dict()

    def process(self, text):
        if text is None:
            return ''

        clean_text = self.language_utils.remove_diacritics(text)
        sentences = re.split(r'[.\n]', clean_text)
        encoded_sentences_list = list()
        for sentence in sentences:
            encoded_words_list = list()
            words = re.split(r'\s|-|,|:|"|\(|\)', sentence)
            for word in words:
                if len(word) == 0:
                    continue
                original_word = word
                if original_word in self.word_reductions_cache:
                    encoded_word = self.word_reductions_cache[original_word]
                else:
                    word = self.language_utils.replace_final_letters(word)
                    # if a word contains letters from other languages, convert only the letters from our language.
                    if self.language_utils.is_word_contains_letters_from_other_languages(word):
                        encoded_word = self.get_reduction_for_word_with_letters_from_other_languages(word)
                    else:
                        word_reductions = self.get_word_reductions(word)
                        encoded_word = ''.join([self.new_unicode_chars_map[reduction] for reduction in word_reductions])
                encoded_words_list.append(encoded_word)
                self.word_reductions_cache[original_word] = encoded_word
            encoded_sentence = " ".join(encoded_words_list)
            if len(encoded_sentence) == 0:
                continue
            encoded_sentences_list.append(encoded_sentence)
        return "\n".join(encoded_sentences_list)

    def get_word_reductions(self, word):
        reduced_word = word
        reductions = []
        while len(reduced_word) > 3:
            # if this word length has no known reductions - return what's left of the word as is
            if len(reduced_word) not in self.reductions_map:
                reductions.extend(self.get_single_chars_reductions(reduced_word))
                break
            reduction = self.get_reduction(reduced_word, 3, 3)
            if reduction is not None:
                position = int(reduction.split(':')[0])
                reductions.append(reduction)
                reduced_word = get_permutation(reduced_word, position, len(reduced_word))
            # if we couldn't find a reduction - return what's left of the word as is
            else:
                reductions.extend(self.get_single_chars_reductions(reduced_word))
                break

        # if we found all reductions and left only with the suspected root - keep it as is
        if len(reduced_word) < 4:
            reductions.extend(self.get_single_chars_reductions(reduced_word))

        reductions.reverse()
        return reductions

    def get_reduction(self, word, depth, width):
        curr_step_reductions = [{"word": word, "reduction": None, "root_reduction": None, "score": 1}]
        word_length = len(word)
        i = 0
        while i < depth and len(curr_step_reductions) > 0 and word_length > 3:
            next_step_reductions = list()
            for reduction in curr_step_reductions:
                possible_reductions = self.get_most_frequent_reduction_keys(
                    reduction["word"],
                    reduction["root_reduction"],
                    reduction["score"],
                    width,
                    word_length
                )
                next_step_reductions += possible_reductions
            curr_step_reductions = list(next_step_reductions)
            i += 1
            word_length -= 1

        max_score_reduction = None
        if len(curr_step_reductions) > 0:
            max_score_reduction = max(curr_step_reductions, key=lambda x: x["score"])["root_reduction"]
        return max_score_reduction

    def get_most_frequent_reduction_keys(self, word, root_reduction, parent_score, number_of_reductions, word_length):
        if len(word) not in self.reductions_map:
            return list()

        possible_reductions = list()
        for reduction, score in self.reductions_map[len(word)].items():
            position, letter = reduction.split(':')
            position = int(position)
            if word[position] == letter:
                permutation = get_permutation(word, position, word_length)
                possible_reductions.append({
                    "word": permutation,
                    "reduction": reduction,
                    "root_reduction": root_reduction if root_reduction is not None else reduction,
                    "score": parent_score * score
                })
                if len(possible_reductions) >= number_of_reductions:
                    break
        return possible_reductions

    def get_reduction_for_word_with_letters_from_other_languages(self, word):
        return ''.join(self.new_unicode_chars_map[char] if self.language_utils.is_letter_in_language(char) else char for char in word)

    @staticmethod
    def get_single_chars_reductions(reduced_word):
        reductions = []
        for char in reduced_word[::-1]:
            reductions.append(char)
        return reductions


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