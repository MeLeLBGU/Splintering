from scripts.params import get_dummy_experiment, Language
from scripts.logger import get_logger, initialize_logger

import sentencepiece as spm

from splintering.text_processors import TextProcessorForDemo
from splintering.utils.languages import LanguageUtilsFactory
from splintering.utils.languages import LanguageUtilsInterface
from splintering.utils.path_utils import SplinterPaths
from splintering.utils.misc import (get_reductions_map_from_file, get_new_unicode_chars_map_from_file,
                                    get_new_unicode_chars_inverted_map_from_file)


def demo(home: SplinterPaths, text: str, language_utils: LanguageUtilsInterface) -> None:
    get_logger().info(f'\nOriginal text: \n{text}')
    text_processor = get_processor(home, language_utils)
    non_encoded_text, encoded_text = text_processor.process(text)
    get_logger().info(f'\n\nNon-encoded splintered text: \n{non_encoded_text}')
    get_logger().info(f'\n\nEncoded splintered text: \n{encoded_text}')
    decoded_text, original_text = text_processor.undo_process(encoded_text)
    get_logger().info(f'\n\nDecoded splintered text: \n{decoded_text}')
    get_logger().info(f'\n\nBack to original text: \n{original_text}')


def get_processor(home: SplinterPaths, language_utils: LanguageUtilsInterface) -> TextProcessorForDemo:
    reductions_map = get_reductions_map_from_file(home)
    new_unicode_chars_map = get_new_unicode_chars_map_from_file(home)
    new_unicode_chars_inverted_map = get_new_unicode_chars_inverted_map_from_file(home)
    text_processor = TextProcessorForDemo(language_utils, reductions_map, new_unicode_chars_map, new_unicode_chars_inverted_map)
    return text_processor


def tokenize_example(home: SplinterPaths, text: str, tokenizer_type: str, vocab_size: int) -> None:
    tokenizer_path = home.get_tokenizers_dir() / f'{tokenizer_type}_{vocab_size}.model'
    sp_tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    encoded_text = " ".join([str(token_id) for token_id in sp_tokenizer.encode(text)])
    get_logger().info(f'tokenized example: {encoded_text}')


if __name__ == '__main__':
    language = Language.find("Hebrew")
    experiment = get_dummy_experiment(language, 'hebrew-letters_subset')
    initialize_logger(experiment)
    get_logger().info(f'Logs will be saved here: {experiment.PATHS.get_logs_dir()}')
    # tokenize_example("עבודה לעבוד במעבדה", "bpe", 128000)
    demo(experiment.PATHS, "נגרר", LanguageUtilsFactory.get_by_language(language))
