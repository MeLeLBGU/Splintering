import sentencepiece as spm

from splintering.utils.logger import get_logger
from splintering.utils.misc import decode_tokens_vocab_file
from splintering.utils.paths import SplinterPaths


def train_tokenizer(paths: SplinterPaths, tokenizer_type, vocab_size, input_path, output_path, is_encoded: bool=True):
    log_path = paths.get_logs_dir() / f"SentencePieceTrainer_{tokenizer_type}_{vocab_size}.log"
    get_logger().info(f'Start training tokenizer {tokenizer_type}_{vocab_size}. Training logs are in logs dir.')
    _train_tokenizer(
        input_file=input_path,
        output_path=output_path,
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        log_path=log_path
    )
    get_logger().info(f'Finished training tokenizer {tokenizer_type}_{vocab_size}.')

    # this is done only for the encoded tokenizer
    if is_encoded:
        decode_tokens_vocab_file(paths, output_path)


def _train_tokenizer(input_file, output_path, tokenizer_type, vocab_size, log_path):
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=output_path,
        vocab_size=vocab_size,
        model_type=tokenizer_type,
        logstream=open(log_path, 'w')
        # split_by_unicode_script=False  # let the vocab contain tokens that contain more than one language
    )
