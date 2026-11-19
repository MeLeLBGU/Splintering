import sentencepiece as spm

from src.logger import get_logger
from src.params import get_run_params
from src.utils.path_utils import get_logs_dir
from src.utils.utils import decode_tokens_vocab_file


def train_tokenizer(tokenizer_type, vocab_size, input_path, output_path):
    log_path = f"{get_logs_dir()}/SentencePieceTrainer - {tokenizer_type}_{vocab_size}.log"
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
    if get_run_params("IS_ENCODED"):
        decode_tokens_vocab_file(output_path)


def _train_tokenizer(input_file, output_path, tokenizer_type, vocab_size, log_path):
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=output_path,
        vocab_size=vocab_size,
        model_type=tokenizer_type,
        logstream=open(log_path, 'w')
        # split_by_unicode_script=False  # let the vocab contain tokens that contain more than one language
    )
