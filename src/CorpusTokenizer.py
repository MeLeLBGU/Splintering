import os

import sentencepiece as spm
from tqdm import tqdm

from src.logger import get_logger


class CorpusTokenizer:

    @staticmethod
    def tokenize_corpus_into_file(tokenizer_path, corpus_path, output_path):
        if os.path.exists(output_path):
            os.remove(output_path)

        get_logger().info(f'start tokenizing corpus {output_path}')
        sp_tokenizer = spm.SentencePieceProcessor(model_file=f'{tokenizer_path}.model')
        with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
            with open(output_path, 'a', encoding='utf-8') as output_file:
                for article_text in tqdm(corpus_file):
                    if article_text is not None:
                        encoded_article = " ".join([str(token_id) for token_id in sp_tokenizer.encode(article_text)])
                        output_file.write(encoded_article + '\n')
        get_logger().info(f'finished tokenizing corpus {output_path}')
