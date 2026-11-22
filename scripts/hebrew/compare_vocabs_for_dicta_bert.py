from scripts.params import set_run_params, get_all_letters_template

from splintering.utils.path_utils import get_logs_dir
from splintering.utils.languages import HebrewUtils
from splintering.experiments.compare_vocabs_for_dicta_bert import get_dicta_bert_tokens, compare_vocabs


experiment = get_all_letters_template()
experiment['EXPERIMENT_NAME'] = '2025-01-06-compare_vocabs_dicta_bert'  # all letters
experiment["TASK_ID"] = '1000000'
experiment['TOKENIZERS_VOCAB_SIZES'] = [128000, 64000, 32000, 10000, 2000, 1000, 800]
set_run_params(experiment)

import os
os.makedirs(get_logs_dir(), exist_ok=True)

language_utils = HebrewUtils()
output_path = f'{get_logs_dir()}/vocabs_comparison.txt'
baseline_vocab, splinter_decoded_vocab = get_dicta_bert_tokens(language_utils)

with open(f'{get_logs_dir()}/SplinterDictaBert - Vocab - decoded.txt', 'w', encoding='utf-8') as file:
    file.write("\n".join(splinter_decoded_vocab))

results = compare_vocabs(baseline_vocab, splinter_decoded_vocab)
with open(output_path, 'a') as file:
    file.write("\n".join(results))
