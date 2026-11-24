from scripts.params import create_experiment_allletters, Language

from splintering.utils.paths import SplinterPaths
from splintering.utils.languages import HebrewUtils
from splintering.experiments.compare_vocabs_for_dicta_bert import get_dicta_bert_tokens, compare_vocabs


experiment = create_experiment_allletters(Language.find("Hebrew"), SplinterPaths("compare_vocabs_dicta_bert"))
experiment.TOKENIZERS_VOCAB_SIZES = [128000, 64000, 32000, 10000, 2000, 1000, 800]

language_utils = HebrewUtils()
output_path = experiment.PATHS.get_logs_dir() / "vocabs_comparison.txt"
baseline_vocab, splinter_decoded_vocab = get_dicta_bert_tokens(experiment.PATHS, language_utils)

with open(experiment.PATHS.get_logs_dir() / 'SplinterDictaBert - Vocab - decoded.txt', 'w', encoding='utf-8') as file:
    file.write("\n".join(splinter_decoded_vocab))

results = compare_vocabs(baseline_vocab, splinter_decoded_vocab)
with open(output_path, 'a') as file:
    file.write("\n".join(results))
