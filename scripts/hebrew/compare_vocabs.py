from scripts.params import create_experiment_allletters, Language

from splintering.utils.languages import LanguageUtilsFactory
from splintering.utils.paths import SplinterPaths
from splintering.experiments.compare_vocabs import list_all_vocab_files, decode_splintered_vocab, compare_vocabs


def main(lang: Language, home: SplinterPaths, all_letters_vocabs_file_path: str, baseline_vocabs_file_path: str):
    experiment = create_experiment_allletters(lang, home)
    experiment.TOKENIZERS_VOCAB_SIZES = [128000, 64000, 32000, 10000, 2000, 1000, 800]

    # clean old runs results
    output_path = home.get_logs_dir() / "vocabs_comparison.txt"
    if output_path.exists():
        output_path.unlink()

    all_letters_vocab_files = list_all_vocab_files(all_letters_vocabs_file_path)
    baseline_vocab_files    = list_all_vocab_files(baseline_vocabs_file_path)

    language_utils = LanguageUtilsFactory.get_by_language(experiment.LANGUAGE)
    for vocab_file in all_letters_vocab_files.keys():
        baseline_vocab_file = baseline_vocab_files.get(vocab_file)
        with open(baseline_vocab_file, 'r', encoding='utf-8') as file:
            baseline_vocab = file.read().splitlines()
        baseline_vocab = [token.split('\t')[0] for token in baseline_vocab]

        splinter_vocab_file = all_letters_vocab_files.get(vocab_file)
        with open(splinter_vocab_file, 'r', encoding='utf-8') as file:
            splinter_vocab = [line.rstrip("\n") for line in file]
        splinter_decoded_vocab = decode_splintered_vocab(home, splinter_vocab, language_utils)

        with open(home.get_logs_dir() / f'{vocab_file}_decoded.txt', 'w', encoding='utf-8') as file:
            file.write("\n".join(splinter_decoded_vocab))

        results = compare_vocabs(baseline_vocab, splinter_decoded_vocab)
        with open(output_path, 'a', encoding='utf-8') as file:
            file.write(f"\n\n{vocab_file}:\n")
            file.write("\n".join(results))


if __name__ == "__main__":
    main(
        Language.find("Hebrew"),
        SplinterPaths("compare_vocabs"),  # TODO: The below paths are hardcoded and therefore bad and wrong.
        all_letters_vocabs_file_path='./experiments/2025-01-21-hebrew-all_letters/results/tokenizers',
        baseline_vocabs_file_path='./experiments/2025-01-21-hebrew-baseline/results/tokenizers'
    )
