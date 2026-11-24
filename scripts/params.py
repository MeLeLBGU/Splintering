from typing import Optional
from pathlib import Path
from dataclasses import dataclass

from langcodes import Language
import copy

from splintering.utils.paths import SplinterPaths


PATH_DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class SplinterExperiment:
    PATHS: SplinterPaths

    RUN_STATIC_CHECKS: bool
    STATIC_CHECKS_CORPORA: list[str]
    TOKENIZE_CORPORA: bool

    LANGUAGE: Language
    SPLINTER_TRAINING_CORPUS_PATH: str
    SPLINTER_TRAINING_CORPUS_NAME: str
    SPLINTER_LETTERS_SUBSET: Optional[list[str]]
    SAVE_CORPORA_INTO_FILE: bool
    IS_ENCODED: bool

    TRAIN_TOKENIZERS: bool
    TOKENIZERS_TYPES: list[str]
    TOKENIZERS_VOCAB_SIZES: list[int]


def get_experiment_default(lang: Language, paths: SplinterPaths) -> SplinterExperiment:
    return SplinterExperiment(
        PATHS=paths,
        RUN_STATIC_CHECKS=True,
        STATIC_CHECKS_CORPORA=['hebrew_diacritized_pre_modern', 'HeNLP_HeDC4'] if lang.to_tag() == "he" else [f'wikimedia_wikipedia_20231101_{lang.to_tag()}'],

        TOKENIZE_CORPORA=False,
        LANGUAGE=lang,
        SPLINTER_TRAINING_CORPUS_PATH='wikimedia/wikipedia',
        SPLINTER_TRAINING_CORPUS_NAME=f'20231101.{lang.to_tag()}',
        SPLINTER_LETTERS_SUBSET=None,
        SAVE_CORPORA_INTO_FILE=False,
        IS_ENCODED=True,

        TRAIN_TOKENIZERS=False,
        TOKENIZERS_TYPES=['unigram', 'bpe'],
        TOKENIZERS_VOCAB_SIZES=[128_000]
    )


def get_dummy_experiment(lang: Language, experiment_name: str) -> SplinterExperiment:
    return get_experiment_default(lang, SplinterPaths(experiment_name, '1000000', base_directory=PATH_DATA))


def create_experiment_baseline(lang: Language, paths: SplinterPaths) -> SplinterExperiment:
    baseline_template = get_experiment_default(lang, paths)
    baseline_template.EXPERIMENT_NAME = f'{lang.language_name()}-baseline'  # baseline
    baseline_template.IS_ENCODED = False  # False = baseline experiment
    baseline_template.SPLINTER_LETTERS_SUBSET = None
    return baseline_template


def create_experiment_allletters(lang: Language, paths: SplinterPaths) -> SplinterExperiment:
    all_letters_template = get_experiment_default(lang, paths)
    all_letters_template.EXPERIMENT_NAME = f'{lang.language_name()}-all_letters'  # all letters
    all_letters_template.IS_ENCODED = True
    all_letters_template.SPLINTER_LETTERS_SUBSET = None  # None = all letters
    return all_letters_template


def create_experiment_someletters(lang: Language, paths: SplinterPaths) -> SplinterExperiment:
    letters_subset_template = get_experiment_default(lang, paths)
    letters_subset_template.EXPERIMENT_NAME = f'{lang.language_name()}-letters_subset'  # letters subset
    letters_subset_template.IS_ENCODED = True
    letters_subset_template.SPLINTER_LETTERS_SUBSET = ["א", "ב", "ה", "ו", "י", "כ", "ל", "מ", "נ", "ש", "ת"]
    return letters_subset_template


def split_to_separate_runs(template: SplinterExperiment) -> list[SplinterExperiment]:
    runs = []
    for vocab_size in template.TOKENIZERS_VOCAB_SIZES:
        run = copy.deepcopy(template)
        run.TOKENIZERS_VOCAB_SIZES = [vocab_size]
        runs.append(run)
    return runs
