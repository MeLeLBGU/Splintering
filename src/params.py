import copy
from datetime import datetime, timezone

run_params = {}  # value is filled by set_run_params in main


def set_run_params(params):
    global run_params
    run_params = params
    run_params["TIMESTAMP"] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def get_run_params(key):
    global run_params
    return run_params[key]


def get_all_run_params():
    global run_params
    return run_params

LANGUAGE = 'he'
LANGUAGE_FULL = 'hebrew'
EXPERIMENT_DATE = '2025-01-21'

if LANGUAGE == 'he':
    STATIC_CHECKS_CORPORA = ['hebrew_diacritized_pre_modern', 'HeNLP_HeDC4']
else:
    STATIC_CHECKS_CORPORA = [f'wikimedia_wikipedia_20231101_{LANGUAGE}']

experiment_template = {
        'LANGUAGE': LANGUAGE,
        'SAVE_CORPORA_INTO_FILE': False,
        'SPLINTER_TRAINING_CORPUS_PATH': 'wikimedia/wikipedia',
        'SPLINTER_TRAINING_CORPUS_NAME': f'20231101.{LANGUAGE}',
        'TRAIN_TOKENIZERS': False,
        'TOKENIZERS_TYPES': ['unigram', 'bpe'],
        'TOKENIZE_CORPORA': False,
        'RUN_STATIC_CHECKS': True,
        'STATIC_CHECKS_CORPORA': STATIC_CHECKS_CORPORA,
    }


def get_all_letters_template():
    all_letters_template = copy.deepcopy(experiment_template)
    all_letters_template['EXPERIMENT_NAME'] = f'{EXPERIMENT_DATE}-{LANGUAGE_FULL}-all_letters'  # all letters
    all_letters_template['IS_ENCODED'] = True
    all_letters_template['SPLINTER_LETTERS_SUBSET'] = None  # None = all letters
    return all_letters_template


def get_letters_subset_template():
    letters_subset_template = copy.deepcopy(experiment_template)
    letters_subset_template['EXPERIMENT_NAME'] = f'{EXPERIMENT_DATE}-{LANGUAGE_FULL}-letters_subset'  # letters subset
    letters_subset_template['IS_ENCODED'] = True
    letters_subset_template['SPLINTER_LETTERS_SUBSET'] = ["א", "ב", "ה", "ו", "י", "כ", "ל", "מ", "נ", "ש", "ת"]
    return letters_subset_template


def get_baseline_template():
    baseline_template = copy.deepcopy(experiment_template)
    baseline_template['EXPERIMENT_NAME'] = f'{EXPERIMENT_DATE}-{LANGUAGE_FULL}-baseline'  # baseline
    baseline_template['IS_ENCODED'] = False  # False = baseline experiment
    baseline_template['SPLINTER_LETTERS_SUBSET'] = None
    return baseline_template


def split_to_separate_runs(template):
    runs = []
    vocab_sizes = [128000, 64000, 32000, 10000, 2000, 1000, 800]
    for vocab_size in vocab_sizes:
        run = copy.deepcopy(template)
        run['TOKENIZERS_VOCAB_SIZES'] = [vocab_size]
        runs.append(run)
    return runs


def get_dummy_experiment(experiment_name: str):
    experiment = copy.deepcopy(get_baseline_template())
    experiment['EXPERIMENT_NAME'] = experiment_name
    experiment["TASK_ID"] = '1000000'
    experiment['TOKENIZERS_VOCAB_SIZES'] = [128000]
    return experiment


experiments = []
experiments.extend(split_to_separate_runs(get_all_letters_template()))
if LANGUAGE == 'he':
    experiments.extend(split_to_separate_runs(get_letters_subset_template()))
experiments.extend(split_to_separate_runs(get_baseline_template()))
