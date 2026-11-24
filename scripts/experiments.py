from scripts.params import split_to_separate_runs, create_experiment_allletters, create_experiment_someletters, \
    create_experiment_baseline, SplinterPaths, SplinterExperiment, Language

LANGUAGE = Language.find("Hebrew")
VOCAB_SIZES = [128000, 64000, 32000, 10000, 2000, 1000, 800]


def create_all_experiments(home: SplinterPaths) -> list[SplinterExperiment]:
    experiments = []
    experiments.extend(split_to_separate_runs(create_experiment_allletters(LANGUAGE, home)))
    if LANGUAGE == Language.find("Hebrew"):
        experiments.extend(split_to_separate_runs(create_experiment_someletters(LANGUAGE, home)))
    experiments.extend(split_to_separate_runs(create_experiment_baseline(LANGUAGE, home)))
    return experiments
