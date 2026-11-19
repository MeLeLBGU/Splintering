import json
import os

import pandas as pd
import sentencepiece as spm
from scipy.stats import pearsonr

from src.TextProcessorInterface import TextProcessorInterface
from src.logger import get_logger, initialize_logger
from src.params import experiments, set_run_params, get_run_params
from src.utils.path_utils import get_corpora_dir, get_raw_data_dir, get_corpus_path, get_tokenized_corpus_path, \
    create_experiment_dirs, get_tokenizer_path
from src.utils.utils import add_static_result_to_file


def main():
    for tokenizer_type in get_run_params("TOKENIZERS_TYPES"):
        for vocab_size in get_run_params("TOKENIZERS_VOCAB_SIZES"):
            tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
            tokenize_cognitive_evaluation_file(tokenizer_path, tokenizer_type, vocab_size)
            cognitive_evaluation = run_cognitive_evaluation(tokenizer_type, vocab_size)
            add_static_result_to_file(cognitive_evaluation)


def save_cognitive_evaluation_corpus(text_processor: TextProcessorInterface):
    get_logger().info(f'Start saving HeLP_The_Hebrew_Lexicon_Project corpus as text file with {text_processor.__class__.__name__}')
    data = pd.read_excel(f'{get_raw_data_dir()}/HeLP_The_Hebrew_Lexicon_Project/raw_LD_data.xlsx')
    filtered_data = filter_data(data)  # do the same filtering as in the original HeLP paper

    aggregated_by_stimuli = filtered_data.groupby("Stimulus", as_index=False).agg(
        Stimulus=("Stimulus", "first"),
        mean_RT=("RT", "mean"),
        mean_Correctness=("Correctness", "mean"),
        n=("RT", "size"),
        Stimulus_type=("Stimulus_type", "first")
    )

    aggregated_by_stimuli["Stimulus"] = [text_processor.process(stimulus) for stimulus in aggregated_by_stimuli["Stimulus"]]
    aggregated_by_stimuli.to_csv(f'{get_corpora_dir()}/HeLP_cognitive_evaluation.txt', index=False)
    get_logger().info(f'Finished saving HeLP_The_Hebrew_Lexicon_Project corpus')


# do the same filtering as in the original HeLP paper
def filter_data(data):
    # remove rows without RT or RT less than 300ms
    filtered_data = data.dropna(subset=["RT"])
    filtered_data = filtered_data[filtered_data["RT"] >= 300]
    aggregated_by_session = (
        filtered_data
        .groupby(["Subject", "Session_date"], as_index=False)
        .agg(
            mean_RT=("RT", "mean"),
            mean_Correct=("Correctness", "mean"),
            n=("RT", "size"),
            Session_date=("Session_date", "first"),
            Subject=("Subject", "first")
        ))

    # Filter sessions with n < 880 or mean_Correct < 0.75
    sessions_to_remove = aggregated_by_session[(aggregated_by_session["n"] < 880) | (aggregated_by_session["mean_Correct"] < 0.75)]
    for _, session in sessions_to_remove.iterrows():
        filtered_data = filtered_data[(filtered_data["Subject"] != session["Subject"]) | (filtered_data["Session_date"] != session["Session_date"])]
    return filtered_data


def tokenize_cognitive_evaluation_file(tokenizer_path, tokenizer_type, vocab_size):
    corpus_path = get_corpus_path('HeLP_cognitive_evaluation')
    tokenized_corpus_path = get_tokenized_corpus_path('HeLP_cognitive_evaluation', tokenizer_type, vocab_size)
    if os.path.exists(tokenized_corpus_path):
        os.remove(tokenized_corpus_path)

    get_logger().info(f'cognitive_evaluation: start tokenizing corpus with {tokenizer_type} tokenizer, vocab size {vocab_size}.')
    sp_tokenizer = spm.SentencePieceProcessor(model_file=f'{tokenizer_path}.model')
    df = pd.read_csv(corpus_path)
    df["Stimulus_tokenized"] = [sp_tokenizer.encode(stimulus) for stimulus in df["Stimulus"]]
    df = df[["Stimulus_tokenized"] + [col for col in df.columns if col != "Stimulus_tokenized"]]
    df.to_csv(tokenized_corpus_path, index=False)
    get_logger().info('cognitive_evaluation: finished tokenizing corpus')


def run_cognitive_evaluation(tokenizer_type, vocab_size):
    tokenized_corpus_path = get_tokenized_corpus_path('HeLP_cognitive_evaluation', tokenizer_type, vocab_size)
    data = pd.read_csv(tokenized_corpus_path)
    data = data.dropna()
    data["wordiness"] = data.apply(get_wordiness, axis=1)
    results = {}
    results.update(get_chunkability("Word", data))
    results.update(get_chunkability("NonWord", data))
    # cognitive score is the average of all the pearson correlations results
    results["cognitive_score"] = sum(abs(value) for value in results.values()) / len(results)
    results["corpus"] = tokenized_corpus_path
    return results


def get_wordiness(word_data):
    number_of_tokens = len(json.loads(word_data["Stimulus_tokenized"]))
    word_length = len(word_data["Stimulus"])
    wordiness = 1 - (number_of_tokens / word_length)
    return wordiness


def get_chunkability(category, data):
    results = {}
    category_data = data[data["Stimulus_type"] == category]
    wordiness = list(category_data["wordiness"])
    reaction_times = list(category_data["mean_RT"])
    wordiness_rc_correlation, _ = pearsonr(wordiness, reaction_times)
    results[f'{category} chunkability rts'] = wordiness_rc_correlation
    accuracy = list(category_data["mean_Correctness"])
    wordiness_accuracy_correlation, _ = pearsonr(wordiness, accuracy)
    results[f'{category} chunkability accuracy'] = wordiness_accuracy_correlation
    return results


if __name__ == '__main__':
    task_id = 0
    experiment = experiments[task_id]
    experiment["TASK_ID"] = task_id
    set_run_params(experiment)
    create_experiment_dirs()
    initialize_logger()
    get_logger().info(f'Experiment started')
    try:
        main()
    except Exception as e:
        get_logger().exception(f'Experiment failed with error: \n{e}')
        exit(1)
    get_logger().info(f'Experiment finished')