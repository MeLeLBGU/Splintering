import csv
import json
import math
import os

FERTILITY_HEADER = "Fertility"
FERTILITY = "fertility"
RENYI_HEADER = "Renyi"
RENYI = "renyi_efficiency"
TOKENS_PER_WORD_HEADER = "Tokens per word distribution"
TOKENS_PER_WORD = "tokens_per_word_distribution"
TOKENS_LENGTH_HEADER = "Tokens length distribution"
TOKENS_LENGTH = "token_length_distribution"
DISTINCT_NEIGHBORS_HEADER = "Distinct neighbors distribution"
DISTINCT_NEIGHBORS = "token_distinct_neighbors_threshold"
TYPES_LENGTH_HEADER = "Types length distribution"
TYPES_LENGTH = "types_length_distribution"
COGNITIVE_EVALUATION_HEADER = "Cognitive evaluation"
COGNITIVE_EVALUATION = "cognitive_score"
WORD_CHUNKABILITY_RTS = "Word chunkability rts"
WORD_CHUNKABILITY_ACCURACY = "Word chunkability accuracy"
NONWORD_CHUNKABILITY_RTS = "NonWord chunkability rts"
NONWORD_CHUNKABILITY_ACCURACY = "NonWord chunkability accuracy"

HEDC4_HEADER = "HeDC4"
PRE_MODERN_HEADER = "pre_modern"
WIKIPEDIA_HEADER = "Wikipedia"

VOCAB_SIZES = [128000, 64000, 32000, 10000, 2000, 1000, 800]

LANGUAGE = 'hebrew'
EXPERIMENT_DATE = '2025-01-21'

if LANGUAGE == 'hebrew':
    CORPORA = [HEDC4_HEADER, PRE_MODERN_HEADER]
    EXPERIMENT_TYPES = ['all_letters', 'letters_subset', 'baseline']
else:
    CORPORA = [WIKIPEDIA_HEADER]
    EXPERIMENT_TYPES = ['all_letters', 'baseline']

def format_static_checks_results_file(file_paths: [str], output_path: str):
    results_map = get_empty_results_map()

    sorted_file_paths = sorted(file_paths)
    for file_path in sorted_file_paths:
        json_data = get_json_data(file_path)
        for result in json_data:
            if FERTILITY in result:
                continue
                # handle_static_check_line(result, results_map)
            elif TYPES_LENGTH in result:
                continue
                # handle_types_length_line(result, results_map)
            elif COGNITIVE_EVALUATION in result:
                handle_cognitive_evaluation_line(result, results_map)
            else:
                raise Exception("Invalid json data")

    save_as_csv(results_map, output_path)


def get_json_data(file_path):
    with open(file_path, "r") as file:
        data = file.read()
    data = f"[{data}]"
    data = data.replace("}\n{", "},\n{")
    json_data = json.loads(data)
    return json_data


def handle_static_check_line(result, results_map):
    corpus_path = result["corpus"].lower()
    corpus_name = get_corpus_name(corpus_path)
    experiment_type = get_experiment_type(corpus_path)
    tokenizer_type = get_tokenizer_type(corpus_path)
    vocab_size = get_vocab_size(corpus_path)
    vocab_size_int = int(vocab_size)
    line_title = f'{experiment_type} - {tokenizer_type}'

    line = results_map[f"{FERTILITY_HEADER} - {corpus_name}"][line_title]
    line[vocab_size] = f'{result[FERTILITY]:.3f}'

    line = results_map[f"{RENYI_HEADER} - {corpus_name}"][line_title]
    line[vocab_size] = f'{result[RENYI]:.3f}'

    line = results_map[f"{TOKENS_PER_WORD_HEADER} - {corpus_name} - {vocab_size}"][line_title]
    fill_tokens_per_word_distribution(line, vocab_size_int, result[TOKENS_PER_WORD].items())

    line = results_map[f"{TOKENS_LENGTH_HEADER} - {corpus_name} - {vocab_size}"][line_title]
    fill_tokens_length_distribution(line, vocab_size_int, result[TOKENS_LENGTH].items())

    line = results_map[f"{DISTINCT_NEIGHBORS_HEADER} - {corpus_name} - {vocab_size}"][line_title]
    fill_distinct_neighbors_distribution(line, vocab_size_int, corpus_name, result[DISTINCT_NEIGHBORS].items())


def fill_tokens_per_word_distribution(line, vocab_size, result_items):
    max_bucket = 4 if (vocab_size > 2000) else 6
    fill_tokens_distributions(line, max_bucket, result_items)


def fill_tokens_length_distribution(line, vocab_size, result_items):
    max_bucket = 8 if (vocab_size > 2000) else 4
    fill_tokens_distributions(line, max_bucket, result_items)


def fill_tokens_distributions(line, max_bucket, result_items):
    if len(line) == 0:
        for i in range(1, max_bucket + 1):
            line[i] = 0
    for key, value in result_items:
        if key == "11+" or int(key) > max_bucket:
            line[max_bucket] += value
        else:
            line[int(key)] = value


def fill_distinct_neighbors_distribution(line, vocab_size, corpus_name, result_items):
    match corpus_name:
        case "pre_modern":
            match vocab_size:
                case 128000 | 64000 | 32000 | 10000:
                    min_bucket = 0
                    buckets_number = 10
                    bucket_width = 50
                case 2000:
                    min_bucket = 0
                    buckets_number = 10
                    bucket_width = 100
                case 1000 | 800:
                    min_bucket = 0
                    buckets_number = 14
                    bucket_width = 50
                case _:
                    raise Exception("Invalid vocab_size")
        case _:
            match vocab_size:
                case 128000 | 64000 | 32000 | 10000:
                    min_bucket = 0
                    buckets_number = 20
                    bucket_width = 500
                case 2000:
                    min_bucket = 1400
                    buckets_number = 6
                    bucket_width = 100
                case 1000:
                    min_bucket = 750
                    buckets_number = 5
                    bucket_width = 50
                case 800:
                    min_bucket = 600
                    buckets_number = 5
                    bucket_width = 40
                case _:
                    raise Exception("Invalid vocab_size")

    max_bucket = min_bucket + bucket_width * (buckets_number - 1)

    if len(line) == 0:
        for key in range(min_bucket, max_bucket + 1, bucket_width):
            line[key] = 0

    for (_, value) in result_items:
        if value > max_bucket:
            bucket = max_bucket
        elif value < min_bucket:
            bucket = min_bucket
        else:
            bucket = math.floor(int(value - min_bucket) / bucket_width) * bucket_width + min_bucket
        line[bucket] += 1


def handle_types_length_line(result, results_map):
    vocab_path = result["vocab"].lower()
    experiment_type = get_experiment_type(vocab_path)
    tokenizer_type = get_tokenizer_type(vocab_path)
    vocab_size = vocab_path.split("_")[-1]
    line_title = f'{experiment_type} - {tokenizer_type}'

    line = results_map[f"{TYPES_LENGTH_HEADER} - {vocab_size}"][line_title]
    fill_types_length_distribution(line, int(vocab_size), result[TYPES_LENGTH].items())


def fill_types_length_distribution(line, vocab_size: int, result_items):
    max_bucket = 8 if (vocab_size > 2000) else 6
    if len(line) == 0:
        for i in range(1, max_bucket + 1):
            line[i] = 0
    for key, value in result_items:
        if int(key) == 0:
            continue
        elif int(key) == 1:
            line[int(key)] = value + 1
        elif int(key) > max_bucket:
            line[max_bucket] += value
        else:
            line[int(key)] = value


def handle_cognitive_evaluation_line(result, results_map):
    corpus_path = result["corpus"].lower()
    experiment_type = get_experiment_type(corpus_path)
    tokenizer_type = get_tokenizer_type(corpus_path)
    vocab_size = get_vocab_size(corpus_path)
    line_title = f'{experiment_type} - {tokenizer_type}'

    line = results_map[COGNITIVE_EVALUATION_HEADER][line_title]
    # line[vocab_size] = (f'{result[COGNITIVE_EVALUATION]:.3f}')
    line[vocab_size] = (f'{result[COGNITIVE_EVALUATION]:.3f}, '
                        f'W-RT: {result[WORD_CHUNKABILITY_RTS]:.3f}, '
                        f'W-ACC: {result[WORD_CHUNKABILITY_ACCURACY]:.3f}, '
                        f'NW-RT: {result[NONWORD_CHUNKABILITY_RTS]:.3f}, '
                        f'NW-ACC: {result[NONWORD_CHUNKABILITY_ACCURACY]:.3f}')

    new_experiment_type = "\\baseline"
    if experiment_type == 'All Letters':
        new_experiment_type = "\\spl"
    elif experiment_type == 'Letters Subset':
        new_experiment_type = "\\ssubset"

    if tokenizer_type == 'UNIGRAM':
        print(f'{tokenizer_type} - {int(vocab_size):06d}: \t\t\t'
              f'& {new_experiment_type:<10}'
              f'& {result[WORD_CHUNKABILITY_RTS]:.3f}          '
              f'& {result[NONWORD_CHUNKABILITY_RTS]:.3f}          '
              f'& {result[WORD_CHUNKABILITY_ACCURACY]:.3f}          '
              f'& {result[NONWORD_CHUNKABILITY_ACCURACY]:.3f}           '
              f'& {result[COGNITIVE_EVALUATION]:.3f}          '
              f'\\\\'
              )


def get_experiment_type(corpus_path):
    if "baseline" in corpus_path:
        return "Baseline"
    elif "subset" in corpus_path:
        return "Letters Subset"
    elif "all_letters" in corpus_path:
        return "All Letters"
    else:
        raise ValueError("type not found")


def get_tokenizer_type(corpus_path):
    if "unigram" in corpus_path:
        return "UNIGRAM"
    elif "bpe" in corpus_path:
        return "BPE"
    else:
        raise ValueError("type not found")


def get_corpus_name(corpus_path):
    if "pre_modern" in corpus_path:
        return PRE_MODERN_HEADER
    elif "hedc4" in corpus_path:
        return HEDC4_HEADER
    elif "wikipedia" in corpus_path:
        return WIKIPEDIA_HEADER
    else:
        raise ValueError("type not found")


def get_vocab_size(corpus_path):
    return corpus_path.split("_")[-1].split(".")[0]


def get_empty_results_map():
    results_map = {COGNITIVE_EVALUATION_HEADER: get_empty_result_line()}
    for corpus in CORPORA:
        results_map[f"{FERTILITY_HEADER} - {corpus}"] = get_empty_result_line()
        results_map[f"{RENYI_HEADER} - {corpus}"] = get_empty_result_line()
        for vocab_size in VOCAB_SIZES:
            results_map[f"{TOKENS_PER_WORD_HEADER} - {corpus} - {vocab_size}"] = get_empty_result_line()
            results_map[f"{TOKENS_LENGTH_HEADER} - {corpus} - {vocab_size}"] = get_empty_result_line()
            results_map[f"{DISTINCT_NEIGHBORS_HEADER} - {corpus} - {vocab_size}"] = get_empty_result_line()
    for vocab_size in VOCAB_SIZES:
        results_map[f"{TYPES_LENGTH_HEADER} - {vocab_size}"] = get_empty_result_line()
    return results_map

def get_empty_result_line():
    if LANGUAGE == 'hebrew':
        return {
            "All Letters - UNIGRAM": {},
            "Letters Subset - UNIGRAM": {},
            "Baseline - UNIGRAM": {},
            "All Letters - BPE": {},
            "Letters Subset - BPE": {},
            "Baseline - BPE": {},
        }
    else:
        return {
            "All Letters - UNIGRAM": {},
            "Baseline - UNIGRAM": {},
            "All Letters - BPE": {},
            "Baseline - BPE": {},
        }

def list_all_files(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "static_checks_results.json" and ('log-0-' in root):
                all_files.append(f'{root}/{file}')
    return all_files


def save_as_csv(data, output_path):
    for key, value in data.items():
        csv_lines = []
        for sub_key, sub_value in value.items():
            csv_line = {"experiment_name": sub_key}
            csv_line.update(sub_value)
            csv_lines.append(csv_line)

        with open(f'{output_path}/{key}.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_lines[0].keys())
            writer.writeheader()
            writer.writerows(csv_lines)


def main():
    files = []
    for experiment_type in EXPERIMENT_TYPES:
        files.extend(list_all_files(f'./experiments/{EXPERIMENT_DATE}-{LANGUAGE}-{experiment_type}/logs/'))

    output_path = f'./experiments/{EXPERIMENT_DATE}-{LANGUAGE}-results'
    os.makedirs(output_path, exist_ok=True)
    format_static_checks_results_file(files, output_path)


if __name__ == "__main__":
    main()
