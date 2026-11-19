import os

from datasets import load_dataset
from tqdm import tqdm

from TextProcessorInterface import TextProcessorInterface
from src.logger import get_logger
from src.utils.path_utils import get_raw_data_dir, delete_corpus_file_if_exists, get_corpus_path
from src.utils.utils import get_corpus_name


def save_corpus_as_text_file(text_processor: TextProcessorInterface, dataset_path: str, dataset_name: str, batch_size=10000):
    corpus_name = get_corpus_name(dataset_path, dataset_name)
    get_logger().info(f'Start saving {corpus_name} corpus as text file with {text_processor.__class__.__name__}')
    dataset = load_dataset(dataset_path, dataset_name, split="train", cache_dir=get_raw_data_dir())
    delete_corpus_file_if_exists(corpus_name)
    for i, j in enumerate(tqdm(range(0, len(dataset), batch_size))):
        dataset_batch = dataset[j:j + batch_size]
        process_batch(text_processor, dataset_batch, corpus_name)
    get_logger().info(f'Finished saving {corpus_name} corpus')


def save_hedc4_corpus(text_processor: TextProcessorInterface, batch_size=10000):
    corpus_name = 'HeNLP_HeDC4'
    get_logger().info(f'Start saving {corpus_name} corpus as text file with {text_processor.__class__.__name__}')
    corpus_path = f'{get_raw_data_dir()}/HeNLP-he_dc4-10percent/HeDC4-10percent-seed42-sample.csv'
    if not os.path.exists(corpus_path):
        get_logger().info(f'HeDC4-10percent file was not found - creating it from full corpus')
        full_corpus = load_dataset('HeNLP/HeDC4', split="train", cache_dir=get_raw_data_dir())
        sample_lines_from_corpus_to_csv(full_corpus, corpus_path)

    dataset = load_dataset("csv", data_files=corpus_path)
    delete_corpus_file_if_exists(corpus_name)
    for i, j in enumerate(tqdm(range(0, len(dataset['train']), batch_size))):
        dataset_batch = dataset['train'][j:j + batch_size]
        process_batch(text_processor, dataset_batch, corpus_name)
    get_logger().info(f'Finished saving {corpus_name} corpus')


def sample_lines_from_corpus_to_csv(full_corpus, output_file):
    get_logger().info(f'Start sample lines from corpus')
    random_seed = 42
    sample_size = int(0.1 * len(full_corpus))
    sampled_dataset = full_corpus.shuffle(seed=random_seed).select(range(sample_size))
    sampled_dataset.to_csv(output_file, index=False)
    get_logger().info(f'Finish sample lines from corpus')


# hebrew_diacritized repository needs to be git cloned as a prerequisite from
# https://github.com/elazarg/hebrew_diacritized.git
# into Splinter/raw_data/hebrew_diacritized folder
def save_pre_modern_corpus(text_processor: TextProcessorInterface):
    corpus_name = 'hebrew_diacritized_pre_modern'
    get_logger().info(f'Start saving {corpus_name} corpus as text file with {text_processor.__class__.__name__}')
    path = f'{get_raw_data_dir()}/hebrew_diacritized/pre_modern'
    files = get_all_files_in_dir(path)
    dataset_batch = {'text': list()}
    for filename in files:
        with open(f'{filename}', 'r', encoding='utf-8') as file:
            dataset_batch['text'].append("\n".join(file.readlines()))
    delete_corpus_file_if_exists(corpus_name)
    process_batch(text_processor, dataset_batch, corpus_name)
    get_logger().info(f'Finished saving {corpus_name} corpus')


def get_all_files_in_dir(directory):
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files]


def process_batch(text_processor, dataset_batch, corpus_name):
    encoded_articles_list = list()
    for article_text in dataset_batch["text"]:
        encoded_article = text_processor.process(article_text)
        encoded_articles_list.append(encoded_article)
    with open(get_corpus_path(corpus_name), 'a', encoding='utf-8') as file:
        file.write("\n".join(encoded_articles_list))
