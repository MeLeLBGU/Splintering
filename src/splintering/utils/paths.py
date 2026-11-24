from pathlib import Path
from datetime import datetime, timezone


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class SplinterPaths:

    def __init__(self, experiment_name: str, task_id: str=None, base_directory: Path=None):
        self._home = base_directory or Path("./splinter/")
        self._experiment_name = experiment_name
        self._task_id = task_id or "0"
        self._timestamp = timestamp()

    def _subpath(self, *parts: str) -> Path:
        path = self._home
        for segment in parts:
            path = path / segment
        path.mkdir(exist_ok=True, parents=True)
        return path

    ### Directories

    def _extend_experiments(self, *parts: str) -> Path:
        return self._subpath("experiments", self._experiment_name, *parts)

    def get_raw_data_dir(self) -> Path:
        return self._subpath("raw_data")

    def get_words_dict_dir(self) -> Path:
        return self._subpath("data", "word_dict")

    def get_logs_dir(self) -> Path:
        return self._extend_experiments("logs", "log-" + self._task_id + "-" + self._timestamp)

    def get_corpora_dir(self) -> Path:
        return self._extend_experiments("corpora")

    def get_corpora_tokenized_dir(self) -> Path:
        return self._extend_experiments("corpora_tokenized")

    def get_tokenizers_dir(self) -> Path:
        return self._extend_experiments("results", "tokenizers")

    def get_splinter_dir(self) -> Path:
        return self._extend_experiments("results", "splinter")

    def get_trigram_models_dir(self) -> Path:
        return self._extend_experiments("results", "trigram_models")

    ### Files

    def get_tokenizer_path(self, tokenizer_type: str, vocab_size: int) -> Path:
        return self.get_tokenizers_dir() / f"{tokenizer_type}_{vocab_size}"

    def get_corpus_path(self, corpus_name: str) -> Path:
        return self.get_corpora_dir() / f"{corpus_name}.txt"

    def get_tokenized_corpus_path(self, corpus_name: str, tokenizer_type: str, vocab_size: int) -> Path:
        return self.get_corpora_tokenized_dir() / f"{corpus_name}_{tokenizer_type}_{vocab_size}.txt"

    def create_experiment_dirs(self) -> list[Path]:
        return [
            self.get_logs_dir(),
            self.get_corpora_dir(),
            self.get_corpora_tokenized_dir(),
            self.get_tokenizers_dir(),
            self.get_splinter_dir()
        ]

    def delete_corpus_file_if_exists(self, corpus_name: str):
        corpus_path = self.get_corpus_path(corpus_name)
        if corpus_path.exists():
            corpus_path.unlink()
