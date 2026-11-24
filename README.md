# Splintering
This repository contains the code used in [Splintering Nonconcatenative Languages for Better Tokenization](https://aclanthology.org/2025.findings-acl.1151/). 

It contains the Splinter algorithm as well as the intrinsic evaluation code.

The downstream evaluation DictaBERT-Splinter models can be found [here](https://huggingface.co/dicta-il/dictabert-splinter).

The following example trains Splinter on Hebrew Wikipedia, encodes the training corpus, and uses it to train a BPE tokenizer:

```python
# Training:
from splintering.trainer import SplinterTrainer
from splintering.text_processors import TextProcessorWithEncoding
from splintering.utils.languages import LanguageUtilsFactory, Language
from splintering.utils.paths import SplinterPaths
from splintering.utils.misc import get_corpus_name
from splintering.utils.logger import initialize_logger

folder = SplinterPaths("example")
logger = initialize_logger(folder)
language = Language.find("Hebrew")
train_dataset_path = 'wikimedia/wikipedia'
train_dataset_name = f'20231101.{language}'
language_utils = LanguageUtilsFactory.get_by_language(language)

# - Create the reductions map, and map the reductions in it into new Unicode characters
splinter_trainer = SplinterTrainer(language_utils, folder)
reductions_map, new_unicode_chars_map, _ = splinter_trainer.train(train_dataset_path, train_dataset_name, None)

# Application:
# - Splinter the corpus
text_processor = TextProcessorWithEncoding(language_utils, reductions_map, new_unicode_chars_map)
text_processor.save_corpus_as_text_file(train_dataset_path, train_dataset_name)

# - Train tokenizer on the splintered corpus 
tokenizer_corpus_path = folder.get_corpus_path(get_corpus_name(train_dataset_path, train_dataset_name))
tokenizer_type = 'bpe'
vocab_size = 128_000

from scripts.train_tokenizer import train_tokenizer  # Thin wrapper around SentencePiece; you can use any package that takes a string corpus.

tokenizer_path = folder.get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
train_tokenizer(folder, tokenizer_type=tokenizer_type, vocab_size=vocab_size, input_path=tokenizer_corpus_path, output_path=tokenizer_path)
```

## Citation

If you use Splinter in your research, please cite ```Splintering Nonconcatenative Languages for Better Tokenization```:

```
@inproceedings{gazit-etal-2025-splintering,
    title = "Splintering Nonconcatenative Languages for Better Tokenization",
    author = "Gazit, Bar  and
      Shmidman, Shaltiel  and
      Shmidman, Avi  and
      Pinter, Yuval",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1151/",
    doi = "10.18653/v1/2025.findings-acl.1151",
    pages = "22405--22417",
    ISBN = "979-8-89176-256-5"
}
```
