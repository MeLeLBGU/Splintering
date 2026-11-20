# Splintering
This repository contains the code used in [Splintering Nonconcatenative Languages for Better Tokenization](https://arxiv.org/abs/2503.14433). 

It contains the Splinter algorithm as well as the intrinsic evaluation code.

The downstream evaluation DictaBERT-Splinter models can be found [here](https://huggingface.co/dicta-il/dictabert-splinter).

The following example trains Splinter on Hebrew Wikipedia, encodes the training corpus, and uses it to train a BPE tokenizer:

```python
# Training:
from splintering.trainer import SplinterTrainer
from splintering.text_processors import TextProcessorWithEncoding
from splintering.utils.languages import LanguageUtilsFactory
from splintering.utils.path_utils import get_tokenizer_path, get_corpus_path
from splintering.utils.misc import get_corpus_name

language = 'he'
train_dataset_path = 'wikimedia/wikipedia'
train_dataset_name = f'20231101.{language}'
language_utils = LanguageUtilsFactory.get_by_language(language)

# - Create the reductions map, and map the reductions in it into new Unicode characters
splinter_trainer = SplinterTrainer(language_utils)
reductions_map, new_unicode_chars_map, _ = splinter_trainer.train(train_dataset_path, train_dataset_name, None)

# Application:
# - Splinter the corpus
text_processor = TextProcessorWithEncoding(language_utils, reductions_map, new_unicode_chars_map)
text_processor.save_corpus_as_text_file(train_dataset_path, train_dataset_name)

# - Train tokenizer on the splintered corpus 
tokenizer_corpus_path = get_corpus_path(get_corpus_name(train_dataset_path, train_dataset_name))
tokenizer_type = 'bpe'
vocab_size = 128_000

from scripts.train_tokenizer import train_tokenizer  # Thin wrapper around SentencePiece; you can use any package that takes a string corpus.
tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
train_tokenizer(tokenizer_type=tokenizer_type, vocab_size=vocab_size, input_path=tokenizer_corpus_path,
                output_path=tokenizer_path)                     
```

## Citation

If you use Splinter in your research, please cite ```Splintering Nonconcatenative Languages for Better Tokenization```:

```
@misc{gazit2025splinteringnonconcatenativelanguagesbetter,
      title={Splintering Nonconcatenative Languages for Better Tokenization}, 
      author={Bar Gazit and Shaltiel Shmidman and Avi Shmidman and Yuval Pinter},
      year={2025},
      eprint={2503.14433},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.14433}, 
}
```