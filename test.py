import spacy
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

"""
To install spacy languages do:
python -m spacy download en
python -m spacy download de
"""
# spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

#
def tokenize_ger(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k(
    language_pair=("de", "en")
)
print()
# german.build_vocab(train_data, max_size=10000, min_freq=2)
# english.build_vocab(train_data, max_size=10000, min_freq=2)