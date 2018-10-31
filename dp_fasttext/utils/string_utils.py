import re
import gensim
from typing import List

from nltk.corpus import stopwords

stops = stopwords.words("english")

regex = re.compile('[^a-zA-Z\s]')


def remove_non_alpha(s: str) -> str:
    return regex.sub('', s)


def parse(s: str) -> str:
    return remove_non_alpha(s.lower()).strip().replace(" ", "_")


def parse_to_tokens(s: str) -> List[str]:
    s = re.sub(' +', ' ', remove_non_alpha(s)).lower()
    tokens = gensim.utils.simple_preprocess(s, deacc=True, min_len=3)

    return tokens


def parse_sentences(sentences: List[str]) -> List[str]:
    keep = []

    for s in sentences:
        tokens = [t for t in parse_to_tokens(s) if t not in stops]

        if len(tokens) >= 5:
            keep.append(" ".join(tokens))

    return keep
