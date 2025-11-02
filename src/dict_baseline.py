# src/dict_baseline.py
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np

from .features import normalise


def load_lexicon(path: Path) -> list[str]:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = Path(path).read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = Path(path).read_text(encoding="utf-8", errors="replace")

    cues = []
    for line in text.splitlines():
        t = line.strip().lower()
        if not t or t.startswith("#"):
            continue
        cues.append(t)
    return list(dict.fromkeys(cues))


def build_lexicon_structs(cues: list[str]):
    """
    Takes a list of "cues" (single words or multi-word phrases) and splits them into two lookup structures:
        - word_set – a set of all single-word cues.
        - phrase_sets – a dict that groups multi-word cues by their length.
    
    Returns:
      word_set: set[str]
      phrase_sets: dict[int, set[tuple[str,...]]], keys are phrase lengths >= 2
    """
    word_set = set()
    phrase_sets = defaultdict(set)
    for c in cues:
        parts = c.split()
        if len(parts) == 1:
            word_set.add(parts[0])
        else:
            phrase_sets[len(parts)].add(tuple(parts))
    return word_set, dict(phrase_sets)


def any_match(tokens: list[str], word_set, phrase_sets) -> bool:
    """ 
    Checks whether any token or phrase from lexicon appears in a tokenised text.
    Initialises a sliding window of k tokens. Checks the initial window, 
    then slides one token at a time across the text.
    """
    # fast single-word check
    if word_set and any(tok in word_set for tok in tokens):
        return True
    # phrase checks by length
    for k, phrases in phrase_sets.items():
        if len(tokens) < k: 
            continue
        # sliding window of k tokens
        window = tuple(tokens[:k])
        if window in phrases:
            return True
        for i in range(k, len(tokens)):
            window = (*window[1:], tokens[i])  # slide by 1
            if window in phrases:
                return True
    return False


class DictionaryBaseline:
    """
    Fast dictionary baseline:
      - preprocess once per text
      - token/n-gram membership checks
      - outputs proba like sklearn: [p0, p1]
    """
    def __init__(self, word_set, phrase_sets, assume_normalised=False):
        self.word_set = word_set
        self.phrase_sets = phrase_sets
        self.assume_normalised = assume_normalised

    @classmethod
    def from_cues(cls, cues: list[str], assume_normalised=False):
        w, p = build_lexicon_structs(cues)
        return cls(w, p, assume_normalised=assume_normalised)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X: list[str]) -> np.ndarray:
        scores = []
        for x in X:
            x_ = x if self.assume_normalised else normalise(x)
            tokens = x_.split()
            hit = any_match(tokens, self.word_set, self.phrase_sets)
            p1 = 1.0 if hit else 0.0
            scores.append(p1)
        p1 = np.asarray(scores, dtype=float)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X: list[str]) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
