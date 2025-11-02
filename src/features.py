# src/features.py
from __future__ import annotations
import re
import numpy as np
from typing import List, Iterable
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


# --- engineered/meta features 
URL_RE     = re.compile(r"https?://\S+|www\.\S+")
USER_RE    = re.compile(r"@\w+")
HTML_RE    = re.compile(r"<[^>]+>")
WS_RE      = re.compile(r"\s+")
ELONG_RE   = re.compile(r"(.)\1{2,}") 
REP_PUNCT  = re.compile(r"[!?]{2,}")
ALPHA_RE   = re.compile(r"[A-Za-z]+")
SEC_PERSON = {"you", "u", "ur", "ya", "youre", "you're"}


class RawTextFeats(BaseEstimator, TransformerMixin):
    """
    Compact raw-text features (column order):
      0: caps_ratio                 (A-Z letters / letters)
      1: exclam_rate                (#'!' / length)
      2: question_rate              (#'?' / length)
      3: punct_ratio                (.,;:!?\"'()[]{} / length)
      4: url_count
      5: elongation_rate            (tokens with elongation / tokens)
      6: allcaps_token_share        (ALLCAP tokens / tokens)
      7: digit_ratio                (digits / length)
      8: repeated_punct_rate        (runs of [!?]{2,} per 100 chars)
      9: elongation_count           (number of elongated tokens)
     10: lex_div                    (unique tokens / tokens)
     11: avg_word_len               (mean alphabetic token length)
     12: second_person_ratio        (2nd-person tokens / tokens)
     13: mention_count              (number of @mentions)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for s in X:
            if s is None:
                s = ""
            s2 = " ".join(str(s).split())
            length = max(1, len(s2))

            # character-level
            letters = [c for c in s2 if c.isalpha()]
            caps_ratio = (sum(1 for c in letters if c.isupper()) / max(1, len(letters)))
            exclam_rate   = s2.count("!") / length
            question_rate = s2.count("?") / length
            punct_ratio   = (sum(1 for c in s2 if c in ".,;:!?\"'()[]{}") / length)
            url_count     = len(URL_RE.findall(s2))
            digit_ratio   = (sum(1 for c in s2 if c.isdigit()) / length)
            rep_runs      = len(REP_PUNCT.findall(s2))
            repeated_punct_rate = rep_runs / (length / 100.0) 
            mention_count = len(USER_RE.findall(s2))

            # token-level
            tokens = s2.split()
            n_tok = max(1, len(tokens))
            tokens_lower = [t.lower() for t in tokens]

            elongation_mask = [1 if ELONG_RE.search(t) else 0 for t in tokens]
            elongation_count = int(np.sum(elongation_mask))
            elongation_rate  = elongation_count / n_tok

            allcaps_token_share = sum(1 for t in tokens if t.isalpha() and t.upper() == t and len(t) >= 2) / n_tok
            second_person_ratio = sum(1 for t in tokens_lower if t in SEC_PERSON) / n_tok

            # lexical diversity and avg alphabetic token length
            alpha_tokens = [m.group(0) for t in tokens for m in [ALPHA_RE.search(t)] if m]
            uniq = len(set(t.lower() for t in alpha_tokens))
            tokc = max(1, len(alpha_tokens))
            lex_div = uniq / tokc
            avg_word_len = (sum(len(t) for t in alpha_tokens) / tokc) if tokc else 0.0

            rows.append([
                caps_ratio, exclam_rate, question_rate, punct_ratio, url_count,
                elongation_rate, allcaps_token_share, digit_ratio, repeated_punct_rate,
                elongation_count, lex_div, avg_word_len, second_person_ratio, mention_count
            ])

        return sp.csr_matrix(np.asarray(rows, dtype=np.float32))


def add_raw_features(texts_raw: list[str]) -> sp.csr_matrix:
    """Helper to compute raw features as a sparse matrix."""
    return RawTextFeats().fit_transform(texts_raw)


def add_profanity_ratio(norm_texts: Iterable[str], lexicon: set[str]) -> sp.csr_matrix:
    vals = []
    for t in norm_texts:
        toks = re.findall(r"[a-z]+", t or "")
        if not toks:
            vals.append([0.0]); continue
        hit = sum(1 for w in toks if w in lexicon)
        vals.append([hit / max(1, len(toks))])
    return sp.csr_matrix(np.asarray(vals), dtype=float)


def normalise(text: str) -> str:
    if not isinstance(text, str): return ""
    x = text.strip()
    x = HTML_RE.sub(" ", x)
    x = URL_RE.sub(" <URL> ", x)
    x = USER_RE.sub(" <USER> ", x)
    x = re.sub(r"(.)\1{2,}", r"\1\1", x)
    x = WS_RE.sub(" ", x)
    return x.lower()


def normalise_corpus(texts: Iterable[str]) -> List[str]:
    return [normalise(t) for t in texts]


class LemmaTokenizer:
    def __init__(self):
        self.use_spacy = False
        try:
            import spacy
            self._nlp = spacy.blank("en")
            self._nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
            self._nlp.initialize()
            self.use_spacy = True
        except Exception:
            from nltk.stem import SnowballStemmer
            self._stemmer = SnowballStemmer("english")

    def __call__(self, text: str):
        text = normalise(text)
        if self.use_spacy:
            doc = self._nlp(text)
            return [t.lemma_.lower() for t in doc if t.is_alpha]
        tokens = re.findall(r"[A-Za-z]+", text)
        return [self._stemmer.stem(t.lower()) for t in tokens]


# TF-IDF vectoriser
def make_word_tfidf(ngram_range=(1,2), min_df=3, 
                    max_features=None, use_lemmatizer=True):
    """
    Word-level TF-IDF.
    """
    kwargs = {
        "analyzer": "word",
        "ngram_range": ngram_range,
        "min_df": int(min_df),
        "max_features": None if not max_features else int(max_features),
        "strip_accents": "unicode",
        "sublinear_tf": True,
        "dtype": np.float32,
        "lowercase": False,
    }
    if use_lemmatizer:
        return TfidfVectorizer(
            tokenizer=LemmaTokenizer(),
            preprocessor=None,
            token_pattern=None,
            **kwargs,
        )
    else:
        return TfidfVectorizer(**kwargs)


def make_char_tfidf(ngram_range=(3, 4), min_df=3, max_features=None):
    """Char-level TF-IDF. """
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=int(min_df),
        max_features=None if not max_features else int(max_features),
        sublinear_tf=True,
        dtype=np.float32,
    )


# --- vectorisers / unions 
def build_feature_union(cfg: dict) -> FeatureUnion:
    """
    Build a FeatureUnion of word + char TF-IDF.
    """
    f = cfg["features"]
    wvec = make_word_tfidf(
        ngram_range=tuple(f["word_ngrams"]),
        min_df=f["min_df_word"],
        max_features=f.get("max_features_word"),
        use_lemmatizer=bool(f.get("use_lemmatizer", True)),
    )
    cvec = make_char_tfidf(
        ngram_range=tuple(f["char_ngrams"]),
        min_df=f["min_df_char"],
        max_features=f.get("max_features_char"),
    )
    return FeatureUnion([("word", wvec), ("char", cvec)], n_jobs=-1)
