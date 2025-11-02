# src/cashing.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from .dict_baseline import load_lexicon
from .features import (
    build_feature_union,
    normalise_corpus,
    add_raw_features,
    add_profanity_ratio,
)
from .data_io import (
    load_raw_dataframe, feature_sig, processed_paths, manifest_matches,
    save_processed_bundle, load_processed_bundle, load_manifest
)


def load_cached_features_or_none(cfg, log):
    """
    Computes the expected manifest path (processed_paths(...)["manifest"]).
    - If present, re-computes feature_sig(cfg, raw_path) and compares hashes.
    - If hashes match, loads all matrices and returns (bundle, sig).
    - If mismatch/missing, returns None.
    """
    raw_path = Path(cfg["paths"]["raw"])
    proc_dir = Path(cfg["paths"]["processed"])
    prefix   = cfg.get("cache", {}).get("filename_prefix", "tfidf")
    p = processed_paths(proc_dir, prefix)

    current_sig = feature_sig(cfg, raw_path)
    saved_sig   = load_manifest(p["manifest"])
    if cfg.get("cache", {}).get("enabled", True) and manifest_matches(current_sig, saved_sig):
        log.info("Loading cached features from %s", proc_dir)
        return load_processed_bundle(p), current_sig
    return None


def build_and_cache_features(cfg, log):
    """ 
    - Loads raw CSV(s) from paths in config.
    - Split into train/val/testusing seed and test/val sizes from config.
    - Normalise text.
    - Build vectoriser and fit on train, transform val/test.
    - Add raw/meta features and profanity ratio if enabled, by hstack-ing to the TF-IDF features.
    - Optional feature selection applied after vectorisation and feature additions.
    """
    raw_path = Path(cfg["paths"]["raw"])
    proc_dir = Path(cfg["paths"]["processed"])
    prefix   = cfg.get("cache", {}).get("filename_prefix", "tfidf")
    p = processed_paths(proc_dir, prefix)

    df = load_raw_dataframe(raw_path)
    X = df["text"].tolist()
    y = df["label"].astype(int).values
    n = len(X)
    idx = np.arange(n)

    # splits
    idx_tr_full, idx_te = train_test_split(idx, test_size=cfg["split"]["test_size"],
                                           stratify=y, random_state=cfg["seed"])
    y_tr_full = y[idx_tr_full]
    idx_tr, idx_va = train_test_split(idx_tr_full, test_size=cfg["split"]["valid_size"],
                                      stratify=y_tr_full, random_state=cfg["seed"])

    # slices
    Xtr_raw = [X[i] for i in idx_tr];   ytr = y[idx_tr]
    Xva_raw = [X[i] for i in idx_va];   yva = y[idx_va]
    Xte_raw = [X[i] for i in idx_te];   yte = y[idx_te]

    # normalise
    Xtr_n = normalise_corpus(Xtr_raw)
    Xva_n = normalise_corpus(Xva_raw)
    Xte_n = normalise_corpus(Xte_raw)

    # vectorise
    vec = build_feature_union(cfg)
    vec.fit(Xtr_n)
    Xtr = vec.transform(Xtr_n); Xva = vec.transform(Xva_n); Xte = vec.transform(Xte_n)

    # meta features
    if cfg["features"].get("use_raw_feats", True):
        Ftr, Fva, Fte = add_raw_features(Xtr_raw), add_raw_features(Xva_raw), add_raw_features(Xte_raw)
        Xtr = sp.hstack([Xtr, Ftr], format="csr"); Xva = sp.hstack([Xva, Fva], format="csr"); Xte = sp.hstack([Xte, Fte], format="csr")

    if cfg["features"].get("use_profanity_ratio", False):
        lex_path = cfg["paths"]["lexicon_path"]
        lex_set = set(load_lexicon(lex_path))
        Pr_tr = add_profanity_ratio(Xtr_n, lex_set); Pr_va = add_profanity_ratio(Xva_n, lex_set); Pr_te = add_profanity_ratio(Xte_n, lex_set)
        Xtr = sp.hstack([Xtr, Pr_tr], format="csr"); Xva = sp.hstack([Xva, Pr_va], format="csr"); Xte = sp.hstack([Xte, Pr_te], format="csr")

    # optional selection
    selector = None
    sel_cfg = cfg.get("selection", {})
    if sel_cfg.get("enabled", False):
        k = int(sel_cfg.get("k_best", 150_000))
        selector = SelectKBest(score_func=chi2, k=k).fit(Xtr, ytr)
        Xtr = selector.transform(Xtr); Xva = selector.transform(Xva); Xte = selector.transform(Xte)

    sig = feature_sig(cfg, raw_path)
    save_processed_bundle(p, Xtr, Xva, Xte, ytr, yva, yte, vec, sig,
                          idx_train=idx_tr, idx_val=idx_va, idx_test=idx_te, selector=selector)
    return (Xtr, Xva, Xte, ytr, yva, yte, vec, selector, idx_tr, idx_va, idx_te), sig


def prepare_cached_features(cfg, log, *, build_if_missing: bool):
    cached = load_cached_features_or_none(cfg, log)
    if cached is not None:
        bundle, sig = cached
        return bundle, sig, True
    if build_if_missing:
        log.info("Cached features not found or outdated - building now…")
        bundle, sig = build_and_cache_features(cfg, log)
        return bundle, sig, False
    return None, None, False