# src/data_io.py
from __future__ import annotations
from pathlib import Path
import json, hashlib, joblib
import numpy as np
import pandas as pd
from scipy import sparse as sp


def _json_dumps_canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def file_sig(path: Path) -> dict:
    """ 
    Returns a dictionary describing a file (size, mtime, md5), used inside the manifest. 
    """
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    stat = p.stat()
    return {
        "exists": True,
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
    }


def feature_sig(cfg: dict, raw_path: Path) -> dict:
    """
    Builds a feature signature from:
    - vectoriser settings (word/char n-grams, min_df, max_features, whether lemmatizer is used),
    - meta feature flags,
    - selection settings (enabled, k_best with chi2),
    - global seed,
    - raw file signature.
    """
    lex_path  = cfg["paths"].get("lexicon_path")
    sig = {
        "features": {
            "word_ngrams": list(cfg["features"]["word_ngrams"]),
            "char_ngrams": list(cfg["features"]["char_ngrams"]),
            "min_df_word": int(cfg["features"]["min_df_word"]),
            "min_df_char": int(cfg["features"]["min_df_char"]),
            "use_lemmatizer": bool(cfg["features"].get("use_lemmatizer", True)),
            "use_raw_feats": bool(cfg["features"].get("use_raw_feats", True)),
            "use_profanity_ratio": bool(cfg["features"].get("use_profanity_ratio", False)),
            "profanity_lexicon_sig": file_sig(lex_path) if lex_path else None,
        },
        "split": {
            "test_size": float(cfg["split"]["test_size"]),
            "valid_size": float(cfg["split"]["valid_size"]),
        },
        "selection": {
            "enabled": bool(cfg["selection"].get("enabled", False)),
            "k_best": int(cfg["selection"]["k_best"]),
        },
        "seed": int(cfg["seed"]),
        "raw_file": file_sig(raw_path),
        }
    sig["hash"] = hashlib.md5(_json_dumps_canonical(sig).encode("utf-8")).hexdigest()
    return sig


def processed_paths(processed_dir: Path, prefix: str = "tfidf") -> dict:
    """ 
    Standardises output file locations (X/y splits, vectorizer, selector, indices, manifest). 
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {
        "manifest":   processed_dir / "feature_manifest.json",
        "vectorizer": processed_dir / f"{prefix}_vectorizer.joblib",
        "X_train":    processed_dir / f"{prefix}_X_train.npz",
        "X_val":      processed_dir / f"{prefix}_X_val.npz",
        "X_test":     processed_dir / f"{prefix}_X_test.npz",
        "y_train":    processed_dir / f"{prefix}_y_train.npy",
        "y_val":      processed_dir / f"{prefix}_y_val.npy",
        "y_test":     processed_dir / f"{prefix}_y_test.npy",
        "idx_train":  processed_dir / f"{prefix}_idx_train.npy",
        "idx_val":    processed_dir / f"{prefix}_idx_val.npy",
        "idx_test":   processed_dir / f"{prefix}_idx_test.npy",
        "selector":   processed_dir / f"{prefix}_selector.joblib",

    }


def save_sparse(path: Path, X):
    path = Path(path)
    if sp.issparse(X):
        sp.save_npz(path, X)
    else:
        raise ValueError("Expected a scipy sparse matrix.")


def load_sparse(path: Path):
    return sp.load_npz(Path(path))


def save_numpy(path: Path, arr):
    path = Path(path)
    np.save(path, np.asarray(arr))


def load_numpy(path: Path):
    return np.load(Path(path), allow_pickle=False)


def save_manifest(path: Path, manifest: dict):
    Path(path).write_text(_json_dumps_canonical(manifest), encoding="utf-8")


def load_manifest(path: Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def manifest_matches(current: dict, saved: dict | None) -> bool:
    if not saved:
        return False
    return current.get("hash") == saved.get("hash")


def load_raw_dataframe(raw_path: Path) -> pd.DataFrame:
    return pd.read_csv(raw_path)


def save_processed_bundle(paths: dict, Xtr, Xva, Xte, ytr, yva, yte, vectorizer, manifest: dict,
                          idx_train, idx_val, idx_test, selector=None):
    """ Writes all artifacts to disk. """
    save_sparse(paths["X_train"], Xtr)
    save_sparse(paths["X_val"],   Xva)
    save_sparse(paths["X_test"],  Xte)
    save_numpy(paths["y_train"],  ytr)
    save_numpy(paths["y_val"],    yva)
    save_numpy(paths["y_test"],   yte)
    save_numpy(paths["idx_train"], idx_train)
    save_numpy(paths["idx_val"],   idx_val)
    save_numpy(paths["idx_test"],  idx_test)
    joblib.dump(vectorizer, paths["vectorizer"])
    save_manifest(paths["manifest"], manifest)
    if selector is not None:
        joblib.dump(selector, paths["selector"])


def load_processed_bundle(paths: dict):
    Xtr = load_sparse(paths["X_train"])
    Xva = load_sparse(paths["X_val"])
    Xte = load_sparse(paths["X_test"])
    ytr = load_numpy(paths["y_train"])
    yva = load_numpy(paths["y_val"])
    yte = load_numpy(paths["y_test"])
    idx_train = load_numpy(paths["idx_train"])
    idx_val   = load_numpy(paths["idx_val"])
    idx_test  = load_numpy(paths["idx_test"])
    vectorizer = joblib.load(paths["vectorizer"])
    selector = None
    if paths["selector"].exists():
        selector = joblib.load(paths["selector"])
    return Xtr, Xva, Xte, ytr, yva, yte, vectorizer, idx_train, idx_val, idx_test, selector
