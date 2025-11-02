# src/interpret.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from .config import load_config
from .utils import get_logger, set_seeds
from .train import prepare_cached_features

def get_scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        return s if getattr(s, "ndim", 1) == 1 else s[:, 0]
    return clf.predict(X)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="logreg",
                    choices=["logreg", "lsvc", "pac"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger()
    set_seeds(cfg["seed"])

    # --- load cached matrices and indices
    (Xtr, Xva, Xte,
     ytr, yva, yte,
     vec_cache, selector_cache,
     idx_tr, idx_va, idx_te), sig, loaded = prepare_cached_features(cfg, log, build_if_missing=False)
    log.info("Loaded cached features. VAL shape: %s", Xva.shape)

    # --- outputs
    outputs_base = Path(cfg["paths"]["outputs"]) / args.model
    errors_dir = outputs_base / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)

    thr = float(args.threshold)
    
    # --- compute scores/preds on VAL
    model_dir  = Path(cfg["paths"]["models"]) / args.model
    model_path = model_dir / f"{args.model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    bundle = joblib.load(model_path)
    clf    = bundle["clf"]

    X = Xva
    n = X.shape[0]
    y_score = np.asarray(get_scores(clf, X)).reshape(-1)
    y_pred  = (y_score >= thr).astype(int).reshape(-1)
    y_true  = np.asarray(yva).reshape(-1)[:n]
    idxs    = np.asarray(idx_va).reshape(-1)[:n]

    # raw texts 
    raw_df    = pd.read_csv(cfg["paths"]["raw"])
    texts_all = raw_df["text"].tolist()

    # --- assemble VAL dataframe
    df_val = pd.DataFrame({
        "y_true": y_true.astype(int),
        "y_pred": y_pred.astype(int),
        "score":  y_score.astype(float),
        "idx":    idxs,
    })

    # --- pick K false positives / false negatives
    fp = df_val[(df_val.y_true == 0) & (df_val.y_pred == 1)].copy()
    fn = df_val[(df_val.y_true == 1) & (df_val.y_pred == 0)].copy()

    fp["conf"] = (fp["score"] - thr).abs()
    fn["conf"] = (fn["score"] - thr).abs()

    fp_top = fp.sort_values("conf", ascending=False).head(args.k).drop(columns=["conf"])
    fn_top = fn.sort_values("conf", ascending=False).head(args.k).drop(columns=["conf"])

    # add text 
    if 'texts_all' not in locals():
        raw_df    = pd.read_csv(cfg["paths"]["raw"])
        texts_all = raw_df["text"].tolist()
    fp_top["text"] = [texts_all[int(i)] for i in fp_top["idx"].tolist()]
    fn_top["text"] = [texts_all[int(i)] for i in fn_top["idx"].tolist()]

    # --- save CSVs
    fp_path = errors_dir / "val_false_positives.csv"
    fn_path = errors_dir / "val_false_negatives.csv"
    fp_top.to_csv(fp_path, index=False)
    fn_top.to_csv(fn_path, index=False)

    log.info("Saved %d FP examples to %s", len(fp_top), fp_path)
    log.info("Saved %d FN examples to %s", len(fn_top), fn_path)

    # --- print a few lines for report copy/paste
    def _show(df, title):
        print(f"\n=== {title} ===")
        for _, row in df.iterrows():
            print(f"[idx={row['idx']}] true={row['y_true']} pred={row['y_pred']} score={row['score']:.3f}")
            txt = str(row["text"]).replace("\n", " ")
            print("   ", (txt[:240] + ('…' if len(txt) > 240 else '')))
    _show(fp_top, f"VAL False Positives (top {args.k}) - {args.model}")
    _show(fn_top, f"VAL False Negatives (top {args.k}) - {args.model}")


if __name__ == "__main__":
    main()