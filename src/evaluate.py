# src/evaluate.py
import argparse
from pathlib import Path
import joblib
import numpy as np

from .config import load_config
from .utils import get_logger
from .metrics import get_scores, eval_classification, save_json, plot_confusion, plot_roc, plot_pr
from .data_io import processed_paths, load_processed_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="logreg",
                    choices=["baseline", "lsvc", "logreg", "pac"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger()
    log.info("Evaluating model '%s'…", args.model)

    # --- load saved model / bundle
    model_dir = Path(cfg["paths"]["models"]) / args.model
    if args.model != "baseline":
        model_path = model_dir / f"{args.model}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
        bundle = joblib.load(model_path)

    # --- get TEST split and vectorised X from cache
    proc_dir = Path(cfg["paths"]["processed"])
    prefix   = cfg.get("cache", {}).get("filename_prefix", "tfidf")
    p = processed_paths(proc_dir, prefix)
    if not p["manifest"].exists():
        raise FileNotFoundError(f"Processed cache not found at {proc_dir}. Run train first.")
    (Xtr, Xva, Xte, ytr, yva, yte, vec_cached, idx_tr, idx_va, idx_te, selector_cached) = load_processed_bundle(p)

    # --- predict + score on TEST
    if args.model == "baseline":
        # Build baseline model from lexicon and evaluate on raw test texts
        import pandas as pd
        from .dict_baseline import load_lexicon, DictionaryBaseline
    
        raw_df = pd.read_csv(cfg["paths"]["raw"])
        X_all_text = raw_df["text"].tolist()
        X_test_texts = [X_all_text[i] for i in idx_te]
    
        lex_path = cfg["paths"]["lexicon_path"]
        cues = load_lexicon(lex_path)
        clf = DictionaryBaseline.from_cues(cues)
    
        y_pred  = clf.predict(X_test_texts)
        y_score = clf.predict_proba(X_test_texts)[:, 1]
    else:
        clf = bundle["clf"]
    
        # Use cached matrix
        X = Xte
        y_pred  = clf.predict(X)
        y_score = get_scores(clf, X)

    res = eval_classification(
        y_true=yte,
        y_pred=y_pred,
        y_score=y_score,
        labels=("non-toxic", "toxic")
    )

    # --- write TEST results alongside model’s outputs/<model>/*
    base = Path(cfg["paths"]["outputs"]) / args.model
    metrics = base / "metrics"; figures = base / "figures"
    metrics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    save_json(res["report"], metrics / "test_classification_report.json")
    save_json({"confusion_matrix": res["confusion_matrix"]}, metrics / "test_confusion_matrix.json")
    save_json({"roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"]}, metrics / "test_auc.json")

    cm = np.array(res["confusion_matrix"])
    plot_confusion(cm, ("non-toxic","toxic"), figures / "test_confusion_matrix.png")
    plot_roc(res["fpr"], res["tpr"], res["roc_auc"], figures / "test_roc.png")
    plot_pr(res["precision_curve"], res["recall_curve"], res["pr_auc"], figures / "test_pr.png")

    log.info("Test ROC-AUC: %.3f | PR-AUC: %.3f | F1: %.3f",
             res["roc_auc"], res["pr_auc"], res["report"]["toxic"]["f1-score"])


if __name__ == "__main__":
    main()
