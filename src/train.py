# src/train.py
import argparse
from pathlib import Path
import joblib
import pandas as pd

# ---- NumPy/scikit-optimize compatibility fix
import numpy as _np 
if not hasattr(_np, "int"):
    _np.int = int
if not hasattr(_np, "float"):
    _np.float = float

from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from sklearn.model_selection import StratifiedKFold
import multiprocessing

from .config import load_config
from .utils import set_seeds, get_logger
from .models import make_estimator
from .metrics import get_scores, eval_classification, save_json, plot_confusion, plot_roc, plot_pr
from .caching import prepare_cached_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="logreg",
                    choices=["logreg", "lsvc", "pac"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger()
    set_seeds(cfg["seed"])

    bundle, sig, loaded = prepare_cached_features(cfg, log, build_if_missing=False)
    if bundle is None:
        log.error("No cached features found. Please run preprocessing first:\n"
                  "  python -m src.preprocess --config %s", args.config)
        raise SystemExit(1)

    (Xtr, Xva, Xte, ytr, yva, yte, vec, selector, idx_tr, idx_va, idx_te) = bundle
    log.info("Feature shapes -- train: %s, val: %s, test: %s", Xtr.shape, Xva.shape, Xte.shape)
    
    # Output dirs
    outputs_base = Path(cfg["paths"]["outputs"]) / args.model
    model_dir = Path(cfg["paths"]["models"]) / args.model
    metrics_dir  = outputs_base / "metrics"
    figures_dir = outputs_base / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # --- choose/train model
    base = make_estimator(cfg, args.model, cached=True)
    if hasattr(base, "random_state"):
        base.set_params(random_state=cfg["seed"])

    tune_cfg = cfg.get("tuning", {})
    use_bayes = bool(tune_cfg.get("enabled", False)) and tune_cfg.get("method", "bayes") == "bayes"
    
    best_path = model_dir / "bayes_best.json"
    skip_if_saved = bool(tune_cfg.get("skip_if_saved", True))

    if use_bayes and skip_if_saved and best_path.exists():
        log.info("Found existing tuned params at %s; skipping BayesSearchCV.", best_path)
        use_bayes = False
    
    # seeded, shuffled CV for reproducibility and better folds
    cv_n = int(tune_cfg.get("cv", 3))
    cv_obj = StratifiedKFold(n_splits=cv_n, shuffle=True, random_state=cfg["seed"])
    
    # evaluate multiple candidates per Bayesian iteration in parallel
    default_points = min(multiprocessing.cpu_count(), 8)
    n_points = int(tune_cfg.get("n_points", default_points))

    if use_bayes:
        space = _bayes_space(args.model)
        if space is None:
            # Fallback: no space defined for this model, use base directly
            model = base
            log.info("No Bayes space for model '%s'; fitting base estimator.", args.model)
        else:
            log.info("Starting BayesSearchCV for model '%s'…", args.model)
            model = BayesSearchCV(
                estimator=base,
                search_spaces=space,
                n_iter=int(tune_cfg.get("n_iter", 25)),
                cv=cv_obj, 
                n_jobs=int(tune_cfg.get("n_jobs", -1)),
                n_points=n_points,  
                scoring=tune_cfg.get("scoring", "f1"),
                random_state=cfg["seed"],
                refit=True,
                verbose=1
            )
    else:
        model = base

    log.info("Fitting model on cached features…")
    model.fit(Xtr, ytr)

    # If tuned, log + persist best params/score and the CV results
    if isinstance(model, BayesSearchCV):
        best = {
            "best_score_cv": float(model.best_score_),
            "best_params": model.best_params_
        }
        log.info("BayesSearchCV best_score_: %.3f (cv=%s)", model.best_score_, tune_cfg.get("cv", 3))
        log.info("BayesSearchCV best_params_: %s", model.best_params_)
        save_json(best, model_dir / "bayes_best.json")
        try:
            pd.DataFrame(model.cv_results_).to_csv(model_dir / "bayes_cv_results.csv", index=False)
        except Exception as e:
            log.warning("Could not save BayesSearchCV cv_results_: %s", e)

    # --- evaluate on validation set
    y_val_pred  = model.predict(Xva)
    y_val_score = get_scores(model, Xva)

    results = eval_classification(
        y_true=yva,
        y_pred=y_val_pred,
        y_score=y_val_score,
        labels=("non-toxic", "toxic")
    )

    # --- save VAL metrics/figures
    save_json(results["report"], metrics_dir / "val_classification_report.json")
    save_json({"confusion_matrix": results["confusion_matrix"]}, metrics_dir / "val_confusion_matrix.json")
    save_json({"roc_auc": results["roc_auc"], "pr_auc": results["pr_auc"]}, metrics_dir / "val_auc.json")

    import numpy as np
    cm = np.array(results["confusion_matrix"])
    plot_confusion(cm, labels=("non-toxic","toxic"), out_path=figures_dir / "val_confusion_matrix.png")
    plot_roc(results["fpr"], results["tpr"], results["roc_auc"], out_path=figures_dir / "val_roc.png")
    plot_pr(results["precision_curve"], results["recall_curve"], results["pr_auc"], out_path=figures_dir / "val_pr.png")

    log.info("Val ROC-AUC: %.3f | PR-AUC: %.3f | F1: %.3f",
             results["roc_auc"], results["pr_auc"], results["report"]["toxic"]["f1-score"])

    # --- save model bundle
    bundle = {"vectorizer": vec, "selector": selector, "clf": model}
    joblib.dump(bundle, model_dir / f"{args.model}.joblib")
    log.info("Saved model bundle to %s", model_dir / f"{args.model}.joblib")


def _bayes_space(model_name: str):
    """
    Return a skopt space dict for the given model key from --model.
    Spaces are chosen to be safe with sparse TF-IDF and common defaults.
    """

    # LogisticRegression
    if model_name == "logreg":
        return {
            # Regularisation strength (log scale)
            "C": Real(1e-3, 1e1, prior='log-uniform'),
            # l1 can help with very high-dimensional sparsity; l2 is default
            "penalty": Categorical(["l1", "l2"]),
            # Reweight minority class if needed
            "class_weight": Categorical([None, "balanced"]),
            # Tolerance can matter on noisy text
            "tol": Real(1e-6, 1e-2, prior="log-uniform"),
        }
    
    # LinearSVC
    if model_name == "lsvc":
        return {
            # Regularisation strength (log scale)
            "C": Real(1e-3, 1e1, prior='log-uniform'),
            # Loss variants
            "loss": Categorical(["hinge", "squared_hinge"]),
            # Reweight minority class if needed
            "class_weight": Categorical([None, "balanced"]),
            # Tolerance can matter on noisy text
            "tol": Real(1e-6, 1e-2, prior="log-uniform"),
        }

    # PassiveAggressiveClassifier
    if model_name == "pac":
        return {
            # Regularisation strength (log scale)
            "C": Real(1e-3, 1e1, prior="log-uniform"),
            # Hinge vs squared_hinge (smother)
            "loss": Categorical(["hinge", "squared_hinge"]),
            # Averaged perceptron variant
            "average": Categorical([False, True, 10, 50, 100]),
            # Convergence tolerance
            "tol": Real(1e-5, 1e-2, prior="log-uniform"),
        }
    return None


if __name__ == "__main__":
    main()
