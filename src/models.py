# src/models.py
from pathlib import Path
import json
import logging
from sklearn.pipeline import Pipeline

from .features import build_feature_union


log = logging.getLogger(__name__)


def _best_params_path(model_name: str, cfg) -> Path:
    """
    Returns models/<model>/bayes_best.json by default.
    """
    return Path(cfg["paths"]["models"]) / model_name / "bayes_best.json"


def _load_best_params(model_name: str, cfg):
    """Load {"best_params": {...}} if it exists; else return None."""
    p = _best_params_path(model_name, cfg)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            params = data.get("best_params") or data
            if isinstance(params, dict):
                return params
        except Exception as e:
            log.warning("Could not read best params at %s: %s", p, e)
    return None


def _apply_supported_params(estimator, params: dict):
    """Set only the keys the estimator actually supports."""
    if not params:
        return
    supported = estimator.get_params(deep=False)
    clean = {k: v for k, v in params.items() if k in supported}
    if clean:
        estimator.set_params(**clean)


def make_estimator(cfg, name: str, cached: bool = False):
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            solver="liblinear",
            max_iter=10_000,
        )
            
    elif name == "lsvc":
        from sklearn.svm import LinearSVC
        clf = LinearSVC(
            max_iter=10_000
        )
        
    elif name == "pac":
        from sklearn.linear_model import PassiveAggressiveClassifier
        clf = PassiveAggressiveClassifier(
            max_iter=10_000,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    
    # if Bayes best params exist, apply them over the defaults
    best = _load_best_params(name, cfg)
    if best:
        _apply_supported_params(clf, best)
        log.info("Loaded tuned params for %s from %s", name, _best_params_path(name, cfg))
    
    if cached:
        # X will be sparse TF-IDF; return classifier only
        return clf
    else:
        # build + fit full pipeline on raw text
        vec = build_feature_union(cfg)
        return Pipeline([("vec", vec), ("clf", clf)])
