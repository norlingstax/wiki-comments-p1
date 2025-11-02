# src/preprocess.py
import argparse

from .config import load_config
from .utils import get_logger, set_seeds
from .caching import prepare_cached_features
    
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger()
    set_seeds(cfg["seed"])

    bundle, sig, loaded = prepare_cached_features(cfg, log, build_if_missing=True)
    if loaded:
        log.info("Cached features already up to date.")
    else:
        log.info("Preprocessing complete. Features cached.")


if __name__ == "__main__":
    main()
