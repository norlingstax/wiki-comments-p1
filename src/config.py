from pathlib import Path
import yaml

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # expand paths to Path objects
    for k, v in cfg.get("paths", {}).items():
        cfg["paths"][k] = Path(v)
    return cfg
