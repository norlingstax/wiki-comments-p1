# src/utils.py
import logging, os, random, numpy as np


def set_seeds(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name="toxic"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    return logging.getLogger(name)
