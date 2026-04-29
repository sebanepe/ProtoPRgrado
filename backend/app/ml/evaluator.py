import os
import joblib
from typing import Dict, Any, List, Tuple


def discover_models(models_dir: str) -> List[Tuple[str, str, str]]:
    """Return list of (model_name, version, path) for files in models_dir."""
    out = []
    if not os.path.isdir(models_dir):
        return out
    for fn in os.listdir(models_dir):
        if not fn.endswith(".pkl"):
            continue
        path = os.path.join(models_dir, fn)
        # expect pattern name_version.pkl
        name_part = fn[:-4]
        if "_" in name_part:
            name, version = name_part.rsplit("_", 1)
        else:
            name, version = name_part, "unknown"
        out.append((name, version, path))
    return out


def load_model(path: str):
    return joblib.load(path)
