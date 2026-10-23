"""Local environment checks. No dataset or model downloads are performed."""

import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import sys


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prepare_local_caches(root: Path) -> None:
    for name, relative in {
        "HF_HOME": ".cache/huggingface",
        "MPLCONFIGDIR": ".cache/matplotlib",
        "XDG_CACHE_HOME": ".cache",
        "JUPYTER_RUNTIME_DIR": ".cache/jupyter/runtime",
    }.items():
        os.environ.setdefault(name, str(root / relative))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def check_environment() -> dict:
    root = project_root()
    prepare_local_caches(root)
    import numpy as np
    import torch
    import yaml

    with (root / "configs/project.yaml").open() as handle:
        config = yaml.safe_load(handle)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    packages = {
        "numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow",
        "torch": "torch", "transformers": "transformers",
        "sentence-transformers": "sentence_transformers", "scikit-learn": "sklearn",
        "matplotlib": "matplotlib", "PyYAML": "yaml", "huggingface-hub": "huggingface_hub",
        "jupyterlab": "jupyterlab", "ipykernel": "ipykernel",
    }
    versions = {}
    for distribution, module in packages.items():
        importlib.import_module(module)
        versions[distribution] = importlib.metadata.version(distribution)

    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    device = "cuda" if cuda else "mps" if mps else "cpu"
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    result = (tensor @ tensor.T).cpu()
    expected = torch.tensor([[5.0, 11.0], [11.0, 25.0]])
    if not torch.allclose(result, expected):
        raise RuntimeError("Tensor arithmetic check failed")

    missing = [relative for relative in config["paths"].values() if not (root / relative).is_dir()]
    if missing:
        raise RuntimeError(f"Missing project directories: {missing}")
    return {
        "status": "passed", "project_root": str(root), "python": sys.version.split()[0],
        "python_executable": sys.executable, "platform": platform.platform(),
        "architecture": platform.machine(), "cpu_count": os.cpu_count(),
        "seed": seed, "device": device, "mps_available": mps, "cuda_available": cuda,
        "tensor_check": "passed", "packages": versions,
        "dataset_downloaded": False, "encoder_downloaded": False, "training_started": False,
        "note": "Fixed random seeds do not guarantee bit-identical results across hardware.",
    }
