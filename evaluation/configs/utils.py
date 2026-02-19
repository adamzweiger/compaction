# evaluation/configs/utils.py
"""Utility functions for loading configuration files."""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any

from compaction.query_generation import QueryConfig


def _load_module(config_path: Path):
    """Load a Python module from a file path."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    sys.modules['config_module'] = config_module
    spec.loader.exec_module(config_module)
    return config_module


def load_query_config(config_path: str = 'repeat') -> QueryConfig:
    """
    Load query generation configuration.

    Args:
        config_path: Name of the query config file (e.g., 'repeat', 'self-study', 'random-vectors').
                     The '.py' extension and 'evaluation/configs/query_generation/' directory
                     are added automatically.

    Returns:
        QueryConfig object.
    """
    if not config_path.startswith('evaluation/configs/'):
        if not config_path.endswith('.py'):
            config_path = f"{config_path}.py"
        config_path = f"evaluation/configs/query_generation/{config_path}"

    config_module = _load_module(Path(config_path))

    if not hasattr(config_module, 'config'):
        raise AttributeError(
            f"Config file {config_path} must define a 'config' variable"
        )

    return config_module.config


def load_algorithm_config(config_path: str = 'fast', target_size: int = None) -> Dict[str, Dict[str, Any]]:
    """
    Load algorithm hyperparameter configuration.

    Args:
        config_path: Name of the algorithm config file (e.g., 'fast', 'best').
                     The '.py' extension and 'evaluation/configs/algorithms/' directory
                     are added automatically.
        target_size: Optional target size to pass to config files that use it.

    Returns:
        Dictionary mapping algorithm names to their hyperparameter dictionaries.
    """
    if not config_path.startswith('evaluation/configs/'):
        if not config_path.endswith('.py'):
            config_path = f"{config_path}.py"
        config_path = f"evaluation/configs/algorithms/{config_path}"

    config_module = _load_module(Path(config_path))

    # Check if config is a callable (function) that takes target_size
    if hasattr(config_module, 'get_config') and callable(config_module.get_config):
        return config_module.get_config(target_size=target_size)

    if not hasattr(config_module, 'config'):
        raise AttributeError(
            f"Config file {config_path} must define a 'config' variable or 'get_config' function"
        )

    return config_module.config


def get_method_kwargs(config: Dict[str, Dict[str, Any]], method_name: str) -> Dict[str, Any]:
    """
    Get hyperparameters for a specific method from the config.

    Args:
        config: Configuration dictionary from load_algorithm_config()
        method_name: Name of the algorithm/method

    Returns:
        Dictionary of hyperparameters for the method (empty dict if not found)
    """
    return config.get(method_name, {})
