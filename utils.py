"""
Utility functions for T.O.M. CLI
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from config import DEFAULT_MODEL_MAX_CONTEXT

logger = logging.getLogger("tom_cli")


def ordinal(n: int) -> str:
    """Return the ordinal suffix string for a given day number."""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    else:
        return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


def load_model_config(model_path: Path) -> Dict[str, Any]:
    """Load model config.json and extract max_position_embeddings"""
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        logger.warning(f"config.json not found at {config_path}, using default context size")
        return {"max_position_embeddings": DEFAULT_MODEL_MAX_CONTEXT}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        max_pos = config.get("max_position_embeddings", DEFAULT_MODEL_MAX_CONTEXT)
        logger.info(f"Model max_position_embeddings: {max_pos:,}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config.json: {e}")
        return {"max_position_embeddings": DEFAULT_MODEL_MAX_CONTEXT}
