import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ActionHistory:

    
    def __init__(self, history_path: str = "./models/history"):
        self.history_path = Path(history_path)
        self.history_path.mkdir(parents=True, exist_ok=True)
    
    def log_action(
        self,
        cat_id: str,
        observation: np.ndarray,
        action: int,
        reward: Optional[float] = None,
    ) -> None:

        cat_history_file = self.history_path / f"{cat_id}.jsonl"
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "observation": observation.tolist(),
            "action": action,
            "reward": reward,
        }
        
        try:
            with open(cat_history_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("action_history_log_failed", cat_id=cat_id, error=str(e))
    
    def get_history(self, cat_id: str, limit: Optional[int] = None) -> list[dict]:

        cat_history_file = self.history_path / f"{cat_id}.jsonl"
        
        if not cat_history_file.exists():
            return []
        
        history = []
        try:
            with open(cat_history_file, "r") as f:
                for line in f:
                    history.append(json.loads(line.strip()))
        except Exception as e:
            logger.error("action_history_read_failed", cat_id=cat_id, error=str(e))
            return []
        
        if limit:
            return history[-limit:]
        return history
    
    def clear_history(self, cat_id: str) -> None:

        cat_history_file = self.history_path / f"{cat_id}.jsonl"
        
        if cat_history_file.exists():
            cat_history_file.unlink()
            logger.info("action_history_cleared", cat_id=cat_id)
    
    def get_history_stats(self, cat_id: str) -> dict:

        history = self.get_history(cat_id)
        
        if not history:
            return {
                "cat_id": cat_id,
                "total_actions": 0,
                "first_action": None,
                "last_action": None,
            }
        
        return {
            "cat_id": cat_id,
            "total_actions": len(history),
            "first_action": history[0]["timestamp"],
            "last_action": history[-1]["timestamp"],
        }
