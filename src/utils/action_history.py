import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ActionHistory:
    def __init__(
        self,
        history_path: str = "./models/history",
        max_entries_per_cat: int = 500,
        max_entry_age_days: int = 30,
        cleanup_interval_actions: int = 100,
    ):
        self.history_path = Path(history_path)
        self.history_path.mkdir(parents=True, exist_ok=True)
        self.max_entries_per_cat = max(1, max_entries_per_cat)
        self.max_entry_age_days = max_entry_age_days if max_entry_age_days > 0 else None
        self.cleanup_interval_actions = max(1, cleanup_interval_actions)
        self._writes_since_cleanup = 0
        self._lock = RLock()

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
            with self._lock:
                with open(cat_history_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                self._prune_cat_history(cat_history_file)
                self._writes_since_cleanup += 1
                if self._writes_since_cleanup >= self.cleanup_interval_actions:
                    self._cleanup_all_histories()
                    self._writes_since_cleanup = 0
        except Exception as e:
            logger.error("action_history_log_failed", cat_id=cat_id, error=str(e))

    def get_history(self, cat_id: str, limit: Optional[int] = None) -> list[dict]:
        cat_history_file = self.history_path / f"{cat_id}.jsonl"

        if not cat_history_file.exists():
            return []

        try:
            with self._lock:
                with open(cat_history_file, "r", encoding="utf-8") as f:
                    history = [json.loads(line.strip()) for line in f if line.strip()]
        except Exception as e:
            logger.error("action_history_read_failed", cat_id=cat_id, error=str(e))
            return []

        if limit:
            return history[-limit:]
        return history

    def clear_history(self, cat_id: str) -> None:
        cat_history_file = self.history_path / f"{cat_id}.jsonl"

        with self._lock:
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

    def _cleanup_all_histories(self) -> None:
        for history_file in self.history_path.glob("*.jsonl"):
            self._prune_cat_history(history_file)

    def _prune_cat_history(self, cat_history_file: Path) -> None:
        if not cat_history_file.exists():
            return

        try:
            with open(cat_history_file, "r", encoding="utf-8") as f:
                raw_lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(
                "action_history_prune_read_failed",
                file=str(cat_history_file),
                error=str(e),
            )
            return

        entries: list[dict] = []
        for raw_line in raw_lines:
            try:
                entry = json.loads(raw_line)
                if self._is_entry_fresh(entry):
                    entries.append(entry)
            except Exception:
                continue

        if len(entries) > self.max_entries_per_cat:
            entries = entries[-self.max_entries_per_cat :]

        new_lines = [json.dumps(entry) for entry in entries]
        if new_lines == raw_lines:
            return

        try:
            if not new_lines:
                cat_history_file.unlink(missing_ok=True)
                return

            with open(cat_history_file, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")
        except Exception as e:
            logger.error(
                "action_history_prune_write_failed",
                file=str(cat_history_file),
                error=str(e),
            )

    def _is_entry_fresh(self, entry: dict) -> bool:
        if self.max_entry_age_days is None:
            return True

        timestamp_raw = entry.get("timestamp")
        if not isinstance(timestamp_raw, str):
            return False

        try:
            timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
        except ValueError:
            return False

        cutoff = datetime.now(timestamp.tzinfo) - timedelta(days=self.max_entry_age_days)
        return timestamp >= cutoff
