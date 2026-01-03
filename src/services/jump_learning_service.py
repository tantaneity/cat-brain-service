import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from src.utils.logger import get_logger

logger = get_logger(__name__)

JUMP_DATA_DIR = Path("data/jump_memory")
DEFAULT_JUMP_FORCE = 1.0
MAX_JUMP_FORCE = 2.5
MIN_JUMP_FORCE = 0.5
FORCE_INCREMENT = 0.15
LEARNING_RATE = 0.3


@dataclass
class JumpAttempt:
    target_id: str
    height_diff: float
    distance: float
    force_used: float
    success: bool


@dataclass
class JumpMemory:
    target_id: str
    learned_force: float
    success_count: int
    fail_count: int
    last_height_diff: float


class JumpLearningService:
    
    def __init__(self):
        self._memories: dict[str, dict[str, JumpMemory]] = {}
        JUMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_jump_force(self, cat_id: str, target_id: str, height_diff: float, distance: float) -> float:
        self._load_cat_memory(cat_id)
        
        memory = self._memories.get(cat_id, {}).get(target_id)
        
        if memory is not None:
            logger.info(
                "jump_force_from_memory",
                cat_id=cat_id,
                target_id=target_id,
                force=memory.learned_force,
                successes=memory.success_count
            )
            return memory.learned_force
        
        base_force = DEFAULT_JUMP_FORCE
        if height_diff > 0.5:
            base_force += (height_diff - 0.5) * 0.3
        
        logger.info(
            "jump_force_default",
            cat_id=cat_id,
            target_id=target_id,
            height_diff=height_diff,
            force=base_force
        )
        return min(base_force, MAX_JUMP_FORCE)
    
    def record_jump_result(
        self,
        cat_id: str,
        target_id: str,
        height_diff: float,
        distance: float,
        force_used: float,
        success: bool
    ) -> float:
        self._load_cat_memory(cat_id)
        
        if cat_id not in self._memories:
            self._memories[cat_id] = {}
        
        memory = self._memories[cat_id].get(target_id)
        
        if memory is None:
            memory = JumpMemory(
                target_id=target_id,
                learned_force=force_used,
                success_count=0,
                fail_count=0,
                last_height_diff=height_diff
            )
            self._memories[cat_id][target_id] = memory
        
        if success:
            memory.success_count += 1
            if memory.success_count > 3 and force_used < memory.learned_force:
                memory.learned_force = force_used + (memory.learned_force - force_used) * (1 - LEARNING_RATE)
        else:
            memory.fail_count += 1
            memory.learned_force = min(memory.learned_force + FORCE_INCREMENT, MAX_JUMP_FORCE)
        
        memory.last_height_diff = height_diff
        
        self._save_cat_memory(cat_id)
        
        logger.info(
            "jump_result_recorded",
            cat_id=cat_id,
            target_id=target_id,
            success=success,
            new_force=memory.learned_force,
            total_successes=memory.success_count,
            total_fails=memory.fail_count
        )
        
        return memory.learned_force
    
    def get_all_memories(self, cat_id: str) -> dict[str, dict]:
        self._load_cat_memory(cat_id)
        memories = self._memories.get(cat_id, {})
        return {k: asdict(v) for k, v in memories.items()}
    
    def reset_target_memory(self, cat_id: str, target_id: str) -> bool:
        self._load_cat_memory(cat_id)
        
        if cat_id in self._memories and target_id in self._memories[cat_id]:
            del self._memories[cat_id][target_id]
            self._save_cat_memory(cat_id)
            return True
        return False
    
    def _get_memory_path(self, cat_id: str) -> Path:
        return JUMP_DATA_DIR / f"{cat_id}_jumps.json"
    
    def _load_cat_memory(self, cat_id: str):
        if cat_id in self._memories:
            return
        
        path = self._get_memory_path(cat_id)
        if not path.exists():
            self._memories[cat_id] = {}
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self._memories[cat_id] = {
                k: JumpMemory(**v) for k, v in data.items()
            }
            logger.info("jump_memory_loaded", cat_id=cat_id, targets=len(self._memories[cat_id]))
        except Exception as e:
            logger.error("jump_memory_load_error", cat_id=cat_id, error=str(e))
            self._memories[cat_id] = {}
    
    def _save_cat_memory(self, cat_id: str):
        if cat_id not in self._memories:
            return
        
        path = self._get_memory_path(cat_id)
        data = {k: asdict(v) for k, v in self._memories[cat_id].items()}
        
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("jump_memory_save_error", cat_id=cat_id, error=str(e))
