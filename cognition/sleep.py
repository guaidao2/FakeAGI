"""
睡眠巩固系统 — 离线经验重放 + 记忆巩固

生物学功能：
  1. 白天积累的经验在睡眠时重放
  2. 重要记忆被巩固，不重要记忆被清除
  3. 突触权重归一化（避免饱和）
  4. 情绪记忆被处理（降低应激）
"""

import numpy as np


class SleepCycle:
    def __init__(self):
        self.is_sleeping = False
        self.sleep_duration = 0
        self.target_sleep = 0
        self.dream_log = []
    
    def should_sleep(self, fatigue: float, circadian: float, energy: float = 1.0) -> bool:
        return fatigue > 0.7 and energy > 0.5  # 能量低时不睡，优先觅食
    
    def should_wake(self, fatigue: float, energy: float = 0) -> bool:
        return (fatigue < 0.2 and self.sleep_duration > 20) or (energy > 1.5 and fatigue < 0.5)
    
    def consolidate(self, replay_buffer: list) -> list:
        """睡眠巩固：重放经验并返回巩固后的记忆"""
        if not replay_buffer:
            return []
        
        n = min(32, len(replay_buffer))
        indices = np.random.choice(len(replay_buffer), n, replace=False)
        
        # 对采样的经验做"巩固"（模拟重放）
        consolidated = []
        for idx in indices:
            item = replay_buffer[idx].copy() if hasattr(replay_buffer[idx], 'copy') else replay_buffer[idx]
            consolidated.append(item)
        
        return consolidated
