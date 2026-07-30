"""
危险感知系统 — 威胁检测 + 回避行为

检测环境中的威胁信号并生成回避驱动力。
"""

import numpy as np


class DangerSystem:
    def __init__(self):
        self.threat_level = 0.0
        self.last_threat_tick = -100
        self.threat_memory = {}  # (x, y) → threat_count
    
    def sense(self, obs: np.ndarray, tick: int) -> float:
        """感知当前威胁，返回威胁等级 [0, 1]"""
        # obs 中如果有负值（危险信号），检测
        danger_signals = obs[obs < -0.5]
        if len(danger_signals) > 0:
            self.threat_level = min(1.0, self.threat_level + 0.3)
            self.last_threat_tick = tick
        else:
            self.threat_level = max(0.0, self.threat_level - 0.01)
        return self.threat_level
    
    def is_threat_nearby(self) -> bool:
        return self.threat_level > 0.3
