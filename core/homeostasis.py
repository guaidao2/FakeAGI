"""
稳态维持器 — 监控关键变量，确保系统不偏离可生存区间

对应生物学的：下丘脑、脑干、自主神经系统
"""

import numpy as np


class Homeostasis:
    def __init__(self):
        # 关键变量的安全区间
        self.bounds = {
            "energy": (0.1, 1.0),
            "integrity": (0.3, 1.0),
            "surprise_rate": (0.0, 0.8),
            "survival_prob": (0.1, 1.0),
        }
        self.alarms = []
    
    def check(self, state: dict) -> list:
        """检查所有关键变量，返回告警列表"""
        self.alarms = []
        for key, (lo, hi) in self.bounds.items():
            val = state.get(key, 0.5)
            if val < lo:
                self.alarms.append({"variable": key, "value": val, 
                                    "severity": "critical", "direction": "low"})
            elif val > hi:
                self.alarms.append({"variable": key, "value": val,
                                    "severity": "warning", "direction": "high"})
        return self.alarms
    
    def in_danger(self) -> bool:
        return any(a["severity"] == "critical" for a in self.alarms)
