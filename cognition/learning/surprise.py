"""
惊奇计算器 — 世界模型的预测误差

惊奇 = 实际观测和预测的差距
高惊奇 → 需要更新模型 / 触发探索
"""

import numpy as np


class SurpriseComputer:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.running_avg = 0.0
    
    def compute(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        error = float(np.mean((predicted - actual) ** 2))
        surprise = min(1.0, error * 5.0)
        self.running_avg = (1 - self.alpha) * self.running_avg + self.alpha * surprise
        return surprise
    
    def get_running_avg(self) -> float:
        return self.running_avg
