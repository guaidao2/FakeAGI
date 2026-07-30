"""
生长引擎 — 容量自适应

检测到 loss/surprise plateau → 扩展隐藏层维度
保留旧权重的基础上增加新容量
"""

import numpy as np


class GrowthMonitor:
    def __init__(self, window=5, min_drop=0.01):
        self.window = window
        self.min_drop = min_drop
        self.loss_history = []
        self.surprise_history = []
    
    def record(self, loss: float, surprise: float):
        self.loss_history.append(loss)
        self.surprise_history.append(surprise)
        if len(self.loss_history) > self.window * 3:
            self.loss_history.pop(0)
            self.surprise_history.pop(0)
    
    def should_grow(self) -> bool:
        if len(self.loss_history) < self.window * 2:
            return False
        recent = self.loss_history[-self.window:]
        earlier = self.loss_history[-self.window*2:-self.window]
        drop = np.mean(earlier) - np.mean(recent)
        return drop < self.min_drop  # loss 不降了，需要生长


class Expander:
    @staticmethod
    def grow_hidden(model, new_hidden: int, old_hidden: int):
        """扩展模型隐藏层维度"""
        if hasattr(model, 'grow'):
            model.grow(new_hidden)
        return model
