"""
好奇心预算分配器 — 决定系统何时探索、探索什么

原则：
  存在概率高 → 好奇心预算多 → 探索未知
  存在概率低 → 好奇心预算少 → 专注已知的生存策略
  惊奇信号 → 好奇心消耗 → 学习/记忆
"""

import numpy as np


class CuriosityManager:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.budget = 0.3
        self.curiosity_map = {}  # "探索对象" → 探索次数
        self.exploration_count = 0
        self.total_explorations = 0
    
    def update_budget(self, survival_prob: float):
        """根据存在概率更新好奇心预算"""
        if survival_prob > 0.7:
            self.budget = min(0.5, self.budget + 0.01)
        elif survival_prob > 0.3:
            self.budget = 0.2
        else:
            self.budget = max(0.0, self.budget - 0.02)
    
    def should_explore(self, surprise: float) -> bool:
        """当前 tick 是否应该探索"""
        return surprise > (1.0 - self.budget)
    
    def record_exploration(self, context_key: str):
        """记录一次探索行为"""
        self.curiosity_map[context_key] = self.curiosity_map.get(context_key, 0) + 1
        self.exploration_count += 1
        self.total_explorations += 1
    
    def get_boredom(self, context_key: str) -> float:
        """对某个上下文的厌倦程度（探索过多次后降低好奇心）"""
        count = self.curiosity_map.get(context_key, 0)
        return min(1.0, count / 10.0)
