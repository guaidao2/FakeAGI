"""
好奇心调度器 — 元认知层

何时探索 vs 利用的元控制：
  1. 生存安全时 → 探索预算高
  2. 生存危险时 → 探索预算趋近于零
  3. 知识缺口大 → 探索优先级提高
  4. 当探索目标超过 duration 没达成 → 放弃，标记知识缺口为"当前不可解"
"""

import numpy as np


class CuriosityScheduler:
    """控制何时探索 vs 何时利用"""
    def __init__(self):
        self.exploration_budget = 0.3   # 0~1
        self.budget_decay = 0.999
        self.curiosity_boost = 0.1
        self.give_up_threshold = 120    # 一个探索目标持续 tick 数后放弃
        self.give_up_counter = {}
        self.total_exploration_ticks = 0
        self.total_exploit_ticks = 0
    
    def update(self, health: float, gap_exists: bool,
               goal_active: bool, confidence: float):
        """更新探索预算"""
        # 基础：安全时增加，危险时减少
        safety_ratio = min(1.0, health * 1.5)
        
        # 知识缺口驱动
        gap_boost = 0.15 if gap_exists else 0.0
        
        # 学习阶段：低置信度需要更多探索
        learning_boost = max(0, (0.3 - confidence)) * 0.5
        
        # 目标追踪中
        goal_boost = 0.05 if goal_active else 0.0
        
        # 计算预算
        self.exploration_budget = (
            self.exploration_budget * self.budget_decay
            + safety_ratio * 0.01
            + gap_boost * 0.02
            + learning_boost * 0.03
            + goal_boost * 0.005
        )
        self.exploration_budget = np.clip(self.exploration_budget, 0.0, 0.8)
    
    def should_explore(self) -> bool:
        """当前 tick 是否应该探索"""
        explore = np.random.random() < self.exploration_budget
        if explore:
            self.total_exploration_ticks += 1
        else:
            self.total_exploit_ticks += 1
        return explore
    
    def should_give_up(self, goal_key: str) -> bool:
        """是否应该放弃当前探索目标"""
        self.give_up_counter[goal_key] = self.give_up_counter.get(goal_key, 0) + 1
        return self.give_up_counter[goal_key] > self.give_up_threshold
    
    def get_balance(self) -> float:
        """返回当前探索/利用平衡值（0=全利用, 1=全探索）"""
        return self.exploration_budget
