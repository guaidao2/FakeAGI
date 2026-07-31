"""
目标层（Goal Layer）— 目标 vs 过程的分离

核心命题（DESIGN_GOALS.md）：
  目标 = 持久状态期望（由身体结构内生，不随观测消失）
  过程 = 达成目标的手段（可更换）
  落差 = 目标态 − 当前态（显式内部变量，驱动信号）
  信息寻求 = 落差高 + 无线索 → 定向搜索（非随机游走）

公理④ 精炼：落差是内源驱动主信号，激活以目标表征存在为前提。
"""

import numpy as np


class Goal:
    """单一目标：目标态 + 当前态读取 + 持久性"""
    def __init__(self, name: str, target_value: float, current_fn,
                 weight: float = 1.0, persistent: bool = True):
        self.name = name
        self.target_value = target_value
        self.current_fn = current_fn      # 返回当前态标量
        self.weight = weight
        self.persistent = persistent
        self.achieved = False
        self.gap_history = []

    def update(self) -> float:
        """计算当前落差（目标态 − 当前态），返回 gap"""
        current = self.current_fn()
        gap = max(0.0, self.target_value - current)
        self.achieved = gap <= 0.01
        self.gap_history.append(gap)
        if len(self.gap_history) > 200:
            self.gap_history.pop(0)
        return gap

    def urgency(self) -> float:
        """紧迫度 = 落差 × 权重（归一化 0-1）"""
        gap = self.update()
        return float(np.clip(gap * self.weight, 0.0, 1.0))


class GoalState:
    """目标池 + 主导目标 + 落差 + 信息寻求动机"""

    def __init__(self):
        self.goals = {}          # name -> Goal
        self.active_goal = None
        self.gap = 0.0
        self.exploration_intent = 0.0   # 信息寻求动机（0-1）
        self.achieved_history = []      # 落差归零事件
        self.signal_threshold = 0.3     # 线索存在阈值（低于则视为"无线索"）

    def register(self, goal: Goal):
        self.goals[goal.name] = goal

    def update(self, has_resource_signal: bool = False) -> dict:
        """每 tick 更新：落差 + 信息寻求动机
        has_resource_signal: 观测中是否有资源线索（食物方向等）
        """
        # 1. 更新所有目标，选最紧迫的
        best_name, best_urgency = None, 0.0
        for name, g in self.goals.items():
            u = g.urgency()
            if u > best_urgency:
                best_name, best_urgency = name, u
        self.active_goal = best_name
        self.gap = best_urgency

        # 2. 信息寻求动机：落差高 + 无线索 → 探索
        if self.gap > 0.1 and not has_resource_signal:
            self.exploration_intent = min(1.0, self.gap)
        else:
            self.exploration_intent = max(0.0, self.exploration_intent - 0.05)

        # 3. 落差归零检测
        g = self.goals.get(self.active_goal) if self.active_goal else None
        if g is not None and g.achieved:
            self.achieved_history.append(self.active_goal)
            if len(self.achieved_history) > 50:
                self.achieved_history.pop(0)

        return {
            "active_goal": self.active_goal,
            "gap": self.gap,
            "exploration_intent": self.exploration_intent,
            "achieved": g.achieved if g else False,
        }

    def get_state_vector(self) -> np.ndarray:
        """供认知核心拼接的向量（3 维：主导目标紧迫度/信息寻求/达成率）"""
        achieved_rate = len(self.achieved_history) / 50.0 if self.achieved_history else 0.0
        return np.array([self.gap, self.exploration_intent, achieved_rate],
                        dtype=np.float32)

    def get_state(self) -> dict:
        return {
            "active_goal": self.active_goal,
            "gap": round(self.gap, 3),
            "exploration_intent": round(self.exploration_intent, 3),
            "goals": {k: round(v.urgency(), 3) for k, v in self.goals.items()},
        }
