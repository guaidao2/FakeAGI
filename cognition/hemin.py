"""
他者模型 (Other Model) — 跟踪"另一个智能体"的行为

原理⑦前置条件：自模型的训练信号之一来自"他者模型"。
系统先建模"另一个 agent 在做什么"，
再通过对比产生"我在做什么"。

实现：
  - 维护一个"他者"轨迹：总是选择与"我"不同的动作
  - 通过对比真实轨迹和反事实轨迹，产生自我定位信号
"""

import numpy as np


class OtherModel:
    def __init__(self, window=50):
        self.window = window
        self.action_history = []        # (action, pos, drive) — 真实历史的"我"
        self.other_trajectory = []      # 反事实的"他者"轨迹
        self.self_other_divergence = 0.0
        self.identity_vector = np.zeros(8)
        
    def record_self_action(self, action: int, pos: tuple, drive: str):
        """记录自身行为，同时生成他者反事实"""
        self.action_history.append({
            "action": action,
            "pos": pos,
            "drive": drive,
        })
        if len(self.action_history) > self.window:
            self.action_history.pop(0)
        
        # 他者选择不同动作（反事实）
        other_actions = [0, 1, 2, 3, 4]
        other_actions.remove(action) if action in other_actions else None
        other_act = np.random.choice(other_actions) if other_actions else 0
        self.other_trajectory.append({
            "action": other_act,
            "pos": pos,
            "drive": drive,
        })
        if len(self.other_trajectory) > self.window:
            self.other_trajectory.pop(0)
    
    def update(self) -> float:
        """计算自我-他者差异度"""
        if len(self.action_history) < 10:
            return 0.0
        
        # 我 vs 他者：行为分布差异
        my_actions = [a["action"] for a in self.action_history[-20:]]
        other_actions = [a["action"] for a in self.other_trajectory[-20:]]
        
        my_hist = np.bincount(my_actions, minlength=5)[:5]
        other_hist = np.bincount(other_actions, minlength=5)[:5]
        my_hist = my_hist / max(1, my_hist.sum())
        other_hist = other_hist / max(1, other_hist.sum())
        
        self.self_other_divergence = float(np.mean(np.abs(my_hist - other_hist)))
        
        # 身份向量
        self.identity_vector = np.concatenate([
            [self.self_other_divergence],
            my_hist[:4],
            other_hist[:3],
        ])
        
        return self.self_other_divergence
    
    def get_divergence(self) -> float:
        return self.self_other_divergence
    
    def get_identity_vector(self) -> np.ndarray:
        return self.identity_vector
