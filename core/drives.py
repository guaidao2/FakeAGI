"""
驱动力系统 — 从身体状态产生多个行为驱动力

真正的生物体有多个独立的驱动力：
  饥饿驱动 → 觅食
  口渴驱动 → 找水  
  疲劳驱动 → 睡眠
  安全驱动 → 回避危险
  社交驱动 → 寻找同类
  好奇驱动 → 探索新环境（在安全时）
  无聊驱动 → 在环境太稳定时主动寻找变化

每个驱动力独立计算，GameNN 的策略选择在这些驱动力的
加权和上做决策。
"""

import numpy as np


class DriveSystem:
    def __init__(self):
        # 各驱动力 [0, 1] — 越大越迫切
        self.hunger = 0.0
        self.thirst = 0.0
        self.fatigue_drive = 0.0
        self.safety = 1.0
        self.curiosity = 0.3
        self.boredom = 0.0
        self.social = 0.0
        
        # 上次进食/饮水时间
        self.last_meal = 0
        self.last_drink = 0
        
        # 环境稳定计数器（用于无聊）
        self.stable_ticks = 0
        self.last_surprise_avg = 0.0
    
    def update(self, body_state: dict, survival_prob: float,
               surprise: float, tick: int, danger_nearby: bool = False,
               repeat_ticks: int = 0):
        """
        从身体状态和外部信号更新所有驱动力
        """
        # 饥饿驱动：能量越低越饿
        self.hunger = np.clip(1.0 - body_state["energy"] / 0.5, 0.0, 1.0)
        if body_state["energy"] < 0.1:
            self.hunger = 1.0  # 极度饥饿
        
        # 口渴驱动
        self.thirst = np.clip(1.0 - body_state["water"] / 0.4, 0.0, 1.0)
        if body_state["water"] < 0.1:
            self.thirst = 1.0
        
        # 疲劳驱动
        self.fatigue_drive = np.clip(body_state["fatigue"] * 0.8, 0.0, 1.0)
        
        # 安全驱动：危险近在咫尺
        if danger_nearby:
            self.safety = 0.0  # 极度不安全
        else:
            self.safety = min(1.0, self.safety + 0.01)
        
        # 好奇驱动：只要活着就保持基础好奇
        base_curiosity = 0.4
        if survival_prob > 0.2:
            base_curiosity += 0.3 * min(1.0, (survival_prob - 0.2) / 0.8)
        self.curiosity = np.clip(base_curiosity, 0.0, 1.0)
        
        # 无聊驱动：行为重复（最近动作重复度高——一直在撞墙/原地转）
        # 审计 B3 修复后的正确语义：原逻辑 surprise<0.05 才积累 stable_ticks——
        # surprise 真实流入后（0.5+ 常态）boredom 永不升 → 探索压制 → 适应崩。
        # 无聊是"行为重复度"不是"环境平静度"（撞墙 surprise 很高但同样无聊）
        self.repeat_ticks = max(0, int(repeat_ticks))
        self.boredom = np.clip(self.repeat_ticks / 200.0, 0.0, 1.0)
        
        # 社交驱动（如果有其他个体在附近时激活）
        # NOTE: 当前为单个体，保留为 0
        self.social = 0.0
    
    def get_dominance(self) -> str:
        drives = {
            "hunger": self.hunger,
            "thirst": self.thirst,
            "fatigue": self.fatigue_drive * 0.3,  # 大幅降权
            "fear": 1.0 - self.safety,
            "curiosity": self.curiosity,
            "boredom": self.boredom * 0.3,
        }
        return max(drives, key=drives.get)
    
    def get_action_bias(self) -> np.ndarray:
        """
        返回驱动力对各动作的偏置。
        格式：[up, left, right, down, sleep, explore_toward_random]
        """
        bias = np.zeros(6)
        if self.hunger > 0.5:
            bias[:4] += 0.3  # 饥饿 → 移动觅食
        if self.thirst > 0.5:
            bias[:4] += 0.3  # 口渴 → 移动找水（与饥饿对称——欲望架构阶段 A）
        if self.fatigue_drive > 0.7:
            bias[4] = 1.0  # 太累 → 睡眠
        if self.curiosity > 0.5 and self.safety > 0.5:
            bias[5] = self.curiosity  # 好奇 → 探索新方向
        if 1 - self.safety > 0.6:
            bias[5] = -1.0  # 恐惧 → 不要探索
            bias[:4] -= 0.3  # 可能后退
        return bias
    
    def get_state_vector(self) -> np.ndarray:
        return np.array([
            self.hunger,
            self.thirst,
            self.fatigue_drive,
            self.safety,
            self.curiosity,
            self.boredom,
        ])
