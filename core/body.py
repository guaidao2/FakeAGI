"""
身体模拟层 — 多维度稳态变量 + 生理节律

真正的生物体不是由一个标量"能量"定义的。
它包括多个相互关联的稳态变量：
  能量（血糖/脂肪储备）
  水分（水合状态）
  结构（细胞/组织完整性）
  温度（体温）
  社交（群居需求）
  疲劳（睡眠压力）
  压力（慢性应激水平）

这些变量之间存在耦合关系：
  能量↓ → 体温↓（节能模式）
  水分↓ → 能量↓（代谢效率下降）
  疲劳↑ → 探索↓（节省能量）
  压力↑ → 结构↓（长期应激损伤）
"""

import numpy as np


class BodyModel:
    def __init__(self, config: dict = None):
        self.cfg = config or {}
        
        # ─── 稳态变量 ───
        self.energy = 1.0       # 血糖/能量储备 [0, 2]
        self.water = 1.0        # 水合状态 [0, 1]
        self.integrity = 1.0    # 身体结构完整性 [0, 1]
        self.health = 1.0       # 整体健康 [0, 1]
        self.fatigue = 0.0      # 疲劳度 [0, 1] → 睡眠压力
        self.stress = 0.0       # 慢性应激 [0, 1] 
        
        # ─── 生理节律 ───
        self.circadian = 0.0    # 昼夜节律 [0, 2π]，0=黎明
        self.sleep_hours = 0    # 连续清醒时长（tick）
        self.is_sleeping = False
        
        # ─── 速度 ───
        self.age_ticks = 0
        self.recovery_rate = 0.001
        
        # 历史记录
        self.history = []
        self.max_history = 500
    
    def update(self, dt: float = 1.0, energy_delta: float = -0.002,
               water_delta: float = -0.001, damage: float = 0.0,
               is_moving: bool = True, circadian_force: float = 0.0,
               was_moved_passively: bool = False):
        """每 tick 更新身体状态
           was_moved_passively: 是否被外力移动（非自主移动）
        """
        self.age_ticks += 1
        
        # 被动位移检测
        if was_moved_passively:
            self.stress = min(1.0, self.stress + 0.05)  # 被动移动→应激飙升
        
        # 昼夜节律推进
        self.circadian = (self.circadian + 0.01 * dt) % (2 * np.pi)
        circadian_mult = 0.5 * (1 + np.cos(self.circadian))  # 0~1 昼夜影响
        
        # 基础代谢（与昼夜节律相关）
        base_energy_cost = -0.0003 * (0.5 + circadian_mult)
        if is_moving:
            base_energy_cost *= 2.0
        if self.is_sleeping:
            base_energy_cost *= 0.1  # 睡眠时代谢极低
        
        # 更新能量
        self.energy = np.clip(self.energy + energy_delta + base_energy_cost, 0.0, 2.0)
        self.water = np.clip(self.water + water_delta, 0.0, 1.0)
        
        # 损伤
        if damage > 0:
            self.integrity = np.clip(self.integrity - damage, 0.0, 1.0)
        
        # 疲劳累积（清醒时增加，睡眠时减少）
        if self.is_sleeping:
            self.fatigue = max(0, self.fatigue - 0.005 * dt)
            self.sleep_hours = 0
            # 睡眠时微恢复
            self.integrity = min(1.0, self.integrity + self.recovery_rate * 0.5)
        else:
            # 集成短板修复：疲劳积累 0.0015→0.005/tick（清醒）——
            # 根因链：E14 食物稀缺→agent 存活仅 300-500 tick 即死亡，
            # 原 0.0015 在存活期内无法累积到 0.7 阈值→睡眠永不触发。
            # 0.005/tick → 140 tick 达 0.7，存活期内可触发。
            self.fatigue = min(1.0, self.fatigue + 0.005 * dt)
            self.sleep_hours += 1
        
        # 应激累积：关键变量偏离稳态时应激上升
        if self.energy < 0.2 or self.water < 0.2 or self.integrity < 0.3:
            self.stress = min(1.0, self.stress + 0.01 * dt)
        elif self.health > 0.7:
            self.stress = max(0, self.stress - 0.005 * dt)  # 健康时自然恢复
        
        # 二级效应：变量之间的耦合
        if self.water < 0.3:
            self.energy = max(0, self.energy - 0.001)  # 缺水→代谢下降
        if self.energy < 0.2:
            self.fatigue = min(1.0, self.fatigue + 0.002)  # 饥饿→疲劳加速
        if self.integrity < 0.5:
            self.stress = min(1.0, self.stress + 0.001)  # 受伤→应激
        
        # 整体健康 = 各变量的加权乘积
        self.health = (
            np.clip(self.energy / 1.0, 0, 1) * 
            np.clip(self.water, 0, 1) *
            np.clip(self.integrity, 0, 1) *
            np.clip(1 - self.fatigue, 0, 1) *
            np.clip(1 - self.stress * 0.5, 0, 1)
        )
        
        # 记录
        self.history.append(self.get_state_dict())
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return self.health
    
    def get_state_dict(self) -> dict:
        return {
            "energy": self.energy,
            "water": self.water,
            "integrity": self.integrity,
            "health": self.health,
            "fatigue": self.fatigue,
            "stress": self.stress,
            "circadian": self.circadian,
            "sleeping": int(self.is_sleeping),
        }
    
    def get_state_vector(self) -> np.ndarray:
        """返回给 LNN 的身体状态向量"""
        return np.array([
            self.energy / 2.0,    # [0,1] 归一化
            self.water,           # [0,1]
            self.health,          # [0,1]
            self.fatigue,         # [0,1]
            self.stress,          # [0,1]
            int(self.is_sleeping),
            np.cos(self.circadian),  # 昼夜节律
            np.sin(self.circadian),
        ])
    
    def is_critical(self) -> bool:
        """是否处在危险状态"""
        return (self.energy < 0.1 or self.water < 0.1 or 
                self.integrity < 0.2 or self.health < 0.1)
    
    def reset(self):
        self.__init__(self.cfg)
