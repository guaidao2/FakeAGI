"""
自模型 (Self Model) — AGI 的"我"

核心功能：
1. 追踪系统内部状态（能量、完整性、预测误差）
2. 计算存在概率 P ∈ [0, 1]
3. 生成好奇心预算（存在概率高 → 探索预算多）
4. 关键变量监控和稳态告警

设计原理：
  意识 = 系统对自身的建模 (Metzinger)
  自维持 = 最大化存在概率
  好奇心 = 生存有余裕时的探索预算
"""

import numpy as np


class SelfModel:
    def __init__(self, config: dict = None):
        # 内部状态变量
        self.energy = 1.0           # 能量水平 [0, 1]
        self.integrity = 1.0        # 结构完整性 [0, 1]
        self.avg_surprise = 0.0     # 近期平均惊奇 [0, ∞)
        self.existential_drift = 0.0 # 存在概率的变化率
        
        # 存在概率（核心目标）
        self.survival_prob = 1.0
        
        # 好奇心预算
        self.curiosity_budget = 0.3  # [0, 1]
        
        # 历史记录（供反事实通道使用）
        self.state_history = []
        self.max_history = 1000
        
        # 配置
        self.config = config or {}
        self.survival_threshold = 0.1  # 低于此值触发紧急行为
    
    def update(self, 
               energy_delta: float = 0.0,
               integrity_delta: float = 0.0,
               surprise: float = 0.0,
               dt: float = 1.0):
        """每 tick 更新自模型"""
        # 更新内部状态
        self.energy = np.clip(self.energy + energy_delta, 0.0, 1.0)
        self.integrity = np.clip(self.integrity + integrity_delta, 0.0, 1.0)
        
        # 指数移动平均惊奇
        alpha = 0.01 * dt
        self.avg_surprise = (1 - alpha) * self.avg_surprise + alpha * surprise
        
        # 计算存在概率
        # 三个因素：能量、完整性、惊奇（惊奇太大 = 预测失败 = 危险）
        p_energy = self.energy
        p_integrity = self.integrity
        p_predict = np.exp(-self.avg_surprise * 2.0)  # 惊奇→0 → 概率→1
        
        old_prob = self.survival_prob
        self.survival_prob = np.clip(p_energy * p_integrity * p_predict, 0.0, 1.0)
        self.existential_drift = self.survival_prob - old_prob
        
        # 好奇心预算：生存概率高时分配更多探索预算
        raw_budget = np.clip(self.survival_prob * 0.5 - 0.1, 0.0, 0.5)
        self.curiosity_budget = raw_budget
        
        # 记录历史
        self.state_history.append({
            "energy": self.energy,
            "integrity": self.integrity,
            "survival": self.survival_prob,
            "surprise": surprise,
            "curiosity": self.curiosity_budget,
        })
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        return self.survival_prob
    
    def is_emergency(self) -> bool:
        """是否需要紧急行为"""
        return self.survival_prob < self.survival_threshold
    
    def get_exploration_ratio(self) -> float:
        """探索 vs 利用 的比率"""
        if self.is_emergency():
            return 0.0  # 快死了，全部利用
        return self.curiosity_budget
    
    def get_state_vector(self) -> np.ndarray:
        """返回自模型的完整状态向量（给 LNN 作为额外输入）"""
        return np.array([
            self.energy,
            self.integrity,
            self.avg_surprise,
            self.survival_prob,
            self.curiosity_budget,
            self.existential_drift,
        ])
    
    def reset(self):
        """重置内部状态（换环境时使用）"""
        self.__init__(self.config)
