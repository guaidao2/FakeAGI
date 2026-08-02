"""
好奇心预算分配器 — 决定系统何时探索、探索什么

原则：
  存在概率高 → 好奇心预算多 → 探索未知
  存在概率低 → 好奇心预算少 → 专注已知的生存策略
  惊奇信号 → 好奇心消耗 → 学习/记忆
"""

import numpy as np
import math


class CuriosityManager:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.budget = 0.3
        self.curiosity_map = {}  # "探索对象" → 探索次数
        self.exploration_count = 0
        self.total_explorations = 0
        # B1 接线（DESIGN_CONCEPTS §7.5）：learning progress 通道——
        # 好奇心由 world_model 误差**下降率**驱动（ICM/Pathak 2017），
        # 非 novelty 计数（noisy-TV 陷阱）。局部可替代/叠加计数。
        self.lp_enabled = self.config.get("lp_enabled", True)
        self._loss_hist = []          # 世界模型误差滑动窗口
        self._lp_value = 0.0          # 当前 learning progress（0~1）
        self._lp_window = self.config.get("lp_window", 50)
    
    def update_learning_progress(self, world_loss: float):
        """喂 world_model 误差——learning progress = 误差下降率。
        loss 下降（在学）→ 高好奇；loss 停滞/上升 → 低好奇。
        与 ICM 一致：奖励"学习进展"而非"新奇"。"""
        if not self.lp_enabled or world_loss is None:
            return
        # nit 修复：NaN/Inf 防护（污染 _lp_value 会永久失效）
        if not math.isfinite(float(world_loss)):
            return
        self._loss_hist.append(float(world_loss))
        if len(self._loss_hist) > self._lp_window:
            self._loss_hist.pop(0)
        n = len(self._loss_hist)
        if n < 10:
            return
        half = n // 2
        recent = np.mean(self._loss_hist[half:])
        earlier = np.mean(self._loss_hist[:half])
        # 下降率 >0 = 在学；归一化到 0~1（clamp）
        self._lp_value = float(np.clip((earlier - recent) / max(abs(earlier), 1e-6), 0.0, 1.0))
    
    @property
    def learning_progress(self) -> float:
        return self._lp_value
    
    def update_budget(self, survival_prob: float):
        """根据存在概率更新好奇心预算"""
        if survival_prob > 0.7:
            self.budget = min(0.5, self.budget + 0.01)
        elif survival_prob > 0.3:
            self.budget = 0.2
        else:
            self.budget = max(0.0, self.budget - 0.02)
        # B1：learning progress 调制预算（在学的系统值得探索更多）
        if self.lp_enabled and self._lp_value > 0.3:
            self.budget = min(0.6, self.budget + 0.05 * self._lp_value)
    
    def should_explore(self, surprise: float) -> bool:
        """当前 tick 是否应该探索"""
        base = surprise > (1.0 - self.budget)
        # B1：learning progress 高时，即使 surprise 低也倾向探索
        if self.lp_enabled and self._lp_value > 0.4 and np.random.random() < self._lp_value * 0.3:
            return True
        return base
    
    def record_exploration(self, context_key: str):
        """记录一次探索行为"""
        self.curiosity_map[context_key] = self.curiosity_map.get(context_key, 0) + 1
        self.exploration_count += 1
        self.total_explorations += 1
    
    def get_boredom(self, context_key: str) -> float:
        """对某个上下文的厌倦程度（探索过多次后降低好奇心）"""
        count = self.curiosity_map.get(context_key, 0)
        return min(1.0, count / 10.0)
