"""
τ 自适应调度器 — 动态调节 LNN 的时间尺度

原理：
  高惊奇 → 缩短 τ → 快速适应新情况
  低惊奇 → 延长 τ → 稳定记忆长期模式
"""

import numpy as np


class TauScheduler:
    def __init__(self, tau_min=1.0, tau_max=50.0, base_dt=0.1):
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.base_dt = base_dt
        self.current_tau = 10.0
    
    def adapt(self, surprise: float, survival_prob: float):
        """根据惊奇和生存概率调节时间尺度"""
        # 高惊奇 → 短 τ（快速学习新事物）
        # 低惊奇 → 长 τ（保持稳定）
        target_tau = np.clip(self.tau_max - surprise * 30, self.tau_min, self.tau_max)
        # 生存概率低时倾向短 τ
        if survival_prob < 0.3:
            target_tau = min(target_tau, 5.0)
        self.current_tau += 0.1 * (target_tau - self.current_tau)
        return self.current_tau
