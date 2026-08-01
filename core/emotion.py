"""
情绪系统（Emotion System）— 显式可测情绪信号

哲学：情感是物理的（② 活着/⑥ 耦合循环）。
情绪不是附加的"感受"，而是生理状态（应激/健康/能量/伤害）+
认知状态（surprise/新异性）的**可测复合信号**，并调制决策：
  - 恐惧（fear）：低能量/高应激/高风险 → 行为激进（冒险觅食）
  - 好奇心（curiosity）：高 surprise/高能量 → 探索行为
  - 平静（calm）：稳态良好 → 常规行为

输出情绪向量 [fear, curiosity, calm]（和为 1），供决策层调制。

验证（test_emotion.py）：
  A. 快饿死 → fear 飙升（情感的物理体现）
  B. 高应激 → fear 上升
  C. 高 surprise → curiosity 上升
  D. 稳态 → calm 主导
  E. 恐惧调制：fear 高时行为激进（探索率/冒险度上升）
"""

import numpy as np


class EmotionSystem:
    def __init__(self, fear_gain=2.5, curiosity_gain=1.5):
        self.fear_gain = fear_gain
        self.curiosity_gain = curiosity_gain
        # 历史（平滑用）
        self.history = []
        self.current = {"fear": 0.0, "curiosity": 0.0, "calm": 1.0}

    def update(self, energy=0.5, water=0.5, health=1.0, stress=0.0,
               surprise=0.0, danger=0.0, tick=0):
        """生理 + 认知 → 情绪向量
        入口 NaN/Inf 防御：非有限输入钳制到安全默认（防上游损坏静默放大）"""
        def _safe(v, default=0.0, lo=0.0, hi=1.0):
            if v is None or not np.isfinite(v):
                return default
            return float(np.clip(v, lo, hi))
        energy = _safe(energy, default=0.5)
        water = _safe(water, default=0.5)
        health = _safe(health, default=1.0)
        stress = _safe(stress)
        surprise = _safe(surprise)
        danger = _safe(danger)

        # 恐惧源：低能量/低水/低健康/高应激/高危险
        need_fear = (1.0 - energy) * 0.6 + (1.0 - health) * 0.8 \
                    + stress * 1.0 + danger * 1.2 + (1.0 - water) * 0.3
        fear = min(1.0, need_fear * self.fear_gain * 0.4)
        # 好奇心源：高 surprise + 状态尚可（恐惧低时不压制好奇）
        curiosity = min(1.0, surprise * self.curiosity_gain * 0.6) * (1.0 - fear * 0.7)
        # 平静：剩余
        calm = max(0.0, 1.0 - fear - curiosity)

        # 归一化（防浮点漂移）
        total = fear + curiosity + calm
        if total > 0:
            fear, curiosity, calm = fear / total, curiosity / total, calm / total

        self.current = {"fear": float(fear), "curiosity": float(curiosity),
                        "calm": float(calm)}
        self.history.append((tick, self.current.copy()))
        if len(self.history) > 500:
            self.history.pop(0)
        return self.current

    def get_state(self) -> dict:
        return dict(self.current)

    def modulate_action(self, base_exploration: float) -> float:
        """情绪调制探索率：恐惧→激进（更高随机率=冒险），好奇→探索"""
        f = self.current["fear"]
        c = self.current["curiosity"]
        # 恐惧高 → 探索率升（冒险觅食，宁闯不饿死）
        # 好奇高 → 探索率升（信息寻求）
        return min(1.0, base_exploration + f * 0.5 + c * 0.3)
