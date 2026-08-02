"""
睡眠巩固系统 — 离线经验重放 + 记忆巩固

生物学功能：
  1. 白天积累的经验在睡眠时重放
  2. 重要记忆被巩固，不重要记忆被清除
  3. 突触权重归一化（避免饱和）
  4. 情绪记忆被处理（降低应激）
"""

import numpy as np


class SleepCycle:
    def __init__(self):
        self.is_sleeping = False
        self.sleep_duration = 0
        self.target_sleep = 0
        self.dream_log = []
    
    def should_sleep(self, fatigue: float, circadian: float, energy: float = 1.0) -> bool:
        # 集成短板修复（E14 睡眠 0 次触发根因）：能量条件 0.5→0.3——
        # 原 0.5 在食物稀缺环境（能量常<0.5）下条件永假，睡眠永不触发。
        # 生物学：疲劳 0.7 是"困"，能量>0.3 即可睡（<0.3 仍优先觅食）。
        return fatigue > 0.7 and energy > 0.3
    
    def should_wake(self, fatigue: float, energy: float = 0) -> bool:
        return (fatigue < 0.2 and self.sleep_duration > 20) or (energy > 1.5 and fatigue < 0.5)
    
    def consolidate(self, replay_buffer: list) -> list:
        """睡眠巩固：重放经验并返回巩固后的记忆。
        B2 接线（DESIGN_CONCEPTS §7.5/CLS）：按显著性加权重放——
        高 surprise（预测误差大=信息量高）的经验优先重放（原均匀抽样
        np.random.choice——没有重要性概念）。"""
        if not replay_buffer:
            return []
        
        n = min(32, len(replay_buffer))
        # 显著性权重：surprise 高的经验优先（+0.1 保底防零权重）
        # nit：None 防护（与 getattr 分支对称）
        weights = []
        for item in replay_buffer:
            if isinstance(item, dict):
                s = item.get("surprise", 0.0)
                s = float(s) if s is not None else 0.0
            else:
                s = getattr(item, "surprise", 0.0) or 0.0
            weights.append(0.1 + s)
        total = sum(weights)
        probs = [w / total for w in weights]
        indices = np.random.choice(len(replay_buffer), n, replace=False,
                                   p=probs)
        
        # 对采样的经验做"巩固"（模拟重放）
        consolidated = []
        for idx in indices:
            item = replay_buffer[idx].copy() if hasattr(replay_buffer[idx], 'copy') else replay_buffer[idx]
            consolidated.append(item)
        
        return consolidated
