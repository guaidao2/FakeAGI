"""
自我评估模块 — 元认知层

定期评估系统自身状态，生成"自我报告"：
  1. 最近 N tick 的生存趋势（稳定/恶化/改善）
  2. 世界模型在哪些观测上预测不准（知识地图）
  3. 哪些动作/策略产生了最好/最差结果
  4. 综合评估：当前"胜任力"水平
"""

import numpy as np
from collections import deque


class SelfAssessment:
    def __init__(self, window: int = 500):
        self.window = window
        self.health_history = deque(maxlen=window)
        self.energy_history = deque(maxlen=window)
        self.surprise_history = deque(maxlen=window)
        self.action_success = {}   # action_idx → [success_count, total_count]
        self.competence = 0.5       # 0~1, 综合胜任力
        self.trend = "stable"       # "improving" / "stable" / "declining"
        self.last_assessment = {}
    
    def record(self, health: float, energy: float, surprise: float,
               action: int, survived: bool):
        self.health_history.append(health)
        self.energy_history.append(energy)
        self.surprise_history.append(surprise)
        
        if action not in self.action_success:
            self.action_success[action] = [0, 0]
        self.action_success[action][1] += 1
        if survived and surprise < 0.3:
            self.action_success[action][0] += 1
    
    def assess(self) -> dict:
        """综合自评"""
        if len(self.health_history) < 50:
            return {"competence": 0.5, "trend": "stable", "note": "insufficient_data"}
        
        recent_h = np.mean(list(self.health_history)[-50:])
        early_h = np.mean(list(self.health_history)[:50])
        trend_h = recent_h - early_h
        
        if trend_h > 0.1:
            self.trend = "improving"
        elif trend_h < -0.1:
            self.trend = "declining"
        else:
            self.trend = "stable"
        
        # 胜率评估
        success_rates = []
        for action, (s, t) in self.action_success.items():
            if t > 0:
                success_rates.append(s / t)
        avg_success = np.mean(success_rates) if success_rates else 0.0
        
        # 综合胜任力
        survival_score = recent_h
        prediction_score = 1.0 - min(1.0, np.mean(list(self.surprise_history)[-50:]))
        action_score = avg_success
        self.competence = 0.5 * survival_score + 0.3 * prediction_score + 0.2 * action_score
        
        self.last_assessment = {
            "competence": self.competence,
            "trend": self.trend,
            "avg_health_50": recent_h,
            "avg_surprise_50": np.mean(list(self.surprise_history)[-50:]),
            "action_success_rate": avg_success,
            "note": "all_nominal" if self.competence > 0.4 else "performance_degraded"
        }
        return self.last_assessment
    
    def get_best_action(self) -> int:
        """返回历史上成功率最高的动作"""
        best_a, best_r = 0, -1
        for a, (s, t) in self.action_success.items():
            if t > 5 and (s / t) > best_r:
                best_r = s / t
                best_a = a
        return best_a
