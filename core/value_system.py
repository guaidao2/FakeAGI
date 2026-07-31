"""
价值系统进化 — 次级价值可调整

原版（⑩条）：价值由架构师注入，不可变。
升级版：分层价值系统。
- 核心价值（不可变）：存续优先、预测误差最小化、生长当容量不足
- 次级价值（可进化）：什么算好/什么算坏 → 由经验调整

例如：
- 初始：食物 = +1.0, 危险 = -1.0
- 经验：吃某种食物会中毒 → 该食物的价值下降
- 经验：某种行为反复导致死亡 → 该行为被标记为负价值

实现：
- ValueSystem 维护价值表 {stimulus: value}
- 核心价值硬编码（不能改）
- 次级价值通过预测误差更新（贝叶斯式）
"""

import numpy as np


class EvolvableValueSystem:
    """可进化的价值系统"""
    def __init__(self):
        # 核心价值（不可变）—— 违反会导致系统自我终止
        self.core_values = {
            "survival": 1.0,       # 存续优先
            "error_min": 1.0,      # 预测误差最小化
            "growth": 1.0,         # 容量不足时生长
        }
        
        # 次级价值（可进化）
        self.secondary_values = {}   # name -> {"value": float, "confidence": float, "updates": int}
        self._init_secondary()
    
    def _init_secondary(self):
        """初始化次级价值（架构师注入的初始值，可被经验修改）"""
        self._set_secondary("food", 0.8, 0.5)
        self._set_secondary("water", 0.7, 0.5)
        self._set_secondary("danger", -0.8, 0.5)
        self._set_secondary("sleep", 0.5, 0.5)
        self._set_secondary("explore", 0.4, 0.5)
    
    def _set_secondary(self, name: str, value: float, confidence: float):
        self.secondary_values[name] = {
            "value": value, "confidence": confidence, "updates": 0
        }
    
    def update_with_experience(self, stimulus: str, outcome: float) -> None:
        """
        用经验更新次级价值。
        outcome > 0 = 好结果（能量增加/存活），< 0 = 坏结果。
        高置信度时更新慢（价值稳定），低置信度时更新快（价值未定）。
        """
        if stimulus not in self.secondary_values:
            # 新刺激：从结果学习初始价值
            self._set_secondary(stimulus, outcome, 0.3)
            return
        
        entry = self.secondary_values[stimulus]
        confidence = entry["confidence"]
        
        # 更新幅度 = 学习率 × (1 - confidence)
        alpha = 0.2 * (1 - confidence)
        entry["value"] = (1 - alpha) * entry["value"] + alpha * outcome
        entry["confidence"] = min(1.0, confidence + 0.05)
        entry["updates"] += 1
    
    def get_value(self, stimulus: str) -> float:
        """获取刺激的价值（核心 + 次级）"""
        if stimulus in self.core_values:
            return self.core_values[stimulus]
        if stimulus in self.secondary_values:
            return self.secondary_values[stimulus]["value"]
        return 0.0
    
    def get_action_weights(self, obs: np.ndarray) -> np.ndarray:
        """
        根据当前观测计算动作价值权重。
        简化：obs 维度与次级价值名称的映射由外部提供。
        返回动作偏好向量。
        """
        # 默认：无额外偏好（由 GameNN 负责）
        return np.zeros(0)
    
    def get_evolution_stats(self) -> dict:
        """价值演化统计"""
        return {
            name: {"value": round(e["value"], 3),
                   "confidence": round(e["confidence"], 3),
                   "updates": e["updates"]}
            for name, e in self.secondary_values.items()
        }
