"""
知识缺口检测器 — 元认知层第0步

跟踪系统的"已知的未知"：
  1. 世界模型预测误差持续偏高 → 环境知识缺口
  2. GameNN 置信度持续偏低 → 策略知识缺口
  3. 反事实预测 vs 真实结果差异大 → 因果知识缺口
  4. 空间记忆中被标记为"未探索"的区域 → 空间知识缺口
"""

import numpy as np
from collections import deque


class KnowledgeGap:
    """表示一个具体的知识缺口"""
    def __init__(self, kind: str, score: float, context: dict = None):
        self.kind = kind          # "world_model" / "strategy" / "causal" / "spatial"
        self.score = score        # 0~1, 越高越需要探索
        self.context = context or {}
        # 上下文可能含：位置、动作、观测索引、时间窗口
        self.created_at = 0
    
    def __repr__(self):
        return f"[GAP {self.kind} s={self.score:.2f}]"


class GapDetector:
    def __init__(self, window: int = 100):
        self.window = window
        self.error_history = deque(maxlen=window)
        self.confidence_history = deque(maxlen=window)
        self.surprise_history = deque(maxlen=window)
        self.causal_error_history = deque(maxlen=50)
        self.reward_history = deque(maxlen=20)
        self.pos_history = deque(maxlen=10)
        self.tick = 0
        self.fast_failure_detected = False
    
    def update(self, world_model_loss: float, gamenn_confidence: float,
               surprise: float, causal_prediction_error: float = None,
               energy_delta: float = 0.0, agent_pos: list = None,
               energy_level: float = None):
        self.tick += 1
        self.error_history.append(world_model_loss)
        self.confidence_history.append(gamenn_confidence)
        self.surprise_history.append(surprise)
        if causal_prediction_error is not None:
            self.causal_error_history.append(causal_prediction_error)
        self.reward_history.append(energy_delta)
        if agent_pos:
            self.pos_history.append(tuple(agent_pos))
        self.fast_failure_detected = False
        # 使用 body.energy 水平或 energy_delta 检测能量是否持续下降
        if energy_level is not None:
            if energy_level < 0.3:
                self.fast_failure_detected = True
        elif len(self.reward_history) >= 10:
            recent_r = list(self.reward_history)[-10:]
            if all(r <= 0.001 for r in recent_r):
                if len(self.pos_history) >= 3 and len(set(self.pos_history)) >= 2:
                    self.fast_failure_detected = True
    
    def detect(self) -> KnowledgeGap:
        """检测当前最突出的知识缺口，返回 None 表示无显著缺口"""
        gaps = []
        
        # 0. 快速失败检测（优先级最高）
        if self.fast_failure_detected:
            gaps.append(KnowledgeGap("causal", 0.9,
                {"error": 1.0, "type": "no_reward", "fast": True}))
        if len(self.error_history) >= 20:
            recent_error = np.mean(list(self.error_history)[-10:])
            earlier_error = np.mean(list(self.error_history)[-20:-10])
            if recent_error > 0.3 and abs(recent_error - earlier_error) < 0.02:
                gaps.append(KnowledgeGap("world_model", 
                    min(1.0, recent_error * 2), {"trend": "plateau"}))
        
        # 2. 策略缺口：GameNN 置信度持续低下
        if len(self.confidence_history) >= 20:
            recent_conf = np.mean(list(self.confidence_history)[-10:])
            if recent_conf < 0.2:
                gaps.append(KnowledgeGap("strategy",
                    min(1.0, (0.3 - recent_conf) * 5), {"confidence": recent_conf}))
        
        # 3. 因果缺口：反事实预测偏离真实
        if len(self.causal_error_history) >= 5:
            causal_err = np.mean(self.causal_error_history)
            if causal_err > 0.5:
                gaps.append(KnowledgeGap("causal",
                    min(1.0, causal_err), {"error": causal_err}))
        
        # 4. 惊奇缺口：持续的高惊奇
        if len(self.surprise_history) >= 20:
            recent_surp = np.mean(list(self.surprise_history)[-10:])
            if recent_surp > 0.4:
                gaps.append(KnowledgeGap("surprise",
                    min(1.0, recent_surp * 2), {"surprise": recent_surp}))
        
        if not gaps:
            return None
        
        # 返回评分最高的缺口
        gaps.sort(key=lambda g: g.score, reverse=True)
        return gaps[0]
