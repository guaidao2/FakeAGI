"""
元-元认知：学习策略管理器

核心缺口：系统能"知道自己不知道"（元认知），
但不能"知道自己该换一种方式学习"（元-元认知）。

实现：
- 检测：当前学习策略（世界模型更新 + GameNN 更新）是否有效
- 策略集：
  1. default      — 标准学习（世界模型 + GameNN）
  2. exploration  — 提高探索率（更多随机动作）
  3. replay_focus — 重点重放高误差经验
  4. growth_boost — 强制触发生长（容量不足时）
  5. rest         — 暂停学习（进入睡眠，巩固记忆）
- 监控：每个策略的效果（误差下降率）
- 切换：效果差 → 换下一个策略；效果好 → 保持
"""

import numpy as np
from collections import deque


class LearningStrategyManager:
    """元-元认知：管理自己的学习过程"""
    def __init__(self):
        self.strategies = ["default", "exploration", "replay_focus",
                           "growth_boost", "rest"]
        self.current = "default"
        self.strategy_scores = {s: 0.5 for s in self.strategies}
        self.error_history = deque(maxlen=200)
        self.strategy_history = deque(maxlen=100)
        self.switch_count = 0
        self.steps_on_current = 0
        self.switch_threshold = 50  # 每个策略至少尝试 50 tick
    
    def update(self, world_loss: float, surprise: float,
               confidence: float, health: float) -> str:
        """
        每 tick 调用。返回当前策略。
        监控误差趋势，必要时切换策略。
        """
        self.error_history.append(world_loss)
        self.steps_on_current += 1
        
        # 记录当前策略的误差趋势
        if len(self.error_history) >= 30:
            recent = np.mean(list(self.error_history)[-10:])
            earlier = np.mean(list(self.error_history)[-20:-10])
            improvement = earlier - recent  # >0 表示误差在下降
            
            # 更新当前策略得分
            self.strategy_scores[self.current] = (
                0.9 * self.strategy_scores[self.current] + 0.1 * improvement
            )
            
            # 切换条件：当前策略无明显改善 + 已尝试足够久
            if (improvement < 0.001 and self.steps_on_current > self.switch_threshold):
                self._switch()
        
        return self.current
    
    def _switch(self):
        """切换到下一个策略（排除当前的最高分策略）"""
        # 简单轮转 + 得分修正
        candidates = [s for s in self.strategies if s != self.current]
        if not candidates:
            return
        # 选择得分最高的候选
        best = max(candidates, key=lambda s: self.strategy_scores[s])
        self.current = best
        self.switch_count += 1
        self.steps_on_current = 0
        self.strategy_history.append(best)
    
    def get_parameters(self) -> dict:
        """根据当前策略返回行为参数覆盖"""
        params = {"exploration": None, "force_grow": False,
                  "replay_focus": False, "force_sleep": False}
        if self.current == "exploration":
            params["exploration"] = 0.6
        elif self.current == "growth_boost":
            params["force_grow"] = False  # 只标记，不强制（避免破坏生长节奏）
        elif self.current == "replay_focus":
            params["replay_focus"] = True
        elif self.current == "rest":
            params["force_sleep"] = True
        return params
    
    def get_stats(self) -> dict:
        return {
            "current_strategy": self.current,
            "switch_count": self.switch_count,
            "strategy_scores": {k: round(v, 3) for k, v in self.strategy_scores.items()},
            "history": list(self.strategy_history)[-5:],
        }
