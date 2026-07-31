"""
隐变量模型 — 对不可观测事物的推断

核心缺口：FakeAGI 假设"世界 = 观测"。真正的智能必须知道"世界 > 观测"。

例如：食物在 (5,5) 但被锁住 → 观测只显示食物方向和开关方向，
系统必须推断"存在一个不可见的锁机制"。

实现：
- LatentState 维护一组隐变量（不可观测因素的当前估计）
- 每个隐变量有：估计值、不确定性、上一次被修正的 tick
- 推断规则：当观测与预测持续不一致 → 更新隐变量以解释差异
- 隐变量参与世界模型的输入（作为额外的隐藏上下文）
"""

import numpy as np


class LatentVariable:
    """单个隐变量"""
    def __init__(self, name: str, dim: int = 1, init: float = 0.0):
        self.name = name
        self.value = np.full(dim, init, dtype=np.float32)
        self.uncertainty = np.ones(dim, dtype=np.float32)  # 1 = 完全不确定
        self.last_updated = -1
        self.update_count = 0
    
    def update(self, new_value: np.ndarray, tick: int):
        """更新估计（指数移动平均）"""
        alpha = 0.3  # 学习率
        self.value = (1 - alpha) * self.value + alpha * new_value
        self.uncertainty = np.clip(self.uncertainty * 0.9, 0.05, 1.0)
        self.last_updated = tick
        self.update_count += 1
    
    def increase_uncertainty(self):
        self.uncertainty = np.clip(self.uncertainty + 0.1, 0.05, 1.0)


class LatentStateModel:
    """
    隐变量状态模型。
    当预测误差持续高且无法归因于已知因素时，
    系统推断存在一个或多个不可观测的隐变量在影响结果。
    """
    def __init__(self):
        self.latents = {}
        self.high_error_streak = 0
        self.last_hidden_context = None
    
    def register(self, name: str, dim: int = 1):
        if name not in self.latents:
            self.latents[name] = LatentVariable(name, dim)
    
    def observe_prediction_error(self, surprise: float, tick: int,
                                 obs: np.ndarray = None) -> bool:
        """
        处理一次预测误差。
        如果误差持续高 → 推断存在隐变量 → 返回 True（表示"世界不止观测"）
        """
        if surprise > 0.3:
            self.high_error_streak += 1
        else:
            self.high_error_streak = max(0, self.high_error_streak - 1)
        
        if self.high_error_streak > 30:
            # 持续高误差 → 隐变量推断
            self.register("hidden_factor", dim=1)
            # 用误差信号更新隐变量（方向未知，用噪声探索）
            noise = np.random.normal(0, 0.1, 1)
            self.latents["hidden_factor"].update(noise, tick)
            return True
        return False
    
    def get_context_vector(self, fixed_dim: int = 4) -> np.ndarray:
        """隐变量上下文（固定维度，防止触发感知生长）"""
        vecs = []
        for name, lat in self.latents.items():
            vecs.append(lat.value)
            vecs.append(lat.uncertainty)
        if not vecs:
            return np.zeros(fixed_dim, dtype=np.float32)
        ctx = np.concatenate(vecs)
        if len(ctx) >= fixed_dim:
            return ctx[:fixed_dim]
        return np.pad(ctx, (0, fixed_dim - len(ctx)))
    
    def explain(self) -> list:
        """返回当前隐变量解释（供日志/语言输出）"""
        explanations = []
        for name, lat in self.latents.items():
            explanations.append(
                f"{name}: value={lat.value[0]:.2f}, "
                f"uncertainty={lat.uncertainty[0]:.2f}, "
                f"updates={lat.update_count}"
            )
        return explanations
