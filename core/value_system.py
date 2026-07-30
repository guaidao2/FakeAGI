"""
价值系统 — 内生的好/坏/安全/危险判断

不依赖外部 reward，而是基于自模型的存在概率变化来赋值。
一个行动导致存在概率上升 → 好
一个行动导致存在概率下降 → 坏
"""

class ValueSystem:
    def __init__(self):
        self.values = {}  # context → value_score
        self.learning_rate = 0.1
    
    def update(self, context_key: str, survival_delta: float):
        """根据存在概率变化学习价值"""
        old = self.values.get(context_key, 0.0)
        self.values[context_key] = old + self.learning_rate * (survival_delta - old)
    
    def get_value(self, context_key: str) -> float:
        return self.values.get(context_key, 0.0)
    
    def is_good(self, context_key: str) -> bool:
        return self.get_value(context_key) > 0.0
