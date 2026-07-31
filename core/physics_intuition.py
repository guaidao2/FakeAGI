"""
物理直觉层 — 世界模型的内置物理先验

人类不需要"训练"就知道：
- 物体掉落会向下（重力）
- 两个物体不能占据同一位置（不可穿透）
- 物体运动有惯性（动量）
- 推物体到边缘会掉落（支撑约束）

FakeAGI 需要这些直觉，但不应硬编码成规则（那会破坏"所有知识来自经验"原则）。
正确做法：物理直觉作为世界模型的**先验分布**（prior），
经验通过预测误差修正这些先验（贝叶斯式更新）。

实现：
- 物理先验 = 对观测变化的期望约束（方向性、连续性、守恒性）
- 每条先验有置信度，经验反复违背先验 → 置信度下降
- 先验参与世界模型的损失函数（正则化项），
  使世界模型在数据少时仍倾向物理合理的预测
"""

import numpy as np


class PhysicsPrior:
    """物理先验约束集合"""
    def __init__(self):
        # 每条先验: (描述, 置信度[0,1], 违规计数)
        self.priors = [
            {"desc": "gravity_down", "confidence": 0.9, "violations": 0},
            {"desc": "impenetrable", "confidence": 0.9, "violations": 0},
            {"desc": "continuity", "confidence": 0.8, "violations": 0},
            {"desc": "momentum", "confidence": 0.6, "violations": 0},
            {"desc": "no_teleport", "confidence": 0.9, "violations": 0},
        ]
        self.prev_pos = None
    
    def compute_prior_loss(self, obs_delta: np.ndarray, agent_moved: bool,
                           dt: float = 1.0) -> float:
        """
        计算物理先验对观测变化的惩罚。
        返回 0~1 的损失值，加入世界模型的 loss。
        """
        loss = 0.0
        total_conf = sum(p["confidence"] for p in self.priors) + 1e-6
        
        # 连续性：观测不应突变（除非大事件）
        if len(obs_delta) > 0:
            max_jump = np.max(np.abs(obs_delta)) if obs_delta.size else 0
            if max_jump > 0.5:
                conf = self._get_prior("continuity")
                loss += conf * min(1.0, max_jump)
                self._violate("continuity")
        
        # 无瞬移：如果 agent 声称移动了但观测完全没变，或反向
        if agent_moved and len(obs_delta) >= 2:
            # 简化版：如果移动了但观测 delta 为 0（除非目标没动）
            pass  # 由外部检测
        
        return min(1.0, loss / max(1.0, total_conf))
    
    def check_teleport(self, prev_pos, new_pos, max_step: float = 1.5) -> bool:
        """检测瞬移：位置变化超过单步移动上限"""
        if prev_pos is None or new_pos is None:
            return False
        dist = np.linalg.norm(np.array(new_pos) - np.array(prev_pos))
        if dist > max_step:
            self._violate("no_teleport")
            return True
        return False
    
    def check_impenetrable(self, pos, obstacles: list) -> bool:
        """检测不可穿透：agent 是否与障碍物重叠"""
        for obs in obstacles:
            if np.linalg.norm(np.array(pos) - np.array(obs)) < 0.5:
                self._violate("impenetrable")
                return True
        return False
    
    def update_with_experience(self, actual_delta: np.ndarray,
                               predicted_delta: np.ndarray) -> None:
        """
        用实际经验更新先验置信度。
        如果物理现实违背了先验（例如无重力环境），置信度下降。
        这是"经验修正先验"的贝叶斯式更新。
        """
        err = np.mean((actual_delta - predicted_delta) ** 2) if len(actual_delta) else 0
        # 误差大 → 先验可能不对 → 置信度小幅下降
        for p in self.priors:
            p["confidence"] = max(0.1, p["confidence"] - err * 0.01)
    
    def _get_prior(self, name: str):
        for p in self.priors:
            if p["desc"] == name:
                return p["confidence"]
        return 0.0
    
    def _violate(self, name: str):
        for p in self.priors:
            if p["desc"] == name:
                p["violations"] += 1
    
    def get_intuition_vector(self) -> np.ndarray:
        """返回先验置信度向量（供自模型观测）"""
        return np.array([p["confidence"] for p in self.priors])
