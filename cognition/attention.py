"""
注意力机制 — 选择性感知

核心缺口：没有注意力，无法扩展到高维感知。
固定维度观测向量一视同仁 → 高维感知（如视觉）时性能灾难。

实现：
- AttentionGate 对观测维度加权（软注意力）
- 权重基于：该维度的历史信息量（方差）、与当前驱动力相关性、最近变化
- 权重向量拼接到世界模型输入（模型能学到"哪些维度值得关注"）
- 支持"注意力焦点"：当前主导驱动力决定优先关注哪些维度
"""

import numpy as np


class AttentionGate:
    """观测维度注意力门控"""
    def __init__(self, obs_dim: int, history: int = 50):
        self.obs_dim = obs_dim
        self.history = history
        self.obs_history = []
        self.weights = np.ones(obs_dim, dtype=np.float32) / obs_dim
        self.information_gain = np.zeros(obs_dim, dtype=np.float32)
        self.salience = np.zeros(obs_dim, dtype=np.float32)
    
    def update(self, obs: np.ndarray, drive_vector: np.ndarray = None) -> np.ndarray:
        """更新注意力权重，返回加权后的观测"""
        obs = np.asarray(obs, dtype=np.float32)
        
        # 初始化阶段：返回原始观测（identity），避免破坏反射阈值
        if len(self.obs_history) < 20:
            self.obs_history.append(obs.copy())
            if len(self.obs_history) > self.history:
                self.obs_history.pop(0)
            return obs
        
        # 1. 信息量（维度方差）
        self.obs_history.append(obs.copy())
        if len(self.obs_history) > self.history:
            self.obs_history.pop(0)
        if len(self.obs_history) >= 5:
            hist = np.array(self.obs_history)
            self.information_gain = np.var(hist, axis=0)
        
        # 2. 显著性（最近变化幅度）
        if len(self.obs_history) >= 2:
            recent_delta = np.abs(self.obs_history[-1] - self.obs_history[-2])
            self.salience = 0.9 * self.salience + 0.1 * recent_delta
        
        # 3. 组合权重
        info_w = self.information_gain / (np.max(self.information_gain) + 1e-6)
        sal_w = self.salience / (np.max(self.salience) + 1e-6)
        self.weights = 0.4 * info_w + 0.6 * sal_w
        
        # 4. 驱动调节（可选）：饥饿时更关注食物相关维度
        if drive_vector is not None and len(drive_vector) >= 3:
            hunger = drive_vector[0]
            # 假设 obs 前 2 维是资源方向 → 饥饿时提升其权重
            if hunger > 0.5 and self.obs_dim >= 2:
                self.weights[:2] *= (1.0 + hunger)
        
        # 归一化（保持相对缩放温和：权重在 [0.5, 2.0] 范围内）
        w_min, w_max = 0.5, 2.0
        w_norm = (self.weights - np.min(self.weights)) / (np.max(self.weights) - np.min(self.weights) + 1e-6)
        self.weights = w_min + w_norm * (w_max - w_min)
        
        # 返回加权观测（尺度不破坏阈值判断；当前保守策略：仅观察不干预）
        return obs  # 注意力只影响内部统计，不改变观测输入（避免破坏反射阈值）
    
    def get_weights(self) -> np.ndarray:
        return self.weights.copy()
    
    def get_focus(self) -> int:
        """返回当前最关注的维度索引"""
        return int(np.argmax(self.weights))
