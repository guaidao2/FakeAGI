"""
P4: 观测抽象层 — 特征通道可增长

原始观测进来 → 特征提取 → 抽象向量（维度随经验增长）。

设计：
  - Channel 池：每个通道 = 一个特征提取器（对原始观测的一部分做变换）
  - 初始通道：原始观测直通（identity）
  - 新信号源接入：新增通道（维度增长）
  - 通道重要性由注意力决定（信息增益驱动）
  - 长期低信息通道：可被剪枝回收

这解决了"被动补丁式增长"问题：观测抽象层是主动的感知组织者，
新环境 → 新通道 → 抽象向量维度增长 → 触发全链路协调生长。
"""

import numpy as np


class FeatureChannel:
    """一个特征通道：把原始观测的一部分变换为特征"""
    def __init__(self, name: str, indices, transform="identity", dim: int = 1):
        self.name = name
        self.indices = list(indices)      # 原始观测的哪些维度进入此通道
        self.transform = transform        # identity / abs / square / norm
        self.dim = dim                    # 输出维度（当前为 1，可扩展）
        self.information_gain = 0.0       # 该通道的信息量（方差）
        self.active = True
        self.history = []

    def extract(self, raw_obs: np.ndarray) -> np.ndarray:
        """从原始观测提取此通道的特征"""
        if not self.active:
            return np.zeros(self.dim, dtype=np.float32)
        vals = [raw_obs[i] for i in self.indices if i < len(raw_obs)]
        if not vals:
            return np.zeros(self.dim, dtype=np.float32)
        if self.transform == "abs":
            out = np.mean(np.abs(vals))
        elif self.transform == "square":
            out = np.mean(np.square(vals))
        elif self.transform == "norm":
            out = np.linalg.norm(vals)
        else:  # identity
            out = np.mean(vals)
        # 信息增益跟踪（滚动方差）
        self.history.append(float(out))
        if len(self.history) > 50:
            self.history.pop(0)
        if len(self.history) >= 10:
            self.information_gain = float(np.var(self.history))
        return np.array([out], dtype=np.float32)


class ObservationAbstraction:
    """
    观测抽象层：原始观测 → 特征通道 → 抽象向量。
    维度 = 通道数，随新信号源增长。
    """
    def __init__(self, raw_dim: int = 4, max_channels: int = 16):
        self.raw_dim = raw_dim
        self.max_channels = max_channels
        self.channels = []
        # 初始：一个直通通道覆盖全部原始观测（identity 保底）
        self.add_channel("raw", list(range(raw_dim)), transform="identity")

    def add_channel(self, name: str, indices, transform="identity"):
        """新增特征通道（新信号源接入 → 维度增长）"""
        if len(self.channels) >= self.max_channels:
            return None
        # 去重：同名通道已存在则不重复
        for c in self.channels:
            if c.name == name:
                return c
        ch = FeatureChannel(name, indices, transform)
        self.channels.append(ch)
        return ch

    def observe(self, raw_obs: np.ndarray) -> np.ndarray:
        """原始观测 → 抽象向量"""
        raw_obs = np.asarray(raw_obs, dtype=np.float32)
        self.raw_dim = max(self.raw_dim, len(raw_obs))
        feats = []
        for ch in self.channels:
            feats.append(ch.extract(raw_obs))
        if not feats:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(feats)

    def get_abstract_dim(self) -> int:
        return len(self.channels)

    def prune_low_info(self, threshold: float = 0.0001):
        """剪枝：长期零信息通道回收（保留 raw 通道）"""
        removed = []
        for ch in self.channels:
            if ch.name != "raw" and ch.information_gain < threshold \
                    and len(ch.history) >= 10:
                ch.active = False
                removed.append(ch.name)
        if removed:
            self.channels = [c for c in self.channels if c.active]
        return removed

    def get_channel_names(self) -> list:
        return [c.name for c in self.channels]

    def get_state_dict(self) -> dict:
        return {
            "raw_dim": self.raw_dim,
            "channels": [
                {"name": c.name, "indices": c.indices,
                 "transform": c.transform, "dim": c.dim}
                for c in self.channels
            ],
        }

    def load_state_dict(self, sd: dict):
        self.raw_dim = sd.get("raw_dim", 4)
        self.channels = []
        for cs in sd.get("channels", []):
            ch = FeatureChannel(cs["name"], cs["indices"],
                                cs.get("transform", "identity"),
                                cs.get("dim", 1))
            self.channels.append(ch)
