"""
跨域迁移（Cross Domain Transfer）— 能力从源域迁移到目标域

原理⑨（生长/进化的路径）：能力不是每域重学，而是跨域复用——
迷宫学的"时序预测 + 策略模式"迁移到新域（如自由觅食/资源采集）。

设计：
  - DomainAdapter：域抽象（观测维度映射 + 动作映射）——源域与目标域的
    观测/动作空间差异由 adapter 桥接
  - CapabilityExtractor：从源域模型提取通用能力（底层权重 = 时序动力学特征）
  - transfer()：源域底层权重 → 目标域（顶层按目标域维度新初始化）
  - 少样本微调：迁移后在目标域少量样本上微调顶层

验证（test_cross_domain.py）：
  A. 零样本迁移：迷宫源域 → 自由域，能力保留（不从头学）
  B. 少样本微调：迁移 + 少量目标样本 → 快速收敛（对比从头学）
  C. 迁移 vs 从头：同训练预算，迁移学习更快（能力复用证据）
"""

import numpy as np


class DomainAdapter:
    """域抽象：源域 ↔ 目标域的观测/动作映射"""
    def __init__(self, src_obs_dim, tgt_obs_dim, src_n_actions, tgt_n_actions):
        self.src_obs_dim = src_obs_dim
        self.tgt_obs_dim = tgt_obs_dim
        self.src_n_actions = src_n_actions
        self.tgt_n_actions = tgt_n_actions

    def project_obs(self, obs: np.ndarray, from_src: bool = True) -> np.ndarray:
        """观测投影：源域观测 ↔ 目标域观测（填充/截断）"""
        if from_src:
            d = self.tgt_obs_dim
            src = np.asarray(obs, dtype=float)
        else:
            d = self.src_obs_dim
            src = np.asarray(obs, dtype=float)
        if len(src) >= d:
            return src[:d].astype(np.float32)
        return np.pad(src, (0, d - len(src))).astype(np.float32)

    def map_action(self, action: int, to_src: bool = False) -> int:
        """动作映射：目标域动作 ↔ 源域动作（取模/截断到公共空间）"""
        if to_src:
            return action % self.src_n_actions
        return action % self.tgt_n_actions


class CapabilityExtractor:
    """能力提取：从源域模型提取通用特征（底层权重向量化）"""
    def __init__(self, src_weights: dict):
        self.src_weights = src_weights

    def extract_bottom(self, layer_names=("W_x", "W_h", "encoder")) -> dict:
        """提取底层（通用时序动力学）权重——这些是域无关的"""
        bottom = {}
        for k, v in self.src_weights.items():
            if any(layer in k for layer in layer_names):
                bottom[k] = v
        return bottom

    def bottom_vector(self) -> np.ndarray:
        """底层权重展平向量（用于相似度对比）"""
        parts = []
        for v in self.extract_bottom().values():
            parts.append(np.asarray(v, dtype=float).flatten())
        if not parts:
            return np.zeros(1)
        return np.concatenate(parts)


class CrossDomainTransfer:
    """跨域迁移引擎：源域能力 → 目标域初始化 + 少样本微调"""
    def __init__(self, adapter: DomainAdapter, extractor: CapabilityExtractor):
        self.adapter = adapter
        self.extractor = extractor
        self.transfer_log = {}

    def transfer(self, tgt_model_weights: dict) -> dict:
        """把源域底层权重注入目标域（顶层保留目标域初始化）"""
        bottom = self.extractor.extract_bottom()
        tgt = dict(tgt_model_weights)
        transferred = 0
        for k, v in bottom.items():
            if k in tgt and tgt[k].shape == v.shape:
                tgt[k] = np.array(v, dtype=np.float32)
                transferred += 1
        self.transfer_log = {
            "transferred_layers": transferred,
            "bottom_used": list(bottom.keys())[:5],
        }
        return tgt

    def few_shot_finetune(self, model_weights: dict, samples: list,
                          lr: float = 0.01, epochs: int = 50) -> dict:
        """少样本微调：迁移后的底层作特征提取器，顶层在其输出上拟合
        这使迁移的底层（时序动力学先验）真正参与——而非仅初始化"""
        w = dict(model_weights)
        if not samples:
            return w
        xs = np.array([s[0] for s in samples], dtype=float)
        ys = np.array([s[1] for s in samples], dtype=float)
        # 用迁移后的底层（W_h 若存在）做特征投影
        features = xs
        if "W_h" in w:
            Wh = np.asarray(w["W_h"], dtype=float)
            # 投影到隐藏空间 + tanh 激活（非线性特征——像 LNN 隐藏层）
            d = min(Wh.shape[0], xs.shape[1])
            if d >= 2:
                proj = xs[:, :d] @ Wh[:d, :min(Wh.shape[1], 16)]
                features = np.tanh(proj)
        X = np.hstack([features, np.ones((len(xs), 1))])
        try:
            theta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        except np.linalg.LinAlgError:
            return w
        w["_top_adapter"] = theta
        w["_feat_dim"] = features.shape[1]
        return w

    def predict(self, w: dict, obs: np.ndarray) -> np.ndarray:
        """用迁移后的权重预测下一步（底层特征 + 顶层适配器）"""
        if "_top_adapter" in w:
            feat = obs
            if "W_h" in w:
                Wh = np.asarray(w["W_h"], dtype=float)
                d = min(Wh.shape[0], len(obs))
                if d >= 2:
                    proj = obs[:d] @ Wh[:d, :min(Wh.shape[1], 16)]
                    feat = np.tanh(proj)
            x = np.hstack([feat, [1.0]])
            return x @ w["_top_adapter"]
        return obs  # 无适配器→恒等（未迁移）

    def similarity(self, w1: dict, w2: dict) -> float:
        """两模型权重相似度（余弦）——按公共层名对齐，维度不同的层跳过"""
        def _vec(w):
            parts = []
            for k, v in w.items():
                if k.startswith("_") or not hasattr(v, "shape"):
                    continue
                # 只取两模型都有的层（按调用方传入的公共键）
                parts.append(np.asarray(v, dtype=float).flatten())
            return np.concatenate(parts) if parts else np.zeros(1)

        common_keys = [k for k in w1 if k in w2 and not k.startswith("_")]
        parts1, parts2 = [], []
        for k in common_keys:
            v1, v2 = np.asarray(w1[k], dtype=float), np.asarray(w2[k], dtype=float)
            if v1.shape != v2.shape:
                continue
            parts1.append(v1.flatten())
            parts2.append(v2.flatten())
        if not parts1:
            return 0.0
        a = np.concatenate(parts1)
        b = np.concatenate(parts2)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(a @ b / (na * nb))
