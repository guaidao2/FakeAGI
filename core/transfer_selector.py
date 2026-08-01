"""
迁移价值评估器（Transferability Selector）— "何时能迁移"的元认知

⑤ 揭示的真正门槛：关系映射是否同构决定迁移成败——
人类触类旁通前会"犹豫"（这个新东西和熟的像不像）。
本模块把这种犹豫工程化：

  - TransferabilityEstimator：对"迁移"动作做可靠性估计（复用 C2 叠加态
    模式——多假设分支：迁移有用/无用/有害，贝叶斯坍缩，不单点信任）
  - TransferSelector：遇到新域时选择"迁移"或"从头学"（argmax 预期收益）
  - 更新信号：迁移后的少样本性能 vs 从头学性能（真实反馈——
    同构域迁移好→可靠性升；异构域迁移差→可靠性降）

验证（test_transfer_selector.py）：
  A. 同构域（迷宫→威胁场）迁移后可靠性升
  B. 异构域（迷宫→逆场：威胁=奖励）迁移后可靠性降
  C. 未知域：系统按估计选择正确（同构选迁移、异构选从头）
  D. 对比：有评估器 vs 无脑迁移（无脑在异构域被拖累）
"""

import numpy as np


class TransferabilityEstimator:
    """迁移可靠性估计（叠加态模式：多假设分支 + 贝叶斯坍缩）"""
    def __init__(self, prior: float = 0.5,
                 hypotheses=(0.8, 0.5, 0.2), frozen: bool = False):
        self.prior = prior
        self.hypotheses = np.array(hypotheses, dtype=float)
        dist = np.abs(self.hypotheses - prior)
        self.weights = np.exp(-dist * 4.0)
        self.weights /= self.weights.sum()
        self.frozen = frozen
        self.outcomes = []          # 每次迁移后 (migrated_perf, scratch_perf)
        self.wins = 0
        self.losses = 0

    @property
    def reliability(self) -> float:
        """迁移可靠性的加权期望（0-1：高=迁移更可能有用）"""
        return float(np.dot(self.weights, self.hypotheses))

    def update(self, migrated_perf: float, scratch_perf: float):
        """迁移后反馈：迁移性能 vs 从头学性能（同预算）
        幅度加权：性能差越大更新越强（赢 4.5 倍 vs 平局更新幅度不同）"""
        if self.frozen:
            return
        # NaN/Inf 防御：非有限性能输入静默按失败处理（防污染权重）
        if not (np.isfinite(migrated_perf) and np.isfinite(scratch_perf)):
            success = False
            strength = 0.05
        elif scratch_perf > 0:
            ratio = migrated_perf / scratch_perf
            # 连续强度函数：ratio=1 时 strength=0.4（迁移无优势=降），
            # 偏离 1 越远强度越大（赢越多升越强、输越多降越强）——无跳变
            strength = min(1.0, 0.4 + abs(ratio - 1.0) * 0.8)
            success = ratio > 1.0
        else:
            success = migrated_perf > 0
            strength = 0.3
        # 似然：假设 h 下观测到成功/失败的概率，按幅度加权
        lik = self.hypotheses if success else (1.0 - self.hypotheses)
        raw = self.weights * np.power(lik, strength)
        floor = max(raw) * 0.01   # 置信度地板（不单点信任，C2 先例）
        raw = np.maximum(raw, floor)
        s = raw.sum()
        if s > 1e-12:
            self.weights = raw / s
        else:
            self.weights = np.ones_like(self.weights) / len(self.weights)
        self.outcomes.append((migrated_perf, scratch_perf))
        if success:
            self.wins += 1
        else:
            self.losses += 1

    def get_state(self) -> dict:
        return {
            "reliability": round(self.reliability, 3),
            "wins": self.wins,
            "losses": self.losses,
            "frozen": self.frozen,
            "hypotheses": [round(float(h), 3) for h in self.hypotheses],
            "weights": [round(float(w), 3) for w in self.weights],
        }


class TransferSelector:
    """迁移选择器：新域到来时，按迁移可靠性决定"迁移 or 从头" """
    def __init__(self, min_reliability: float = 0.5):
        self.min_reliability = min_reliability  # 低于此→从头学（迁移不划算）
        self.estimator = TransferabilityEstimator()
        self.selected = None
        self.history = []

    def choose(self, domain_id: str) -> str:
        """对新域做选择：迁移 or 从头（argmax 迁移可靠性）"""
        if self.estimator.reliability >= self.min_reliability:
            self.selected = "transfer"
        else:
            self.selected = "scratch"
        self.history.append((domain_id, self.selected,
                             round(self.estimator.reliability, 3)))
        return self.selected

    def observe_feedback(self, migrated_perf: float, scratch_perf: float):
        """迁移后的真实反馈更新可靠性"""
        self.estimator.update(migrated_perf, scratch_perf)

    def get_state(self) -> dict:
        return {
            "selected": self.selected,
            "estimator": self.estimator.get_state(),
            "history": self.history[-10:],
        }
