"""
过程选择器（Process Selector）— 语言 × 目标层的统一

设计（DESIGN_PROCESS_SELECTION.md）：
  目标导向的行为选择：落差 → 世界模型评估各过程的预期落差消解率 → argmax

过程：
  - "ask"（问路）：语言通道，消耗 1 tick（机会成本，intrinsic），
    环境响应可能有噪声/失败率（C1）
  - "sweep"（扫掠）：InfoSeeker 定向搜索，有记忆系统性覆盖

可靠性估计（C2，防"被教歪"）：
  - 每个过程维护在线估计的"预期落差消解率"（0-1）
  - 失败（用了过程但落差未消解）→ 估计下降（在线更新，非固化）
  - 成功（落差消解）→ 估计上升
  - 这是标量版叠加态坍缩：估计 = 该过程"可靠"的后验概率

选择逻辑（C3，模型驱动非查表）：
  - 落差 > 阈值时，比较各过程预期收益 = 可靠性 × (1 - 成本)
  - 选 argmax；可靠性低于保底阈值时不再选择该过程（N2 场景）
"""

import numpy as np


class ProcessEstimator:
    """单个过程的可靠性估计（在线更新，非固化）"""
    def __init__(self, name: str, prior: float = 0.5,
                 lr_up: float = 0.1, lr_down: float = 0.15):
        self.name = name
        self.reliability = prior      # 预期落差消解率（0-1）
        self.prior = prior
        self.lr_up = lr_up
        self.lr_down = lr_down        # 失败更新更快（对称破缺：信任易碎）
        self.outcomes = []            # 每次使用的结果记录（True/False）
        self.frozen = False           # N1 负对照：冻结不更新

    def update(self, success: bool):
        """使用后更新：成功→升，失败→降（在线更新）"""
        if self.frozen:
            return
        if success:
            self.reliability = min(1.0, self.reliability + self.lr_up)
        else:
            self.reliability = max(0.0, self.reliability - self.lr_down)
        self.outcomes.append(success)
        if len(self.outcomes) > 100:
            self.outcomes.pop(0)

    def expected_gain(self, cost: float = 0.0) -> float:
        """预期收益 = 可靠性 × (1 - 成本)（intrinsic 成本：机会成本等）"""
        return self.reliability * (1.0 - cost)

    def freeze(self):
        """N1 负对照：锁死估计"""
        self.frozen = True

    def get_state(self) -> dict:
        return {
            "name": self.name,
            "reliability": round(self.reliability, 3),
            "frozen": self.frozen,
            "uses": len(self.outcomes),
            "recent_success_rate": (sum(self.outcomes[-20:]) / max(1, len(self.outcomes[-20:]))
                                    if self.outcomes else 0.0),
        }


class ProcessSelector:
    """过程选择器：评估各过程预期收益 → 选择"""

    def __init__(self, min_reliability: float = 0.15,
                 ask_cost: float = 0.15):
        self.min_reliability = min_reliability  # 低于此不再选择（N2 场景）
        self.ask_cost = ask_cost                # 问路机会成本（占 tick，intrinsic）
        self.sweep_cost = 0.05                  # 扫掠也有微小成本（移动能耗）
        self.estimators = {
            "ask": ProcessEstimator("ask", prior=0.5),
            "sweep": ProcessEstimator("sweep", prior=0.3),
        }
        self.selected = None
        self.selection_history = []   # (tick, selected, ask_rel, sweep_rel)

    def update_outcome(self, process: str, success: bool):
        """过程使用后更新可靠性（成功/失败在线修正）"""
        if process in self.estimators:
            self.estimators[process].update(success)

    def choose(self, gap: float, tick: int = 0) -> str:
        """落差驱动选择：评估各过程预期收益 → argmax
        可靠性低于阈值时以 3% 概率试探（防死锁——V3 可逆需要）
        返回 "ask" / "sweep" / "none"
        """
        if gap <= 0.15:
            self.selected = "none"
            return "none"
        best, best_gain = None, -1e9
        for name, est in self.estimators.items():
            if est.reliability < self.min_reliability:
                # 试探：低概率重新尝试（可靠性可逆的前提）
                if np.random.random() >= 0.05:
                    continue
            cost = self.ask_cost if name == "ask" else self.sweep_cost
            gain = est.expected_gain(cost)
            if gain > best_gain:
                best, best_gain = name, gain
        self.selected = best if best is not None else "none"
        self.selection_history.append(
            (tick, self.selected,
             self.estimators["ask"].reliability,
             self.estimators["sweep"].reliability))
        return self.selected

    def get_state(self) -> dict:
        return {
            "selected": self.selected,
            "ask": self.estimators["ask"].get_state(),
            "sweep": self.estimators["sweep"].get_state(),
        }
