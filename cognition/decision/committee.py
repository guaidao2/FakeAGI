"""
皮层决策委员会 — 人脑式并行决策仲裁

人脑决策不是单一模块，而是多个并行系统竞争 + 前额叶执行控制：

  基底节（习惯）   —— 快速、自动化的 Q 值动作提议
  边缘系统（情感） —— 驱动力驱动的动作偏好（饥饿→觅食）
  前额叶（规划）   —— 多步前瞻，预期效用评估（慢、灵活）
  元认知（监控）   —— 知识缺口检测 → 目标重定向
  反射（本能）     —— 硬接线映射（安全/危急时主导）

仲裁机制（前额叶执行控制）：
  1. 每个决策者对每个动作投票（支持度向量）
  2. 情境权重：由当前主导驱动力/危机程度决定各系统权重
  3. 加权求和 → 选择支持度最高的动作
  4. 冲突检测：前两名得分接近 → 深思模式（提升规划权重、压低反射权重）
  5. 恐慌模式：健康极低 → 反射/边缘权重飙升，规划权重归零
"""

import numpy as np


class DecisionCommittee:
    def __init__(self, n_actions: int = 5):
        self.n_actions = n_actions
        # 决策者权重（情境动态调整）
        self.weights = {
            "reflex": 0.35,     # 反射/本能
            "limbic": 0.25,     # 边缘/驱动力
            "habit": 0.25,      # 习惯/GameNN
            "plan": 0.10,       # 规划/前额叶
            "meta": 0.05,       # 元认知
        }
        self.last_votes = {}
        self.conflict_mode = False   # 深思模式
        self.panic_mode = False      # 恐慌模式
        self.exploration_ratio = 0.1
        
    def reflex_vote(self, obs: np.ndarray, drive_bias: np.ndarray,
                    body_state: dict, secondary_reached: bool = False) -> np.ndarray:
        """反射/本能投票：朝向主要目标（obs[0:2]），危急时强化"""
        vote = np.zeros(self.n_actions)
        if len(obs) >= 2:
            dx, dy = obs[0], obs[1]
            hungry = body_state.get("energy", 1.0) < 0.8
            thirsty = body_state.get("water", 1.0) < 0.6
            if (hungry or thirsty or secondary_reached) and (abs(dx) > 0.05 or abs(dy) > 0.05):
                if abs(dx) > abs(dy):
                    a = 3 if dx > 0 else 2
                else:
                    a = 4 if dy > 0 else 1
                vote[a] = 1.0
        return vote
    
    def limbic_vote(self, drive_bias: np.ndarray) -> np.ndarray:
        """边缘系统投票：驱动力偏置直接映射为动作支持"""
        vote = np.zeros(self.n_actions)
        if drive_bias is not None and len(drive_bias) >= self.n_actions:
            vote = np.clip(drive_bias[:self.n_actions], 0, 1).astype(float)
        return vote
    
    def habit_vote(self, gamenn_probs: np.ndarray) -> np.ndarray:
        """习惯/GameNN 投票：Q 值 softmax 概率"""
        vote = np.zeros(self.n_actions)
        if gamenn_probs is not None and len(gamenn_probs) == self.n_actions:
            vote = np.asarray(gamenn_probs, dtype=float)
        return vote
    
    def plan_vote(self, plan_scores: np.ndarray) -> np.ndarray:
        """前额叶/规划投票：前瞻模拟的动作评分"""
        vote = np.zeros(self.n_actions)
        if plan_scores is not None and len(plan_scores) == self.n_actions:
            vote = np.asarray(plan_scores, dtype=float)
        return vote
    
    def meta_vote(self, meta_action: int) -> np.ndarray:
        """元认知投票：知识缺口重定向"""
        vote = np.zeros(self.n_actions)
        if meta_action is not None and 0 <= meta_action < self.n_actions:
            vote[meta_action] = 1.0
        return vote
    
    def compute_weights(self, health: float, stress: float,
                        confidence: float, energy: float) -> dict:
        """情境权重：由危机程度和置信度动态调整"""
        w = dict(self.weights)
        
        # 恐慌模式：健康极低 + 应激高 → 反射/边缘主导，规划归零
        self.panic_mode = health < 0.3 and stress > 0.5
        if self.panic_mode:
            w["reflex"] = 0.65
            w["limbic"] = 0.30
            w["habit"] = 0.05
            w["plan"] = 0.0
            w["meta"] = 0.0
            return w
        
        # 深思模式：置信度低（学习期）或冲突 → 规划/元认知权重提升
        self.conflict_mode = confidence < 0.15
        if self.conflict_mode:
            w["reflex"] = 0.25
            w["limbic"] = 0.20
            w["habit"] = 0.20
            w["plan"] = 0.20
            w["meta"] = 0.15
            return w
        
        # 能量低：边缘系统（驱动力）权重上升
        if energy < 0.4:
            w["limbic"] += 0.15
            w["reflex"] += 0.10
            w["plan"] -= 0.10
            w["meta"] -= 0.05
        
        # 归一化
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}
    
    def decide(self, votes: dict, health: float, stress: float,
               confidence: float, energy: float,
               exploration_ratio: float = 0.1) -> dict:
        """
        加权仲裁：
        - 每个决策者投票 → 加权求和 → argmax
        - 探索：以 exploration_ratio 概率随机选择
        - 冲突检测：前两名接近 → 报告 conflict（供外部深思）
        """
        self.exploration_ratio = exploration_ratio
        w = self.compute_weights(health, stress, confidence, energy)
        self.last_votes = {k: v.tolist() for k, v in votes.items() if v is not None}
        
        # 加权求和
        total = np.zeros(self.n_actions)
        for name, vote in votes.items():
            if vote is not None and name in w:
                total += w[name] * vote
        
        # 冲突检测：前两名差距 < 10% → 冲突
        sorted_idx = np.argsort(total)[::-1]
        if len(sorted_idx) >= 2:
            gap = total[sorted_idx[0]] - total[sorted_idx[1]]
            self.conflict_mode = gap < 0.1 * max(1.0, total[sorted_idx[0]])
        else:
            self.conflict_mode = False
        
        # 探索
        if np.random.random() < exploration_ratio and not self.panic_mode:
            action = np.random.randint(0, self.n_actions - 1)  # 排除睡眠
        else:
            action = int(sorted_idx[0])
        
        return {
            "action": action,
            "weights": w,
            "conflict": self.conflict_mode,
            "panic": self.panic_mode,
            "votes": self.last_votes,
            "scores": total.tolist(),
        }
    
    def get_state(self) -> dict:
        return {
            "conflict_mode": self.conflict_mode,
            "panic_mode": self.panic_mode,
            "weights": dict(self.weights),
            "last_votes": self.last_votes,
        }
