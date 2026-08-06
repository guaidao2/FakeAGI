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
            "language": 0.30,   # 语言指令（主动请求的信息优先——意图）
            "concept": 0.20,    # 概念（动机选择器——③：身体经验压缩
                                # 的"关注选择器"；价值经委员会加权，
                                # 不直驱动作——护栏裁决 B 模式）
                                # 注：无 concept 票时（direct 模式）
                                # 该权重零参与——total 无归一化依赖，
                                # 零影响严格成立（decide 仅对有的票
                                # 加权求和，argmax/conflict 不受影响）
        }
        self.last_votes = {}
        self.conflict_mode = False   # 深思模式
        self.panic_mode = False      # 恐慌模式
        self.exploration_ratio = 0.1
        
    def reflex_vote(self, obs: np.ndarray, drive_bias: np.ndarray,
                    body_state: dict, secondary_reached: bool = False) -> np.ndarray:
        """反射/本能投票：朝向主要目标（obs[0:2] 食物 / obs[2:4] 水——按紧迫度），危急时强化"""
        vote = np.zeros(self.n_actions)
        if len(obs) >= 4:
            dx, dy = obs[0], obs[1]
            wx, wy = obs[2], obs[3]
            energy = body_state.get("energy", 1.0)
            water = body_state.get("water", 1.0)
            hungry = energy < 0.8
            thirsty = water < 0.6
            # 方向选择：口渴且（不饿 或 水比食物更危急）→ 朝水
            if thirsty and (not hungry or water < 0.3):
                dx, dy = wx, wy
            if (hungry or thirsty or secondary_reached) and (abs(dx) > 0.05 or abs(dy) > 0.05):
                if abs(dx) > abs(dy):
                    a = 3 if dx > 0 else 2
                else:
                    a = 4 if dy > 0 else 1
                vote[a] = 1.0
        return vote
    
    def limbic_vote(self, drive_bias: np.ndarray) -> np.ndarray:
        """边缘系统投票：驱动力偏置映射为动作支持（修正索引错位）
        drive_bias 格式: [up, left, right, down, sleep, explore]
        动作编号: 0=stay, 1=up, 2=left, 3=right, 4=down
        """
        vote = np.zeros(self.n_actions)
        if drive_bias is not None and len(drive_bias) >= 4:
            # 重映射: up→1, left→2, right→3, down→4
            vote[1] = drive_bias[0]  # up
            vote[2] = drive_bias[1]  # left
            vote[3] = drive_bias[2]  # right
            vote[4] = drive_bias[3]  # down
        return vote

    def language_vote(self, language_dir: int, trust: float = 0.5) -> np.ndarray:
        """语言指令投票：方向词（east/west/north/south）→ 动作偏好
        主动请求的信息 = 指令级（幅值固定 1.0，与反射同级）。
        trust 只控制是否投票（假线索坍缩后不再投票），不削弱指令强度。
        方向词映射：east→3(右), west→2(左), north→1(上), south→4(下)
        """
        vote = np.zeros(self.n_actions)
        if language_dir is not None and 1 <= language_dir <= 4:
            vote[language_dir] = 1.0  # 指令级强度
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
                        confidence: float, energy: float,
                        concept_active: bool = False) -> dict:
        """情境权重：由危机程度和置信度动态调整
        concept_active：本 tick 是否有 concept 票——无票时从分母剔除
        concept 权重（direct 模式零影响严格成立：归一化分母恢复
        1.30，argmax/conflict/language 加成全部严格复原）"""
        w = dict(self.weights)
        if not concept_active:
            w.pop("concept", None)  # review should-fix：分母剔除
        
        # 恐慌模式：健康极低 + 应激高 → 反射/边缘主导，规划/语言归零
        self.panic_mode = health < 0.3 and stress > 0.5
        if self.panic_mode:
            w["reflex"] = 0.65
            w["limbic"] = 0.30
            w["habit"] = 0.05
            w["plan"] = 0.0
            w["meta"] = 0.0
            w["language"] = 0.0  # 危机时不听词（保命优先）
            w["concept"] = 0.0   # 恐慌时不听概念（同语言——保命优先）
            # security LOW：归一化（原靠常量巧合恰为 1.0——未来调
            # panic 权重会破坏尺度）
            total = sum(w.values())
            return {k: v / total for k, v in w.items()}
        
        # 深思模式：置信度低（学习期）或冲突 → 规划/元认知权重提升
        self.conflict_mode = confidence < 0.15
        if self.conflict_mode:
            w["reflex"] = 0.25
            w["limbic"] = 0.20
            w["habit"] = 0.20
            w["plan"] = 0.20
            w["meta"] = 0.15
            w["language"] = 0.10  # 深思时语言权重压低（多路证据仲裁）
            # 归一化（与正常/恐慌分支尺度一致）
            total = sum(w.values())
            return {k: v / total for k, v in w.items()}
        
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
        w = self.compute_weights(health, stress, confidence, energy,
                                 concept_active=("concept" in votes
                                                 and votes["concept"] is not None))
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

        # P8b 意图优先：主动请求的语言指令（方向词）压过本能反射
        # 生物对应：问路后按指示走（前额叶指令 > 本能习惯）
        # 恐慌模式例外：危机时不听词（保命优先，与权重归零一致）
        if ("language" in votes and votes["language"] is not None
                and not self.panic_mode):
            lang_action = int(np.argmax(votes["language"]))
            if votes["language"][lang_action] > 0:
                total[lang_action] += 0.6  # 指令加成（冲突时语言胜出）
                self.last_votes["language"] = votes["language"].tolist()
                # 加成后重算排序与冲突标志（否则加成被忽略）
                sorted_idx = np.argsort(total)[::-1]
                if len(sorted_idx) >= 2:
                    gap = total[sorted_idx[0]] - total[sorted_idx[1]]
                    self.conflict_mode = gap < 0.1 * max(1.0, total[sorted_idx[0]])
        
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
