"""
他者模型（真他者）— 共享环境中的独立智能体跟踪

原理⑦前置（他者模型开始）：系统跟踪"另一个体"的位置/行为模式/意图，
并基于他者行为调整自身策略（竞争：回避抢资源；合作：跟随共享）。
与 hemin.py（反事实影子）不同——这是真实独立实体。

设计（DESIGN_OTHER_AGENT.md）：
  - OtherAgent：共享环境中的独立实体（有自己的目标/移动策略）
  - OtherModel：主系统对他者的跟踪（位置记忆 + 行为模式估计 + 意图分类）
  - 意图分类：合作（共享资源区域）/ 竞争（独占资源区域）
  - 策略调整：竞争模式→回避他者当前位置；合作模式→跟随他者
"""

import numpy as np


class OtherAgent:
    """共享环境中的他者实体（真实独立智能体）"""
    def __init__(self, size=16, strategy="competitor", pos=None):
        self.size = size
        self.strategy = strategy        # competitor / cooperator / wanderer
        self.pos = list(pos) if pos else [np.random.randint(0, size),
                                          np.random.randint(0, size)]
        self.target = None
        self.history = []

    def choose_action(self, food_pos):
        """他者策略：竞争→朝食物；合作→绕行共享；游荡→随机"""
        if self.strategy == "competitor":
            # 直接朝食物走（抢）
            dx = food_pos[0] - self.pos[0]
            dy = food_pos[1] - self.pos[1]
            if abs(dx) > abs(dy):
                return 3 if dx > 0 else 2
            return 4 if dy > 0 else 1
        elif self.strategy == "cooperator":
            # 合作者：明确远离食物（不抢资源，让出觅食区）
            dx = food_pos[0] - self.pos[0]
            dy = food_pos[1] - self.pos[1]
            if abs(dx) > abs(dy):
                return 2 if dx > 0 else 3   # 朝反方向（远离食物）
            return 1 if dy > 0 else 4
        else:  # wanderer
            return np.random.randint(1, 5)

    def step(self, action):
        dirs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dirs[action % 5]
        self.pos[0] = max(0, min(self.size - 1, self.pos[0] + dx))
        self.pos[1] = max(0, min(self.size - 1, self.pos[1] + dy))
        self.history.append(tuple(self.pos))
        if len(self.history) > 200:
            self.history.pop(0)

    def near(self, pos, radius=2):
        return abs(self.pos[0] - pos[0]) + abs(self.pos[1] - pos[1]) <= radius


class OtherModel:
    """主系统对他者的跟踪（位置记忆 + 行为模式 + 意图分类）"""
    def __init__(self, window=100, conflict_radius=3):
        self.window = window
        self.conflict_radius = conflict_radius
        self.positions = []              # 他者位置历史
        self.conflict_count = 0          # 与他者竞争（同时近资源）次数
        self.coexist_count = 0           # 与他者和平共存次数
        self.last_seen = None
        self.intent = "unknown"          # competitor / cooperator / unknown
        self.intent_confidence = 0.0

    def observe(self, other_pos, my_pos, food_pos, tick):
        """每 tick 观察他者位置，更新意图估计
        冲突事件：他者进入食物半径（独占资源）——用事件计数判别意图"""
        self.last_seen = tuple(other_pos)
        self.positions.append(tuple(other_pos))
        if len(self.positions) > self.window:
            self.positions.pop(0)

        d_other_food = abs(other_pos[0] - food_pos[0]) + abs(other_pos[1] - food_pos[1])
        if d_other_food <= self.conflict_radius:
            self.conflict_count += 1
        else:
            self.coexist_count += 1

        # 意图分类：冲突占比（事件比例）高→competitor；低→cooperator
        total = self.conflict_count + self.coexist_count
        if total > 50:   # 需要足够样本（2000 tick 的 2.5%）
            ratio = self.conflict_count / total
            if ratio > 0.25:   # 他者 ≥25% 时间占资源 → 竞争者
                self.intent = "competitor"
                self.intent_confidence = min(1.0, (ratio - 0.25) * 2)
            else:
                self.intent = "cooperator"
                self.intent_confidence = min(1.0, (0.25 - ratio) * 2)

    def get_avoidance(self, my_pos):
        """竞争模式：回避他者当前位置（反向方向）"""
        if self.intent != "competitor" or self.last_seen is None:
            return None
        dx = my_pos[0] - self.last_seen[0]
        dy = my_pos[1] - self.last_seen[1]
        # 远离他者：选 dx/dy 主导方向
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2   # 右/左
        return 4 if dy > 0 else 1       # 下/上

    def get_follow(self, my_pos):
        """合作模式：跟随他者（朝向他者位置）"""
        if self.intent != "cooperator" or self.last_seen is None:
            return None
        dx = self.last_seen[0] - my_pos[0]
        dy = self.last_seen[1] - my_pos[1]
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        return 4 if dy > 0 else 1

    def get_state(self) -> dict:
        return {
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 3),
            "conflict_count": self.conflict_count,
            "coexist_count": self.coexist_count,
            "last_seen": self.last_seen,
        }
