"""
他者可靠性建模（社会智能核心——信任绑定在说话者而非词）

社会智能 = 对"说话者"可靠性建模（"他说 west 但上次他错了"→
降对"他"的信任——盲从不是协作，是控制）。

SpeakerTrust：每个说话者的可靠性估计——
- 成功（语言引导后吃到）→ trust +0.1（慢升）
- 失败（语言引导后没吃到）→ trust -0.3（快降——质疑能力）
- 低信任说话者的词 → 语言权重打折（不被误导）
"""
import numpy as np


class SpeakerTrust:
    def __init__(self, init_trust: float = 0.5):
        self.trusts = {}          # speaker_id -> trust
        self.history = {}         # speaker_id -> (success, fail)
        self.init_trust = init_trust

    def get(self, speaker_id: str) -> float:
        if speaker_id is None:
            return 0.0            # 无说话者 → 无信任（不可信）
        return self.trusts.get(speaker_id, self.init_trust)

    def observe_outcome(self, speaker_id: str, success: bool):
        """观察说话者的指导结果：成功+0.1（慢升），失败-0.3（快降）"""
        if speaker_id is None:
            return
        t = self.get(speaker_id)
        if success:
            t = min(1.0, t + 0.1)
        else:
            t = max(0.0, t - 0.3)   # 大幅降信——一次误导重罚
        self.trusts[speaker_id] = t
        h = self.history.setdefault(speaker_id, [0, 0])
        h[0 if success else 1] += 1

    def weight(self, speaker_id: str) -> float:
        """说话者的语言权重（trust→[0.1, 1.0]——低信任打折但不断绝）"""
        t = self.get(speaker_id)
        return 0.1 + 0.9 * t

    def summary(self) -> dict:
        return {k: (round(v, 2), self.history.get(k, [0, 0]))
                for k, v in self.trusts.items()}
