"""
组合式反事实 — 升级想象通道

原版想象通道：把经验中的动作替换成另一个已知动作（轻微变异）。
升级版：组合已有概念创造新场景（真正的反事实）。

人类能做到：
- "如果这个世界没有水，生物会怎样？"（概念组合）
- "如果我把钥匙给狗而不是拿在手里呢？"（角色替换）

实现：
- ConceptBank：概念库（从经验中提取的抽象概念：物体、位置、动作、状态）
- 组合生成：从概念库抽取 2-3 个概念，组合成新场景描述
- 新场景通过世界模型模拟 → 生成"从未发生的经验" → 加入训练数据
"""

import numpy as np
from collections import deque


class Concept:
    """一个抽象概念（从经验中归纳）"""
    def __init__(self, name: str, kind: str, vector: np.ndarray, freq: int = 1):
        self.name = name          # 概念名（如 "switch", "food", "danger"）
        self.kind = kind          # 类型（object/place/action/state）
        self.vector = vector      # 概念向量（连续表示）
        self.freq = freq          # 出现频率
    
    def __repr__(self):
        return f"[Concept {self.kind}:{self.name} f={self.freq}]"


class ConceptBank:
    """概念库：从经验中提取 + 组合生成"""
    def __init__(self, max_concepts: int = 50):
        self.concepts = []
        self.max_concepts = max_concepts
        self.combo_history = deque(maxlen=100)
    
    def extract_from_obs(self, obs: np.ndarray, action: int, result: dict) -> None:
        """从一次经验中提取概念"""
        # 观测中的显著特征（方差大 = 重要）
        if len(obs) > 0:
            salient = np.argsort(np.abs(obs))[-2:]  # 最显著的 2 维
            for idx in salient:
                name = f"obs_{idx}"
                self._add_or_boost(Concept(name, "feature", np.array([obs[idx]])))
        
        # 动作作为概念
        self._add_or_boost(Concept(f"act_{action}", "action",
                                   np.array([action])))
        
        # 结果作为概念
        energy_gain = result.get("energy_delta", 0)
        if energy_gain > 0.05:
            self._add_or_boost(Concept("reward_positive", "state",
                                       np.array([energy_gain])))
        elif energy_gain < -0.02:
            self._add_or_boost(Concept("reward_negative", "state",
                                       np.array([energy_gain])))
    
    def _add_or_boost(self, concept: Concept):
        for c in self.concepts:
            if c.name == concept.name and c.kind == concept.kind:
                c.freq += 1
                return
        if len(self.concepts) < self.max_concepts:
            self.concepts.append(concept)

    # ─── 概念层接入（DESIGN_CONCEPTS §3 阶段 1：价值锚聚类）───
    def add_value_anchored(self, obs: np.ndarray, v_up: bool) -> str:
        """价值锚聚类：V 上升事件更新"可消耗物"概念簇。
        概念 = 观测簇 × 价值绑定——只有 V 上升的观测进簇。
        返回概念名（"consumable_N"）或空串。
        验证见 test_concept_value.py（跨形态泛化成立）。"""
        if not v_up or obs is None or len(obs) == 0:
            return ""
        vec = np.asarray(obs, dtype=np.float32).flatten()
        # 找/建"consumable"簇（价值锚聚类质心）
        best_i, best_d = -1, 1e9
        for i, c in enumerate(self.concepts):
            if c.kind == "consumable":
                d = float(np.linalg.norm(c.vector - vec))
                if d < best_d:
                    best_i, best_d = i, d
        if best_i >= 0 and best_d < 1.5:
            # 更新质心（在线 k-means）+ 增强频率
            self.concepts[best_i].vector = (0.9 * self.concepts[best_i].vector
                                            + 0.1 * vec)
            self.concepts[best_i].freq += 1
            return self.concepts[best_i].name
        if len(self.concepts) < self.max_concepts:
            name = f"consumable_{len(self.concepts)}"
            self.concepts.append(Concept(name, "consumable", vec.copy()))
            return name
        return ""
    
    def generate_combo(self, n: int = 3) -> list:
        """
        组合式反事实生成：
        随机抽取 n 个概念组合成一个"假设场景"。
        返回概念列表。
        """
        if len(self.concepts) < 2:
            return []
        # 按频率加权采样
        freqs = np.array([c.freq for c in self.concepts], dtype=np.float32)
        probs = freqs / np.sum(freqs)
        idx = np.random.choice(len(self.concepts), size=min(n, len(self.concepts)),
                               replace=False, p=probs)
        combo = [self.concepts[i] for i in idx]
        # 记录组合（去重）
        combo_key = tuple(sorted([c.name for c in combo]))
        if combo_key not in self.combo_history:
            self.combo_history.append(combo_key)
        return combo
    
    def combo_to_scenario(self, combo: list) -> str:
        """把概念组合转成场景描述（供日志/语言输出）"""
        names = [c.name for c in combo]
        kinds = [c.kind for c in combo]
        return f"假设: 如果{' + '.join(names)} ({'+'.join(kinds)})"
    
    def get_stats(self) -> dict:
        return {
            "concept_count": len(self.concepts),
            "combo_count": len(self.combo_history),
            "top_concepts": [c.name for c in sorted(
                self.concepts, key=lambda x: -x.freq)[:5]]
        }
