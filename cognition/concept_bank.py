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
        # ①+② 概念价值预测（DESIGN_CONCEPTS §3 阶段 2 前置）：
        # 概念伴随的 V 值 EMA——预测"这个可消耗物值多少"
        self.value_ema = 0.5      # 初始中性（V∈[0,1]）
        self.value_count = 0
        self.value_history = deque(maxlen=20)
        self.symbols = []         # 阶段 3：绑定到本概念的词（符号化）

    def predict_value(self) -> float:
        """预测该概念的价值（出现时 V 的 EMA）——引导条件用"""
        return self.value_ema

    def update_value(self, v: float):
        """记录该概念出现时的 V 值（在线 EMA，α=0.2）
        security LOW 修复：None 防护下沉 API（原仅调用侧防护——
        float(None) 会 TypeError）"""
        if v is None:
            return
        self.value_count += 1
        self.value_history.append(v)
        self.value_ema = 0.8 * self.value_ema + 0.2 * float(v)
    
    def __repr__(self):
        return f"[Concept {self.kind}:{self.name} f={self.freq} v={self.value_ema:.2f}]"

    # ─── 阶段 3：符号化（词↔概念绑定——"语言是符号压缩"落地）───
    def bind_symbol(self, symbol: str):
        """绑定一个语言符号（词）到本概念——共现学习（Hebbian）：
        概念激活时同时出现的词 → 绑定。重复绑定幂等。
        review nit：None 显式防护（str(None) 会绑定 "None" 字符串）"""
        if symbol is None:
            return
        s = str(symbol).strip().lower()
        if s and s not in self.symbols:
            self.symbols.append(s)

    def activate_by_symbol(self, symbol: str) -> bool:
        """符号激活：词是否绑定本概念（听到词→概念被激活）"""
        s = str(symbol).strip().lower()
        return bool(s and s in self.symbols)


class ConceptBank:
    """概念库：从经验中提取 + 组合生成"""
    def __init__(self, max_concepts: int = 50):
        self.concepts = []
        self.max_concepts = max_concepts
        self.combo_history = deque(maxlen=100)
        # ① 阶段 4：抽象组（概念图——抽象名 → 子概念类型组）
        self.abstracts = {"consumable": ["consumable"]}  # 默认注册
    
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
    def add_value_anchored(self, obs: np.ndarray, v_up: bool,
                           v: float = None) -> str:
        """价值锚聚类：V 上升事件更新"可消耗物"概念簇。
        概念 = 观测簇 × 价值绑定——只有 V 上升的观测进簇。
        返回概念名（"consumable_N"）或空串。
        v：出现时的身体价值（用于概念价值预测——①+②）。
        验证见 test_concept_value.py（跨形态泛化成立）。"""
        if not v_up or obs is None or len(obs) == 0:
            return ""
        vec = np.asarray(obs, dtype=np.float32).flatten()
        # should-fix：维度防护——质心与 obs 维度不匹配时跳过
        # （P4 观测增长机制会让 obs 维度变化——否则 ValueError 被吞
        #  且概念层静默失效每 tick 刷 WARN）
        dim = vec.shape[0]
        best_i, best_d = -1, 1e9
        for i, c in enumerate(self.concepts):
            if c.kind == "consumable" and c.vector.shape[0] == dim:
                d = float(np.linalg.norm(c.vector - vec))
                if d < best_d:
                    best_i, best_d = i, d
        if best_i >= 0 and best_d < 1.5:
            # 更新质心（在线 k-means）+ 增强频率 + 价值 EMA
            self.concepts[best_i].vector = (0.9 * self.concepts[best_i].vector
                                            + 0.1 * vec)
            self.concepts[best_i].freq += 1
            if v is not None:
                self.concepts[best_i].update_value(v)
            return self.concepts[best_i].name
        if len(self.concepts) < self.max_concepts:
            name = f"consumable_{len(self.concepts)}"
            c = Concept(name, "consumable", vec.copy())
            if v is not None:
                c.update_value(v)
            self.concepts.append(c)
            return name
        return ""
    
    # ─── 概念驱动行为（DESIGN_CONCEPTS §3 阶段 2 前置：概念→行为）───
    def match_concept(self, obs: np.ndarray, kind: str = "consumable",
                      threshold: float = 1.5):
        """观测→概念匹配：找与 obs 最相似的概念（欧氏距离，与
        add_value_anchored 同度量）。返回 (name, dist, matched,
        value_pred)——匹配则 matched=True；value_pred=该概念的价值
        预测（①+②：预测驱动引导——"像 + 预测值高"才停留）。"""
        if obs is None or len(obs) == 0:
            return "", 1e9, False, 0.0
        vec = np.asarray(obs, dtype=np.float32).flatten()
        dim = vec.shape[0]
        best_i, best_d = -1, 1e9
        for i, c in enumerate(self.concepts):
            if c.kind == kind and c.vector.shape[0] == dim:
                d = float(np.linalg.norm(c.vector - vec))
                if d < best_d:
                    best_i, best_d = i, d
        if best_i >= 0 and best_d < threshold:
            return (self.concepts[best_i].name, best_d, True,
                    self.concepts[best_i].predict_value())
        # review warn：不匹配分支必须同 4 元组契约（原 3 元组——
        # main.py matched[3] 会 IndexError）
        return "", best_d, False, 0.0

    # ─── 阶段 3：符号化（词↔概念绑定）───
    def bind_symbols(self, concept_name: str, symbols: list):
        """概念绑定一组词（共现学习——概念激活时听到的词）。
        幂等；未知概念名忽略。security LOW：symbols None/非列表防护。"""
        if not symbols:
            return False
        for c in self.concepts:
            if c.name == concept_name:
                for s in symbols:
                    c.bind_symbol(s)
                return True
        return False

    def activate_by_symbol(self, symbol: str):
        """符号→概念：听到词返回绑定的概念 (name, value_pred, found)。
        found=True 表示词有"所指"（绑定过概念）——语言接地。
        security LOW：None 显式防护（与 Concept 级对称）。"""
        if symbol is None:
            return "", 0.0, False
        s = str(symbol).strip().lower()
        if not s:
            return "", 0.0, False
        for c in self.concepts:
            if c.activate_by_symbol(s):
                return c.name, c.predict_value(), True
        return "", 0.0, False

    # ─── 语言生成（①——概念→词：看到概念→说出绑定符号）───
    def speak(self, obs: np.ndarray, kind: str = "consumable",
              threshold: float = 1.5):
        """说：当前观测匹配概念 → 输出绑定符号（词）。
        返回 (word, concept_name, spoke)——spoke=True 表示
        "有可说"（概念有绑定词）；无匹配/无绑定词 → 沉默。
        语言生成方向：概念激活→符号输出（与 activate_by_symbol 反向）。"""
        _, _, matched, _ = self.match_concept(obs, kind=kind,
                                              threshold=threshold)
        if not matched:
            return "", "", False
        # 找匹配概念（复用 match 逻辑拿概念对象）
        vec = np.asarray(obs, dtype=np.float32).flatten()
        dim = vec.shape[0]
        best_i, best_d = -1, 1e9
        for i, c in enumerate(self.concepts):
            if c.kind == kind and c.vector.shape[0] == dim:
                d = float(np.linalg.norm(c.vector - vec))
                if d < best_d:
                    best_i, best_d = i, d
        if best_i >= 0 and best_d < threshold:
            c = self.concepts[best_i]
            if c.symbols:
                return c.symbols[0], c.name, True
        return "", "", False

    def add_abstract_group(self, name: str, kinds: list):
        """注册抽象组：抽象名 → 子概念类型组（如
        "consumable" → ["consumable"]——食物/水源共享"可消耗物"
        抽象）。抽象 = 价值模式共享的子概念集合。
        review nit：kinds None 防护（list(None) TypeError）。"""
        if not hasattr(self, 'abstracts'):
            self.abstracts = {}
        self.abstracts[name] = list(kinds) if kinds else []

    def match_abstract(self, obs: np.ndarray, abstract_name: str,
                       threshold: float = 1.5):
        """抽象匹配：任一子概念类型匹配 → 抽象激活。
        返回 (matched, value_pred)——跨子概念泛化
        （"可消耗物"抽象：食物簇或水源簇观测都激活）。"""
        kinds = getattr(self, 'abstracts', {}).get(abstract_name, [])
        if not kinds:
            return False, 0.0
        for kind in kinds:
            _, _, m, vp = self.match_concept(obs, kind=kind,
                                             threshold=threshold)
            if m:
                return True, vp
        return False, 0.0

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
