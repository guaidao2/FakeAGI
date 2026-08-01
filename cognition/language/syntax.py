"""
P8c 句法层（Syntax Layer）— 词序列→短语组合

语言从词级→句级：不是硬编码语法规则，而是从经验学词序模式：
  - BigramTracker：统计相邻词对的出现频率（"food east" 常见 = 该顺序合法）
  - PhraseBuilder：用模板 slot 组合多词输出（[resource] [direction]），
    顺序由经验统计决定（哪个顺序更常见就输出哪个）
  - 与 LinguisticOrgan.select_words 衔接：多词输出替代单词

验证（test_syntax.py）：
  A. 词序学习：高频 bigram 被识别为合法顺序
  B. 短语生成：按经验顺序组词（不是固定模板）
  C. 顺序反转：经验变化后输出顺序跟着变（可学习语法）
  D. 与 select_words 集成：多词输出（n_words>1 按句法排序）
"""

import numpy as np
from collections import Counter


class BigramTracker:
    """相邻词对频率统计（从听到/说出的序列学习词序）"""
    def __init__(self, vocab: list):
        self.vocab = list(vocab)
        self.bigram_counts = Counter()   # (w1, w2) -> count

    def observe_sequence(self, words: list):
        """记录一个词序列的所有相邻对"""
        for i in range(len(words) - 1):
            self.bigram_counts[(words[i], words[i + 1])] += 1

    def order_score(self, w1: str, w2: str) -> float:
        """w1 在 w2 前的得分（归一化：0-1，>0.5 表示 w1-w2 更常见）"""
        fwd = self.bigram_counts.get((w1, w2), 0)
        bwd = self.bigram_counts.get((w2, w1), 0)
        total = fwd + bwd
        if total == 0:
            return 0.5  # 未见→中性
        return fwd / total

    def top_pairs(self, n=10):
        return self.bigram_counts.most_common(n)


class PhraseBuilder:
    """短语组合：模板 slot 填充，顺序由经验统计决定"""
    def __init__(self, vocab: list, slots=("resource", "direction")):
        self.slots = slots
        self.tracker = BigramTracker(vocab)

    def learn(self, sequences: list):
        """从听到的短语序列学习词序"""
        for seq in sequences:
            self.tracker.observe_sequence(seq)

    def build(self, words: list, n_words: int = 2) -> list:
        """把候选词排成短语：贪心按 bigram 得分排序
        返回按经验顺序排列的词序列（n_words 个）"""
        if n_words <= 1 or len(words) <= 1:
            return words[:n_words]
        # 对每对 (a,b) 计算 order_score，构建有向得分
        best_order = list(words[:n_words])
        # 简化：对相邻位置贪心选择最高分顺序
        improved = True
        while improved:
            improved = False
            for i in range(len(best_order) - 1):
                a, b = best_order[i], best_order[i + 1]
                if self.tracker.order_score(b, a) > self.tracker.order_score(a, b):
                    # 交换更符合经验
                    best_order[i], best_order[i + 1] = b, a
                    improved = True
        return best_order

    def syntax_state(self) -> dict:
        return {
            "pairs_learned": len(self.tracker.bigram_counts),
            "top_pairs": self.tracker.top_pairs(5),
        }


def integrate_with_select(organ, builder, state_vec, vocab,
                          n_words: int = 2) -> list:
    """与 LinguisticOrgan 集成：select_words 取 top-n 候选 → 句法排序输出"""
    candidates = organ.select_words(state_vec, vocab, n_words=max(2, n_words))
    return builder.build(candidates, n_words=n_words)
