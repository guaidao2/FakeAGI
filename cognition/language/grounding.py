"""
语言器官（Linguistic Organ）— 符号接地

核心原理（对应 DESIGN_LANGUAGE.md）：
  - 词的意义 = 预测收益：一个词"有意义"当且仅当听到它能降低世界模型预测误差
  - 理解 = 世界模型条件化于语言输入（词 → 状态预测修正）
  - 说话 = 最小描述长度：选择使预测误差下降最大的词（逆接地）

与 P6 器官同构：token 流 → 嵌入 → 特征向量 → 接入认知核心。
词汇表可增长（Embedding 扩展，复用生长逻辑）。
"""

import torch
import torch.nn as nn
import numpy as np


class LinguisticOrgan(nn.Module):
    """语言器官：token 序列 → 固定维度语言向量"""

    def __init__(self, vocab_size: int = 32, embed_dim: int = 8,
                 hidden_dim: int = 16, output_dim: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # 2: token + position
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        # 说话路径的逆映射（状态 → 词偏好）
        self.word_probe = nn.Linear(output_dim, vocab_size)

    def encode(self, tokens) -> torch.Tensor:
        """token 序列 → 语言向量 [batch, output_dim]
        tokens: list[int] 或 [batch, seq] tensor
        """
        if isinstance(tokens, (list, tuple)) and len(tokens) > 0 \
                and isinstance(tokens[0], (list, tuple)):
            # [batch, seq]
            batch = len(tokens)
            seq = max(len(t) for t in tokens)
            padded = []
            for t in tokens:
                padded.append(list(t) + [0] * (seq - len(t)))
            idx = torch.tensor(padded, dtype=torch.long, device=self._dev())
        else:
            seq = torch.tensor(tokens if not isinstance(tokens, torch.Tensor)
                               else tokens, dtype=torch.long,
                               device=self._dev())
            if seq.dim() == 1:
                seq = seq.unsqueeze(0)
            idx = seq
        emb = self.token_embed(idx)  # [batch, seq, embed_dim]
        pos = torch.arange(emb.shape[1], device=emb.device).float().unsqueeze(0) \
            .unsqueeze(-1) / max(1.0, emb.shape[1])
        pos = pos.expand(emb.shape[0], emb.shape[1], 1)
        combined = torch.cat([emb, pos.expand(-1, -1, emb.shape[-1])], dim=-1)
        # 均值池化 + 编码
        pooled = combined.mean(dim=1)  # [batch, embed_dim*2]
        return self.encoder(pooled)

    def _dev(self):
        return next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")

    def select_words(self, state_vec: np.ndarray, vocab: list,
                     n_words: int = 1) -> list:
        """说话：从状态向量选择最有信息量的词（最小描述长度）
        用 word_probe 输出词偏好，取 top-n（贪心近似）
        """
        with torch.no_grad():
            sv = torch.tensor(state_vec, dtype=torch.float32,
                              device=self._dev()).unsqueeze(0)
            logits = self.word_probe(sv)  # [1, vocab_size]
            probs = torch.softmax(logits, dim=1)[0]
            top = torch.topk(probs, min(n_words, len(vocab))).indices.tolist()
        return [vocab[i] for i in top if i < len(vocab)]

    def grow_vocab(self, new_size: int):
        """词汇表生长：Embedding 加行 + word_probe 输出扩展（保留旧权重）"""
        if new_size <= self.vocab_size:
            return
        dev = self._dev()
        old_emb = self.token_embed.weight.data.clone().cpu()
        old_probe = self.word_probe.weight.data.clone().cpu()
        old_probe_b = self.word_probe.bias.data.clone().cpu() \
            if self.word_probe.bias is not None else None
        self.vocab_size = new_size
        self.token_embed = nn.Embedding(new_size, self.embed_dim)
        self.word_probe = nn.Linear(self.output_dim, new_size)
        with torch.no_grad():
            self.token_embed.weight[:old_emb.shape[0]] = old_emb
            self.word_probe.weight[:old_probe.shape[0]] = old_probe
            if old_probe_b is not None:
                self.word_probe.bias[:old_probe_b.shape[0]] = old_probe_b
        self.to(dev)


class SymbolGrounding:
    """符号接地：管理语言器官 + 与世界模型的接地训练"""

    def __init__(self, vocab: list, vocab_size: int = 32, embed_dim: int = 8,
                 output_dim: int = 8):
        self.vocab = vocab
        self.vocab_size = max(vocab_size, len(vocab))
        self.organ = LinguisticOrgan(vocab_size=self.vocab_size,
                                     embed_dim=embed_dim,
                                     output_dim=output_dim)
        # 接地统计：词 → 预测误差下降量
        self.grounding_score = {w: 0.0 for w in vocab}
        self.usage_count = {w: 0 for w in vocab}

    def tokenize(self, words: list) -> list:
        """词 → token id（未登录词 → 0 或忽略）"""
        ids = []
        for w in words:
            if w in self.vocab:
                ids.append(self.vocab.index(w))
        return ids

    def record_grounding(self, words: list, error_drop: float):
        """记录接地：词出现时预测误差下降量"""
        for w in words:
            if w in self.grounding_score:
                self.grounding_score[w] += error_drop
                self.usage_count[w] += 1

    def grounding_stats(self) -> dict:
        return {
            w: (self.grounding_score[w] / max(1, self.usage_count[w]))
            for w in self.vocab
        }
