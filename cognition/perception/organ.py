"""
P6 器官本体 — 可生长的感知器官

Organ = 原语链（NEAT 式基因型）
  - forward：串行处理输入，输出固定维度（下游无感）
  - fitness：预测误差下降量（A 度量——只统计学习进展，防好奇漩涡）
  - replicate：链尾复制同构原语（皮层柱模式）
  - mutate：参数/原语变异（NEAT 历史标记）
  - prune：低贡献原语修剪（突触修剪）
"""

import torch
import torch.nn as nn
import numpy as np


class Organ(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 patches: list = None, organ_id: int = 0,
                 modality: str = "continuous"):
        super().__init__()
        self.organ_id = organ_id
        self.modality = modality
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patches = nn.ModuleList(patches if patches else [])
        # fitness 状态（A 度量：误差下降量）
        self.fitness = 0.0
        self.error_history = []       # 竞争期误差序列
        self.base_error = None        # 竞争开始时的基准误差
        self.age = 0                  # 存活 tick
        self.mature = False           # 竞争期结束标记
        self.mutation_count = 0
        self.innovation_id = f"organ_{organ_id}"

    # ─── 前向 ───
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for p in self.patches:
            h = p(h)
        # 输出维度对齐：截断/填充到 output_dim
        if h.shape[-1] != self.output_dim:
            if h.shape[-1] > self.output_dim:
                h = h[..., :self.output_dim]
            else:
                pad = torch.zeros(*h.shape[:-1],
                                  self.output_dim - h.shape[-1],
                                  device=h.device)
                h = torch.cat([h, pad], dim=-1)
        return h

    # ─── fitness 更新（A 度量：误差下降量）───
    def update_fitness(self, error: float):
        """竞争期每 tick 调用。fitness = 基准误差 - 当前误差（下降量）"""
        self.error_history.append(error)
        if len(self.error_history) > 200:
            self.error_history.pop(0)
        if self.base_error is None:
            self.base_error = error
        # 用最近 20 步平均 vs 基准：误差下降越多 fitness 越高
        recent = np.mean(self.error_history[-20:]) if len(self.error_history) >= 20 \
            else np.mean(self.error_history)
        self.fitness = max(0.0, self.base_error - recent)

    # ─── 结构生长 ───
    def replicate(self) -> bool:
        """皮层柱模式：链尾追加同构原语（复制最后一块）"""
        if not self.patches:
            return False
        last = self.patches[-1]
        if hasattr(last, "_identity") and last._identity:
            return False  # identity 不复制
        # 复制最后一块（深拷贝参数）
        import copy
        child = copy.deepcopy(last)
        child.innovation_id = f"{self.innovation_id}.r{self.mutation_count}"
        # 小幅扰动新块（避免与旧块完全相同）
        with torch.no_grad():
            for p in child.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        self.patches.append(child)
        self.mutation_count += 1
        return True

    def mutate(self) -> bool:
        """NEAT 式变异：随机选一块做参数扰动"""
        if not self.patches:
            return False
        idx = np.random.randint(len(self.patches))
        p = self.patches[idx]
        with torch.no_grad():
            for param in p.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        self.mutation_count += 1
        return True

    def prune(self) -> bool:
        """突触修剪：移除贡献最小的原语（fitness 停滞时）"""
        if len(self.patches) <= 1:
            return False
        # 简化策略：移除第一个非 identity 原语（保守修剪）
        for i, p in enumerate(self.patches):
            if not (hasattr(p, "_identity") and p._identity):
                del self.patches[i]
                self.mutation_count += 1
                return True
        return False

    # ─── 序列化 ───
    def describe(self) -> str:
        return "->".join(getattr(p, "patch_type", "?") for p in self.patches) or "empty"
