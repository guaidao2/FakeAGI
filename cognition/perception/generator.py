"""
P6 器官生成器 — 超量生成 + 竞争选择 + 生存门控

对应神经营养假说：
  1. 新模态出现（高维输入 + 持续高熵）→ 超量生成候选器官（3-5 个）
  2. 候选并行竞争 N tick，fitness = 预测误差下降量（A 度量）
  3. 竞争结束：fitness 最优者保留（成熟），其余凋亡
  4. 好奇增益 = 生存状态门控（饥饿/威胁高 → 好奇低 → 不生成新器官）

fitness 只用"误差下降量"（学习进展）——防好奇漩涡（沉迷白噪声）。
"""

import torch
import torch.nn as nn
import numpy as np

from .organ import Organ
from .primitives import (
    IdentityPatch, LinearPatch, NormPatch, ConvPatch, PoolPatch,
    PRIMITIVE_REGISTRY,
)


class OrganGenerator:
    def __init__(self, output_dim: int = 8, max_organs: int = 3,
                 competition_ticks: int = 100, survival_gate: float = 0.5):
        self.output_dim = output_dim      # 器官输出维度（下游观测通道数）
        self.max_organs = max_organs      # 同一模态最多共存器官数
        self.competition_ticks = competition_ticks
        self.survival_gate = survival_gate  # 生存状态门控：<0.5 禁止生成
        self.registry = {}                # modality -> 成熟器官列表
        self.candidates = {}              # modality -> 竞争中的候选
        self.next_organ_id = 0
        self.generation_count = 0
        self.pruned_count = 0

    # ─── 模态类型推断（从输入特征）───
    @staticmethod
    def infer_modality(input_dim: int) -> str:
        if input_dim >= 64:
            return "pixel"      # 高维 → 像素流（视觉）
        if input_dim >= 16:
            return "continuous" # 中维 → 连续流
        return "continuous"     # 低维 → 连续流（直通即可）

    # ─── 生成候选器官（超量生成）───
    def generate_candidates(self, modality: str, input_dim: int,
                            n_candidates: int = 3) -> list:
        """为某模态生成多个候选器官（不同原语组合）"""
        cands = []
        registry = PRIMITIVE_REGISTRY.get(modality, [LinearPatch, NormPatch])
        for i in range(n_candidates):
            patches = []
            # 每个候选 = 随机 1-3 个原语的组合
            n_patches = np.random.randint(1, min(3, len(registry)) + 1)
            for _ in range(n_patches):
                cls = np.random.choice(registry)
                if cls is LinearPatch:
                    patches.append(LinearPatch(in_dim=input_dim,
                                               out_dim=self.output_dim))
                elif cls is NormPatch:
                    patches.append(NormPatch(dim=input_dim))
                elif cls is ConvPatch:
                    patches.append(ConvPatch(in_dim=input_dim))
                elif cls is PoolPatch:
                    patches.append(PoolPatch(in_dim=input_dim))
                else:
                    patches.append(IdentityPatch())
            organ = Organ(input_dim=input_dim, output_dim=self.output_dim,
                          patches=patches, organ_id=self.next_organ_id,
                          modality=modality)
            self.next_organ_id += 1
            cands.append(organ)
        return cands

    # ─── 竞争期推进 ───
    def competition_step(self, modality: str, error_by_organ: dict):
        """每 tick 调用：更新各候选 fitness"""
        if modality not in self.candidates:
            return
        for organ in self.candidates[modality]:
            err = error_by_organ.get(organ.organ_id, 1.0)
            organ.update_fitness(err)
            organ.age += 1

    # ─── 竞争结算（选择/凋亡）───
    def settle_competition(self, modality: str):
        """竞争期结束：fitness 最优保留，其余凋亡"""
        if modality not in self.candidates:
            return None
        cands = self.candidates[modality]
        if not cands:
            return None
        # 选择 fitness 最高者
        best = max(cands, key=lambda o: o.fitness)
        best.mature = True
        if modality not in self.registry:
            self.registry[modality] = []
        self.registry[modality].append(best)
        self.pruned_count += len(cands) - 1
        # 清空候选
        self.candidates[modality] = []
        self.generation_count += 1
        return best

    # ─── 主入口：每 tick 调用 ───
    def step(self, modality: str, input_dim: int,
             errors: dict, survival_state: float = 1.0) -> dict:
        """
        errors: {organ_id: error}（含成熟器官）
        返回事件字典：{"generated": [...], "settled": organ, "none": True}
        """
        events = {"generated": [], "settled": None, "none": True}
        # 1. 是否需要新器官：该模态无成熟器官 且 生存状态允许（生存门控）
        has_mature = modality in self.registry and self.registry[modality]
        if not has_mature and modality not in self.candidates:
            # 好奇增益 = 生存门控：饥饿/威胁高 → 不生成
            if survival_state >= self.survival_gate:
                cands = self.generate_candidates(modality, input_dim,
                                                 n_candidates=3)
                self.candidates[modality] = cands
                events["generated"] = cands
                events["none"] = False
        # 2. 竞争期推进
        if modality in self.candidates and self.candidates[modality]:
            self.competition_step(modality, errors)
            # 竞争期结束判定
            cands = self.candidates[modality]
            if cands and cands[0].age >= self.competition_ticks:
                best = self.settle_competition(modality)
                events["settled"] = best
                events["none"] = False
        return events

    # ─── 获取成熟器官 ───
    def get_organ(self, modality: str):
        if modality in self.registry and self.registry[modality]:
            return self.registry[modality][-1]  # 最新成熟的
        return None

    # ─── 状态 ───
    def get_state(self) -> dict:
        return {
            "generation_count": self.generation_count,
            "pruned_count": self.pruned_count,
            "organs": {m: [o.describe() for o in orgs]
                       for m, orgs in self.registry.items()},
            "candidates": {m: [o.describe() for o in c]
                           for m, c in self.candidates.items()},
        }
