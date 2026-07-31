"""
P6 结构原语库 — 可生长器官的构建单元

设计（对应四大理论）：
  - NEAT 拓扑进化：原语链是"基因"，可变异/复制
  - 神经营养假说：超量生成候选器官，使用选择决定保留
  - 皮层柱模式：器官 = 原语的重复堆叠
  - 进化发育：新模态起点是单通道检测器（光敏蛋白）

每个原语：
  - 是 nn.Module，可被 Organ 串联
  - 有 identity 版本（纯直通，不改变输入）——保证"低维直通"路径
  - 有学习版本（可训练参数）
"""

import torch
import torch.nn as nn


class PatchBase(nn.Module):
    """原语基类：记录结构元信息（用于变异/复制/历史标记）"""
    def __init__(self, patch_type: str, out_dim: int = None):
        super().__init__()
        self.patch_type = patch_type
        self.out_dim = out_dim
        self.innovation_id = None  # NEAT 历史标记（变异时分配）

    def extra_repr(self):
        return f"type={self.patch_type}, out={self.out_dim}"


class ConvPatch(PatchBase):
    """局部卷积原语：捕获空间局部模式（视觉器官的"视网膜细胞"）
    输入: [batch, channels, H, W] 或 [batch, H*W]（自动视为 1 通道）
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel: int = 3, in_dim: int = None):
        super().__init__("conv", out_dim=None)
        self.in_channels = in_channels
        self.kernel = kernel
        self.in_dim = in_dim  # 1D 输入的维度（H*W）
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel,
                              padding=kernel // 2)
        self._identity = False  # 可学习

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # [batch, in_dim] → [batch, 1, H, W]
            batch = x.shape[0]
            side = int(round(self.in_dim ** 0.5))
            x = x.view(batch, 1, side, side)
        elif x.dim() == 3:
            # [batch, channels, flat] → [batch, channels, H, W]
            batch, ch, flat = x.shape
            side = int(round(flat ** 0.5))
            x = x.view(batch, ch, side, side)
        out = self.conv(x)  # [batch, out_channels, H, W]
        # 展平到 2D：[batch, out_channels*H*W]
        return out.flatten(1)

    def expected_out(self, in_dim: int) -> int:
        side = int(round(in_dim ** 0.5))
        return self.out_channels * side * side


class PoolPatch(PatchBase):
    """降采样原语：压缩空间信息（"下采样"——信息浓缩）
    输入: [batch, in_dim] 或 [batch, ch, H, W]（展平输入）
    """
    def __init__(self, in_dim: int = None, kernel: int = 2):
        super().__init__("pool", out_dim=None)
        self.in_dim = in_dim
        self.kernel = kernel
        self.pool = nn.AvgPool2d(kernel)
        self._identity = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            batch = x.shape[0]
            side = int(round(self.in_dim ** 0.5))
            x = x.view(batch, 1, side, side)
            out = self.pool(x)
            return out.flatten(1)
        if x.dim() == 3:
            batch, ch, flat = x.shape
            side = int(round(flat ** 0.5))
            x = x.view(batch, ch, side, side)
            out = self.pool(x)
            return out.flatten(1)
        return self.pool(x).flatten(1)

    def expected_out(self, in_dim: int) -> int:
        side = int(round(in_dim ** 0.5))
        ns = max(1, side // self.kernel)
        return ns * ns


class EmbedPatch(PatchBase):
    """嵌入原语：离散/符号输入 → 稠密向量（文本/ID 流的"符号皮层"）"""
    def __init__(self, n_embed: int = 256, embed_dim: int = 16):
        super().__init__("embed", out_dim=embed_dim)
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(n_embed, embed_dim)
        self._identity = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = x.long().clamp(0, self.n_embed - 1)
        return self.embed(idx).flatten(1)

    def expected_out(self, in_dim: int) -> int:
        return in_dim * self.embed_dim


class NormPatch(PatchBase):
    """归一化原语：标准化输入分布（器官的"稳态调节"）"""
    def __init__(self, dim: int = None):
        super().__init__("norm", out_dim=None)
        self.dim = dim
        self.norm = nn.LayerNorm(dim) if dim else None
        self._identity = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is None:
            self.norm = nn.LayerNorm(x.shape[-1])
        return self.norm(x)

    def expected_out(self, in_dim: int) -> int:
        return in_dim


class LinearPatch(PatchBase):
    """线性变换原语：通用特征变换（器官的"皮层投射"）"""
    def __init__(self, in_dim: int = None, out_dim: int = None):
        super().__init__("linear", out_dim=out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim) if in_dim and out_dim else None
        self._identity = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc is None:
            self.fc = nn.Linear(x.shape[-1], self.out_dim or x.shape[-1])
        return torch.tanh(self.fc(x))

    def expected_out(self, in_dim: int) -> int:
        return self.out_dim or in_dim


class IdentityPatch(PatchBase):
    """恒等原语：直通（保证低维路径零行为变化 + 器官退化时的保底）"""
    def __init__(self):
        super().__init__("identity", out_dim=None)
        self._identity = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def expected_out(self, in_dim: int) -> int:
        return in_dim


# 原语注册表：生成器按模态类型选择
PRIMITIVE_REGISTRY = {
    "continuous": [LinearPatch, NormPatch],   # 连续流：线性+归一化
    "pixel": [ConvPatch, PoolPatch, NormPatch],  # 像素流：卷积+池化+归一化
    "discrete": [EmbedPatch, LinearPatch],    # 符号流：嵌入+线性
}
