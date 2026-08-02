"""
世界模型 — 预测下一状态，条件于动作

改写为：hidden_{t} + action → hidden_{t+1}
使得系统能回答"如果我选了另一个动作会怎样"（反事实推理）
"""

import torch
import torch.nn as nn
import numpy as np


class WorldModel(nn.Module):
    def __init__(self, input_dim: int = 64, n_actions: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        
        # 动作嵌入
        self.action_embed = nn.Embedding(n_actions, input_dim // 4)
        act_dim = input_dim // 4
        
        # 预测器：hidden + action_emb → next_hidden
        self.predictor = nn.Sequential(
            nn.Linear(input_dim + act_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
        )
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        # B3 接线（DESIGN_CONCEPTS §7.5/meta-RL）：慢副本 EMA——
        # 快权重每 tick 更新（快速适应），慢影子按 EMA 追踪长期统计；
        # 稳态门控（高应激/低能量冻结慢更新——Hubel 可塑性门控最小版）
        self.ema_decay = 0.99
        self.shadow = {}
        self._register_shadow()
    
    def _register_shadow(self):
        with torch.no_grad():
            for name, p in self.named_parameters():
                self.shadow[name] = p.detach().clone()
    
    def _ema_update(self, gate: float = 1.0):
        """EMA 慢副本更新——gate<1 时冻结慢学习（应激高/能量低时保护已有表征）
        review blocking 修复：设备不匹配时惰性重注册
        （__init__ 在 CPU 注册 → .to(cuda) 后首步 EMA 必崩 RuntimeError）"""
        if gate <= 0.0:
            return
        with torch.no_grad():
            # 设备/形状校验：任一不匹配 → 重注册（.to()/grow 后安全）
            stale = False
            for name, p in self.named_parameters():
                s = self.shadow.get(name)
                if (s is None or s.device != p.device
                        or s.shape != p.shape):
                    stale = True
                    break
            if stale:
                self._register_shadow()
                return  # 重注册后本步不更新（下一 tick 正常 EMA）
            for name, p in self.named_parameters():
                self.shadow[name] = (self.ema_decay * self.shadow[name]
                                     + (1 - self.ema_decay) * p.detach())
    
    def get_slow_params(self):
        """慢副本参数（决策/评估用——长期统计视角）"""
        return self.shadow
    
    def predict(self, h: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        if action is None:
            pad = torch.zeros(*h.shape[:-1], self.input_dim // 4, device=h.device)
            return self.predictor(torch.cat([h, pad], dim=-1))
        emb = self.action_embed(action)
        while emb.dim() > h.dim():
            emb = emb.squeeze(1)
        while emb.dim() < h.dim():
            emb = emb.unsqueeze(0)
        # 维度安全：如果 h 和嵌入拼接后与预测器不匹配，截断到预测器的输入维度
        combined = torch.cat([h, emb], dim=-1)
        expected_in = self.predictor[0].in_features
        if combined.shape[-1] != expected_in:
            if combined.shape[-1] > expected_in:
                combined = combined[..., :expected_in]
            else:
                pad = torch.zeros(*combined.shape[:-1], expected_in - combined.shape[-1], device=combined.device)
                combined = torch.cat([combined, pad], dim=-1)
        return self.predictor(combined)
    
    def train_step(self, h: torch.Tensor, target: torch.Tensor, action: torch.Tensor = None,
                   gate: float = 1.0) -> float:
        pred = self.predict(h.detach(), action)
        if pred.shape[-1] != target.shape[-1]:
            min_d = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_d]
            target = target[..., :min_d]
        loss = self.loss_fn(pred, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        # B3：慢副本 EMA（gate<1 时冻结慢学习——稳态门控）
        self._ema_update(gate=gate)
        return loss.item()
    
    def imagine(self, h: torch.Tensor, action_seq: list) -> list:
        """想象一个行为序列的结果
        输入：起始 hidden + 动作序列
        输出：模拟的 hidden 轨迹
        """
        trajectory = [h.detach().clone()]
        current_h = h.detach().clone()
        with torch.no_grad():
            for a in action_seq:
                act_t = torch.tensor([a], device=current_h.device) if not isinstance(a, torch.Tensor) else a
                if act_t.dim() == 0:
                    act_t = act_t.unsqueeze(0)
                next_h = self.predict(current_h.unsqueeze(0) if current_h.dim() == 1 else current_h, act_t.unsqueeze(0) if act_t.dim() == 0 else act_t)
                trajectory.append(next_h)
                current_h = next_h
        return trajectory
    
    def grow(self, new_dim: int):
        """生长时扩展（保持设备一致）"""
        dev = next(self.parameters()).device if list(self.parameters()) else None
        old_act_w = self.action_embed.weight.data.clone().cpu()
        old_1w = self.predictor[0].weight.data.clone().cpu()
        old_1b = self.predictor[0].bias.data.clone().cpu() if self.predictor[0].bias is not None else None
        old_2w = self.predictor[2].weight.data.clone().cpu()
        old_2b = self.predictor[2].bias.data.clone().cpu() if self.predictor[2].bias is not None else None
        
        act_dim = new_dim // 4
        self.action_embed = nn.Embedding(self.n_actions, act_dim)
        self.predictor = nn.Sequential(
            nn.Linear(new_dim + act_dim, new_dim),
            nn.Tanh(),
            nn.Linear(new_dim, new_dim),
        )
        
        with torch.no_grad():
            h_act = min(old_act_w.shape[1], act_dim)
            self.action_embed.weight[:, :h_act] = old_act_w[:, :h_act]
            h = min(old_1w.shape[0], new_dim); w = min(old_1w.shape[1], new_dim + act_dim)
            self.predictor[0].weight[:h, :w] = old_1w[:h, :w]
            if old_1b is not None:
                self.predictor[0].bias[:h] = old_1b[:h]
            h = min(old_2w.shape[0], new_dim); w = min(old_2w.shape[1], new_dim)
            self.predictor[2].weight[:h, :w] = old_2w[:h, :w]
            if old_2b is not None:
                self.predictor[2].bias[:h] = old_2b[:h]
        
        self.input_dim = new_dim
        if dev is not None:
            self.to(dev)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        # B3：生长后 shadow 重注册（维度变化——防尺寸不匹配）
        self._register_shadow()
