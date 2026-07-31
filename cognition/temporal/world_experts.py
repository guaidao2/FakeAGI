"""
P1b: 多专家世界模型 — 分情境预测

每个 MoE 专家带一个专属世界预测头，专精自己领域的状态转移。
路由激活哪个专家 → 用哪个预测头 → 预测误差返回给专家（局部过拟合）。

结构：
- 共享 encoder：hidden + action_emb → 特征
- 专家预测头池：每个头 = 2 层 MLP（各自过拟合一类情境）
- 路由：由 MoE 激活权重决定使用哪些头的加权预测
"""

import torch
import torch.nn as nn


class ExpertWorldHead(nn.Module):
    """单个专家的世界预测头"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
        )


class MultiExpertWorldModel(nn.Module):
    """
    多专家世界模型：共享 action 嵌入 + 专家预测头池。
    与 MoERouter 配合：激活的专家 → 对应预测头加权输出。
    """
    def __init__(self, input_dim: int = 64, n_actions: int = 5,
                 max_experts: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.max_experts = max_experts

        self.action_embed = nn.Embedding(n_actions, input_dim // 4)
        act_dim = input_dim // 4
        self.feat_dim = input_dim + act_dim

        # 共享特征投影（可选，简化：直接用拼接特征）
        self.heads = nn.ModuleList([ExpertWorldHead(input_dim)])
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def ensure_expert(self, n_experts: int):
        """确保预测头数量与 MoE 专家数一致（新专家 → 新预测头）"""
        while len(self.heads) < min(n_experts, self.max_experts):
            self.heads.append(ExpertWorldHead(self.input_dim))
            print(f"[WorldExperts] 新预测头 #{len(self.heads)-1} 创建", flush=True)

    def _encode(self, h: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        if action is None:
            pad = torch.zeros(*h.shape[:-1], self.input_dim // 4, device=h.device)
            return torch.cat([h, pad], dim=-1)
        emb = self.action_embed(action)
        while emb.dim() > h.dim():
            emb = emb.squeeze(1)
        while emb.dim() < h.dim():
            emb = emb.unsqueeze(0)
        combined = torch.cat([h, emb], dim=-1)
        # 维度安全
        if combined.shape[-1] != self.feat_dim:
            if combined.shape[-1] > self.feat_dim:
                combined = combined[..., :self.feat_dim]
            else:
                pad = torch.zeros(*combined.shape[:-1],
                                  self.feat_dim - combined.shape[-1],
                                  device=combined.device)
                combined = torch.cat([combined, pad], dim=-1)
        return combined

    def predict(self, h: torch.Tensor, action: torch.Tensor = None,
                expert_weights: dict = None) -> torch.Tensor:
        """
        预测下一状态。expert_weights: {expert_id: weight}
        无权重时用 head[0]（默认头）。
        """
        feat = self._encode(h, action)
        if not expert_weights:
            return self.heads[0](feat)
        # 加权专家预测
        out = None
        total_w = 0.0
        for eid, w in expert_weights.items():
            if eid < len(self.heads):
                h_out = self.heads[eid](feat)
                out = h_out if out is None else out + w * h_out
                total_w += w
        if out is None:
            return self.heads[0](feat)
        return out / max(total_w, 1e-8)

    def train_step(self, h: torch.Tensor, target: torch.Tensor,
                   action: torch.Tensor = None,
                   expert_weights: dict = None) -> float:
        pred = self.predict(h.detach(), action, expert_weights)
        if pred.shape[-1] != target.shape[-1]:
            min_d = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_d]
            target = target[..., :min_d]
        loss = self.loss_fn(pred, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def grow(self, new_dim: int):
        """生长时扩展（与 WorldModel.grow 类似，重建 heads）"""
        old_act_w = self.action_embed.weight.data.clone().cpu()
        # 保存 net[0] 和 net[2]（两层都迁移，不只第一层）
        old_heads_1w = [h.net[0].weight.data.clone().cpu() for h in self.heads]
        old_heads_1b = [h.net[0].bias.data.clone().cpu() for h in self.heads]
        old_heads_2w = [h.net[2].weight.data.clone().cpu() for h in self.heads]
        old_heads_2b = [h.net[2].bias.data.clone().cpu() for h in self.heads]

        self.input_dim = new_dim
        self.feat_dim = new_dim + new_dim // 4
        self.action_embed = nn.Embedding(self.n_actions, new_dim // 4)
        n_heads = len(self.heads)
        self.heads = nn.ModuleList([ExpertWorldHead(new_dim) for _ in range(n_heads)])

        with torch.no_grad():
            h_act = min(old_act_w.shape[1], new_dim // 4)
            self.action_embed.weight[:, :h_act] = old_act_w[:, :h_act]
            for i, h in enumerate(self.heads):
                # net[0]: 输入层（new_dim → new_dim），迁移左上角
                old_w1 = old_heads_1w[i]
                h_old1 = old_w1.shape[0]
                h.net[0].weight[:h_old1, :h_old1] = old_w1[:, :h_old1]
                h.net[0].bias[:h_old1] = old_heads_1b[i][:h_old1]
                # net[2]: 输出层（new_dim → new_dim），同样迁移左上角
                old_w2 = old_heads_2w[i]
                h_old2 = old_w2.shape[0]
                h.net[2].weight[:h_old2, :h_old2] = old_w2[:, :h_old2]
                h.net[2].bias[:h_old2] = old_heads_2b[i][:h_old2]

        # 重建 optimizer（旧 optimizer 引用已废弃的参数张量，训练梯度会落空）
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)

    def get_state_dict(self) -> dict:
        return self.state_dict()

    def load_state_dict(self, sd: dict, strict: bool = True):
        super().load_state_dict(sd, strict=strict)
