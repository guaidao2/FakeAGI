"""
薛定谔叠加态世界模型（Superposition World Model）

哲学对应（十大定律之 ③ 智能 / ④ 误差修正）：
  现实世界在"被观测之前"处于多可能状态叠加；系统对未来的预测
  不应是单一确定性向量，而应是多个可能分支的叠加（波函数）。

  当真实观测到达 → 坍缩：与观测匹配的分支振幅上升，不匹配的衰减。
  所有分支都与观测不符 → 容量不足 → 分裂出新分支（生长，P4 哲学）。

这解决了确定性世界模型的根本缺陷：当环境存在隐藏规则（如实验4的
"踩开关→解锁食物"），单值模型永远学不会（预测恒错），而叠加态模型
会同时维持"吃食物→能量↑"和"吃食物→能量不变"两个假设分支，
直到证据坍缩掉错误分支，并继续维持"可能有开关"的探索性分支。
"""

import torch
import torch.nn as nn
import numpy as np


class SuperpositionBranch(nn.Module):
    """一个可能世界分支：独立的小预测网络"""
    def __init__(self, input_dim: int, n_actions: int, branch_id: int = 0):
        super().__init__()
        self.branch_id = branch_id
        self.input_dim = input_dim
        self.n_actions = n_actions
        act_dim = max(2, input_dim // 4)
        self.action_embed = nn.Embedding(n_actions, act_dim)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim + act_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
        )
        # 分支置信度：该分支被历史证据支持的程度（坍缩权重，softmax 归一化）
        self.register_buffer("amplitude", torch.tensor(1.0 / 3.0))
        self.register_buffer("hit_count", torch.tensor(0.0))
        self.register_buffer("miss_count", torch.tensor(0.0))

    def forward(self, h: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        if action is None:
            pad = torch.zeros(*h.shape[:-1], self.action_embed.embedding_dim, device=h.device)
            return self.predictor(torch.cat([h, pad], dim=-1))
        emb = self.action_embed(action.to(h.device) if action.device != h.device else action)
        while emb.dim() > h.dim():
            emb = emb.squeeze(1)
        while emb.dim() < h.dim():
            emb = emb.unsqueeze(0)
        combined = torch.cat([h, emb], dim=-1)
        expected_in = self.predictor[0].in_features
        if combined.shape[-1] != expected_in:
            if combined.shape[-1] > expected_in:
                combined = combined[..., :expected_in]
            else:
                pad = torch.zeros(*combined.shape[:-1],
                                  expected_in - combined.shape[-1],
                                  device=combined.device)
                combined = torch.cat([combined, pad], dim=-1)
        return self.predictor(combined)


class SuperpositionWorldModel(nn.Module):
    """
    叠加态世界模型：
      - 多个分支（SuperpositionBranch）各自预测一个"可能未来"
      - amplitude 表示各分支的坍缩权重（softmax）
      - collapse()：真实观测到达 → 更新权重，衰减不匹配分支
      - split()：容量不足 → 分裂新分支（生长）
    """

    def __init__(self, input_dim: int = 64, n_actions: int = 5,
                 n_branches: int = 3, max_branches: int = 7):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.max_branches = max_branches
        self.branches = nn.ModuleList([
            SuperpositionBranch(input_dim, n_actions, branch_id=i)
            for i in range(n_branches)
        ])
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        # A 步骤：叠加态价值头（多假设价值——各分支预测 ΔV 后振幅加权）
        # 惰性创建：pred 实际维度可能 ≠ input_dim（hidden 拼接），首次匹配
        self.value_head = None
        self.value_head_weight = 0.5
        self.collapse_temp = 1.0
        self.collapse_history = []  # (tick, dominant_branch, entropy)
        self.last_entropy = 0.0

    def _amplitudes(self, device) -> torch.Tensor:
        return torch.softmax(
            torch.tensor([b.amplitude.item() for b in self.branches],
                         device=device) / self.collapse_temp, dim=0)

    # ─── 预测 ───
    def predict(self, h: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """确定性预测接口（兼容旧调用）：加权平均所有分支"""
        preds = [b(h, action) for b in self.branches]
        amps = self._amplitudes(h.device)
        out = torch.zeros_like(preds[0])
        for i, p in enumerate(preds):
            out = out + amps[i] * p
        return out

    def predict_dist(self, h: torch.Tensor, action: torch.Tensor = None):
        """叠加预测：返回 (分支预测列表, 振幅列表)"""
        preds = [b(h, action) for b in self.branches]
        amps = self._amplitudes(h.device)
        return preds, amps

    # ─── 坍缩：观测到达，更新分支权重 ───
    def collapse_with_predictions(self, preds: list, actual: torch.Tensor,
                                  tick: int = 0) -> float:
        """用真实观测坍缩叠加态。返回坍缩后熵（残余不确定性）。"""
        with torch.no_grad():
            target = actual.detach()
            errors = []
            for b, p in zip(self.branches, preds):
                if p.shape[-1] != target.shape[-1]:
                    min_d = min(p.shape[-1], target.shape[-1])
                    e = float(self.loss_fn(p[..., :min_d], target[..., :min_d]))
                else:
                    e = float(self.loss_fn(p, target))
                errors.append(e)
            logits = torch.tensor([-e / max(0.05, self.collapse_temp)
                                   for e in errors])
            new_amps = torch.softmax(logits, dim=0)
            for i, b in enumerate(self.branches):
                # copy_ 保持 buffer 设备一致（直接赋值会把 buffer 换成 CPU tensor）
                b.amplitude.copy_(new_amps[i])
                if errors[i] < 0.02:
                    b.hit_count += 1.0
                else:
                    b.miss_count += 1.0
            probs = new_amps.cpu().numpy()
            entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
            self.last_entropy = entropy
            self.collapse_history.append((tick, int(new_amps.argmax().item()), entropy))
            if len(self.collapse_history) > 200:
                self.collapse_history.pop(0)
            return entropy

    # ─── 训练 ───
    def train_step(self, h: torch.Tensor, target: torch.Tensor,
                   action: torch.Tensor = None, gate: float = 1.0,
                   dv_target: torch.Tensor = None) -> float:
        """训练所有分支（振幅加权损失：被证据支持的分支优先学习）
        A 步骤：叠加态价值预测——加权输出经 value_head 预测 ΔV
        （多假设价值的坍缩；B3 gate 忽略——分支即多假设无慢副本）"""
        amps = self._amplitudes(h.device)
        self.optimizer.zero_grad()
        total_loss = 0.0
        for i, b in enumerate(self.branches):
            pred = b(h.detach(), action)
            t = target.detach()
            if pred.shape[-1] != t.shape[-1]:
                min_d = min(pred.shape[-1], t.shape[-1])
                pred = pred[..., :min_d]
                t = t[..., :min_d]
            loss = self.loss_fn(pred, t)
            total_loss = total_loss + amps[i] * loss
            # A：分支级价值预测（各分支对自己的下一状态预测价值）
            if dv_target is not None:
                vdim = pred.shape[-1]
                if (self.value_head is None
                        or self.value_head.weight.shape[1] != vdim):
                    import torch.nn as nn
                    self.value_head = nn.Linear(vdim, 1).to(pred.device)
                    # 维度变化 → 重建 optimizer（含 value_head 参数）
                    self.optimizer = torch.optim.AdamW(
                        self.parameters(), lr=0.001)
                dv_pred = self.value_head(pred.detach())
                dv_t = dv_target.detach().reshape(dv_pred.shape)
                total_loss = total_loss + amps[i] * self.value_head_weight * \
                    self.loss_fn(dv_pred, dv_t)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item()

    # ─── 生长：分支分裂 ───
    def split(self) -> bool:
        """分裂一个新分支（继承最强分支权重 + 小扰动）"""
        if len(self.branches) >= self.max_branches:
            return False
        amps = [b.amplitude.item() for b in self.branches]
        parent_idx = int(np.argmax(amps))
        parent = self.branches[parent_idx]
        child = SuperpositionBranch(self.input_dim, self.n_actions,
                                    branch_id=len(self.branches))
        dev = next(self.parameters()).device if list(self.parameters()) else None
        if dev is not None:
            child.to(dev)
        child.load_state_dict(parent.state_dict())
        with torch.no_grad():
            for p in child.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        child.amplitude = torch.tensor(parent.amplitude.item() * 0.6,
                                       device=dev if dev is not None else None)
        self.branches.append(child)
        # 重建优化器：新分支参数必须进入 param_groups（否则永不更新）
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return True

    def should_split(self, threshold_ratio: float = 0.85) -> bool:
        """何时分裂：最近多次坍缩熵接近理论上限（均匀分布）→ 所有分支都无法
        区分观测（全局坍缩失败）→ 容量不足。阈值用熵/ln(n) 归一化，
        不受分支数影响（均匀分布熵=ln(n)，比值=1.0）。"""
        if len(self.branches) >= self.max_branches:
            return False
        recent = self.collapse_history[-20:]
        if len(recent) < 10:
            return False
        n = len(self.branches)
        max_entropy = np.log(n) if n > 1 else 1.0
        avg_entropy = np.mean([e for _, _, e in recent])
        ratio = avg_entropy / max_entropy
        worst_branch = min(b.miss_count.item() for b in self.branches)
        return ratio > threshold_ratio and worst_branch > 5

    # ─── 兼容旧接口 ───
    def imagine(self, h: torch.Tensor, action_seq: list) -> list:
        trajectory = [h.detach().clone()]
        current_h = h.detach().clone()
        with torch.no_grad():
            for a in action_seq:
                act_t = torch.tensor([a], device=current_h.device) if not isinstance(a, torch.Tensor) else a
                if act_t.dim() == 0:
                    act_t = act_t.unsqueeze(0)
                next_h = self.predict(current_h.unsqueeze(0) if current_h.dim() == 1 else current_h,
                                     act_t.unsqueeze(0) if act_t.dim() == 0 else act_t)
                trajectory.append(next_h)
                current_h = next_h
        return trajectory

    def grow(self, new_dim: int):
        """维度生长：所有分支扩容（继承权重）"""
        dev = next(self.parameters()).device if list(self.parameters()) else None
        old_states = [b.state_dict() for b in self.branches]
        old_amps = [b.amplitude.item() for b in self.branches]
        old_hits = [b.hit_count.item() for b in self.branches]
        old_misses = [b.miss_count.item() for b in self.branches]
        self.branches = nn.ModuleList([
            SuperpositionBranch(new_dim, self.n_actions, branch_id=i)
            for i in range(len(self.branches))
        ])
        if dev is not None:
            self.to(dev)  # 先迁移新分支到设备，再拷贝（避免 buffer/参数设备不一致）
        with torch.no_grad():
            for i, b in enumerate(self.branches):
                old = old_states[i]
                b.amplitude = torch.tensor(old_amps[i], device=dev)
                b.hit_count = torch.tensor(old_hits[i], device=dev)
                b.miss_count = torch.tensor(old_misses[i], device=dev)
                for key in old:
                    if key in b.state_dict():
                        o = old[key].to(dev)  # 旧权重迁到目标设备
                        n = b.state_dict()[key]
                        if o.shape == n.shape:
                            n.copy_(o)
                        else:
                            slices = tuple(slice(0, min(a, c)) for a, c in zip(o.shape, n.shape))
                            try:
                                n[slices] = o[slices]
                            except Exception:
                                pass
        self.input_dim = new_dim
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)

    # ─── 序列化辅助 ───
    def branch_stats(self) -> list:
        return [(b.branch_id, round(b.amplitude.item(), 3),
                 int(b.hit_count.item()), int(b.miss_count.item()))
                for b in self.branches]
