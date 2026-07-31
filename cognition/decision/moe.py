"""
P1: MoE 专家路由 — 过拟合的组织化

大脑的做法：让每个皮层区域专门化（局部过拟合），由前额叶/丘脑
决定激活哪些区域。MoE 给了过拟合一个合法位置——每个专家可以放心
过拟合自己的专长，路由器保证错误专家不会被错误激活。

实现：
- ExpertPool：专家池（每个专家 = 原型向量 + 专属小网络）
- MoERouter：情境向量 → 与各专家原型相似度 → top-K 激活权重
- 专家创建：新情境反复出现且所有专家相似度低 → 创建新专家
- 专家退役：长期低激活 → 剪枝回收
- 集成：决策委员会的投票权重 × 专家激活权重

情境向量 = 观测 + 自状态 + 最近误差模式
"""

import os
import numpy as np
import torch
import torch.nn as nn


class Expert:
    """一个专家：专精一类情境"""
    def __init__(self, eid: int, prototype: np.ndarray,
                 state_dim: int, n_actions: int,
                 created_tick: int = 0):
        self.id = eid
        self.prototype = prototype.copy()  # 情境原型（路由器匹配用）
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.created_tick = created_tick
        self.last_active_tick = created_tick
        self.activation_count = 0
        self.prediction_error = 1.0  # 该专家领域的平均预测误差
        # 专属决策小网络（局部过拟合）
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.01)


class MoERouter:
    """MoE 路由器：情境 → 专家激活权重"""
    def __init__(self, state_dim: int, n_actions: int,
                 max_experts: int = 8, top_k: int = 2,
                 create_threshold: float = 0.35,
                 retire_threshold: int = 3000,
                 device: str = "cpu"):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.max_experts = max_experts
        self.top_k = top_k
        self.create_threshold = create_threshold
        self.retire_threshold = retire_threshold
        self.device = device
        self.experts = []
        self.next_id = 0
        self.tick = 0
        # 待创建统计：情境 → 出现次数（用于"反复出现才创建"）
        self.unmatched_contexts = {}

    def _situation_vector(self, obs: np.ndarray, self_state: np.ndarray,
                          surprise: float = 0.0) -> np.ndarray:
        """情境向量 = 观测 + 自状态 + 误差"""
        parts = []
        if len(obs):
            parts.append(np.asarray(obs, dtype=np.float32))
        if len(self_state):
            parts.append(np.asarray(self_state, dtype=np.float32))
        if parts:
            vec = np.concatenate(parts)
        else:
            vec = np.zeros(self.state_dim, dtype=np.float32)
        if len(vec) < self.state_dim:
            vec = np.pad(vec, (0, self.state_dim - len(vec)))
        else:
            vec = vec[:self.state_dim]
        vec[-1] = min(1.0, surprise)  # 最后一维注入误差
        return vec

    def route(self, obs: np.ndarray, self_state: np.ndarray,
              surprise: float = 0.0) -> tuple:
        """
        路由：返回 (激活权重 dict {expert_id: w}, 情境向量)
        无专家时返回空。
        """
        self.tick += 1
        sv = self._situation_vector(obs, self_state, surprise)

        # 空专家池：创建第一个专家（模仿"出生即有一个原型专家"）
        if not self.experts:
            self.create_expert(sv)
            return {self.experts[0].id: 1.0}, sv

        # 计算与各专家原型的余弦相似度
        sims = []
        for e in self.experts:
            denom = (np.linalg.norm(e.prototype) * np.linalg.norm(sv) + 1e-8)
            sim = float(np.dot(e.prototype, sv) / denom)
            sims.append(sim)

        sims = np.array(sims)
        top_idx = np.argsort(sims)[::-1][:self.top_k]

        # softmax 权重
        exp_s = np.exp(sims[top_idx] * 3.0)  # 温度 1/3
        weights = exp_s / np.sum(exp_s + 1e-8)

        activations = {}
        for i, eid in enumerate([self.experts[j].id for j in top_idx]):
            e = self.experts[top_idx[i]]
            e.last_active_tick = self.tick
            e.activation_count += 1
            activations[e.id] = float(weights[i])

        # 专家创建检测：最高相似度仍低于阈值
        best_sim = sims[top_idx[0]]
        if best_sim < self.create_threshold:
            key = tuple(np.round(sv[:4] * 4).astype(int))  # 情境签名
            self.unmatched_contexts[key] = self.unmatched_contexts.get(key, 0) + 1
            if (self.unmatched_contexts[key] >= 20 and
                    len(self.experts) < self.max_experts):
                self.create_expert(sv)
                self.unmatched_contexts.pop(key, None)

        # 专家退役：长期未激活
        self._retire_idle()

        return activations, sv

    def create_expert(self, prototype: np.ndarray) -> int:
        """创建新专家（模仿海马体新生，绑定情境原型）"""
        eid = self.next_id
        self.next_id += 1
        e = Expert(eid, prototype, self.state_dim, self.n_actions,
                   created_tick=self.tick)
        e.net.to(self.device)
        self.experts.append(e)
        print(f"[MoE] 新专家 #{eid} 创建 (专家数={len(self.experts)})", flush=True)
        return eid

    def _retire_idle(self):
        """剪枝：长期低激活的专家回收"""
        for e in self.experts[:]:
            if (self.tick - e.last_active_tick > self.retire_threshold
                    and len(self.experts) > 1):
                self.experts.remove(e)
                print(f"[MoE] 专家 #{e.id} 退役（长期未激活）", flush=True)

    def get_action(self, activations: dict, state: np.ndarray) -> tuple:
        """
        专家加权决策：每个激活专家用自己的网络投票，按激活权重加权。
        返回 (action, 专家投票详情)
        """
        if not activations or not self.experts:
            return None, None
        q_total = np.zeros(self.n_actions)
        details = {}
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            for eid, w in activations.items():
                e = self._find(eid)
                if e is None:
                    continue
                q = e.net(s).squeeze(0).cpu().numpy()
                q_total += w * q
                details[eid] = {"weight": w, "q": q.tolist()}
        action = int(np.argmax(q_total))
        return action, details

    def learn(self, activations: dict, state: np.ndarray, action: int,
              reward: float, next_state: np.ndarray = None, gamma: float = 0.9):
        """被激活的专家在线学习（局部过拟合各自的领域）"""
        if not activations:
            return
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_t = torch.tensor([action], device=self.device)
        for eid, w in activations.items():
            e = self._find(eid)
            if e is None:
                continue
            q_net = e.net
            q_sa = q_net(s).gather(1, a_t.unsqueeze(1)).squeeze()
            if next_state is not None:
                ns = torch.tensor(next_state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
                with torch.no_grad():
                    next_q = q_net(ns).max().item()
                target = reward + gamma * next_q
            else:
                target = reward
            target_t = torch.tensor(target, dtype=q_sa.dtype, device=self.device)
            loss = nn.functional.mse_loss(q_sa, target_t)
            e.optimizer.zero_grad()
            loss.backward()
            e.optimizer.step()
            # 更新专家领域误差（用于后续路由置信度）
            e.prediction_error = 0.9 * e.prediction_error + 0.1 * loss.item()

    def _find(self, eid: int):
        for e in self.experts:
            if e.id == eid:
                return e
        return None

    def get_state_dict(self) -> dict:
        """序列化（供 checkpoint 持久化）"""
        return {
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "experts": [
                {
                    "id": e.id,
                    "prototype": e.prototype,
                    "net": e.net.state_dict(),
                    "created_tick": e.created_tick,
                    "last_active_tick": e.last_active_tick,
                    "activation_count": e.activation_count,
                    "prediction_error": e.prediction_error,
                }
                for e in self.experts
            ],
            "next_id": self.next_id,
            "tick": self.tick,
        }

    def load_state_dict(self, sd: dict):
        """反序列化（校验维度一致性，不一致则显式告警）"""
        saved_state_dim = sd.get("state_dim", self.state_dim)
        if saved_state_dim != self.state_dim:
            print(f"[MoE] 维度不匹配: 保存 state_dim={saved_state_dim} "
                  f"vs 当前 {self.state_dim} — 专家网络将按当前维度重建", flush=True)
        self.experts = []
        self.next_id = sd["next_id"]
        self.tick = sd["tick"]
        for es in sd["experts"]:
            e = Expert(es["id"], np.array(es["prototype"]),
                       saved_state_dim, sd.get("n_actions", self.n_actions),
                       created_tick=es["created_tick"])
            if saved_state_dim == self.state_dim:
                try:
                    e.net.load_state_dict(es["net"])
                except Exception as ex:
                    print(f"[MoE] 专家 #{e.id} 权重恢复失败: {ex}", flush=True)
            e.last_active_tick = es["last_active_tick"]
            e.activation_count = es["activation_count"]
            e.prediction_error = es["prediction_error"]
            e.net.to(self.device)
            self.experts.append(e)
