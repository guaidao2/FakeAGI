"""
GameNN 决策模块 — 完整版（含 Q 学习 + 博弈矩阵）

基于原版 GameNN-WorldModel 的策略头实现。
输入：LNN 的 hidden state 向量
输出：选中的动作 + 策略索引
学习：TD 误差更新 Q 值 + 策略权重
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StrategyQNet(nn.Module):
    """每个策略头对应一个状态→动作 Q 网络"""
    def __init__(self, state_dim, n_actions, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
    
    def forward(self, state):
        return self.net(state)


class GameNNDecision:
    """
    完整 GameNN 决策模块。
    
    每个策略头有独立的 Q 网络。
    策略选择基于混合 score：Q 值 + 策略权重 + 探索噪声。
    博弈矩阵记录策略间胜负。
    """
    
    def __init__(self, n_strategies=4, n_actions=5, state_dim=64, gamma=0.9, lr=0.01):
        self.n_strategies = n_strategies
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 0.3
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 每个策略一个 Q 网络
        self.q_nets = [StrategyQNet(state_dim, n_actions).to(self.device) for _ in range(n_strategies)]
        self.optimizers = [torch.optim.AdamW(q.parameters(), lr=lr) for q in self.q_nets]
        
        # 策略权重（博弈矩阵边缘概率）
        self.strategy_weights = np.ones(n_strategies) / n_strategies
        self.strategy_scores = np.zeros(n_strategies)
        self.strategy_counts = np.ones(n_strategies)
        
        # 博弈矩阵
        self.game_matrix = np.zeros((n_strategies, n_strategies))
        
        # 缓存（用于学习）
        self.last_state = None
        self.last_strategy = None
        self.last_action = None
        
        # 置信度跟踪（用于反射抑制）
        self.strategy_update_counts = np.zeros(n_strategies)  # 每个策略的更新次数
        self.confidence = 0.0  # 整体置信度 [0, 1]
        
    def get_confidence(self) -> float:
        """返回学习系统的置信度。越高→越应该用学习替代反射"""
        min_updates = 20  # 最少需要 20 次更新才可靠
        ratio = min(1.0, np.mean(self.strategy_update_counts) / min_updates)
        return ratio * (1.0 - self.epsilon)  # 置信度随探索率降低而升高
    
    def select_action(self, state: np.ndarray) -> tuple:
        """选择动作，返回 (action_idx, strategy_idx)"""
        self.last_state = state.copy()
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # ε-贪心选策略
        if np.random.random() < self.epsilon:
            strategy_idx = np.random.randint(self.n_strategies)
        else:
            strategy_idx = np.argmax(self.strategy_weights)
        
        # 策略网络选动作
        with torch.no_grad():
            q_values = self.q_nets[strategy_idx](state_t)
            action = int(q_values.argmax().item())
        
        self.last_strategy = strategy_idx
        self.last_action = action
        return action, strategy_idx
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """返回当前最优策略的动作概率分布（供决策委员会投票）"""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            # 用策略权重加权各策略网络的 Q 值
            q_total = torch.zeros(self.n_actions, device=self.device)
            for i, net in enumerate(self.q_nets):
                q_total += self.strategy_weights[i] * net(state_t).squeeze(0)
            probs = torch.softmax(q_total, dim=-1)
        return probs.cpu().numpy()
    
    def learn(self, reward: float, next_state: np.ndarray = None, done: bool = False,
              action: int = None):
        """TD 误差学习（审计 W2：action 参数——实际执行动作归因。
        此前用 process 内部建议动作（self.last_action）学习，而 reward 来自
        委员会实际执行动作（语言/元认知/概念覆盖时归因错位污染 Q 值）"""
        if self.last_state is None or self.last_strategy is None:
            return
        
        # 实际执行动作（若提供）；否则回退建议动作（向后兼容）
        exec_action = action if action is not None else self.last_action
        
        # 维度防护（grow 后 last_state 可能旧维——决策与学习跨 tick，
        # grow_state_dim 重建 q_net 后旧状态需对齐；next_state 同理）
        q_net = self.q_nets[self.last_strategy]
        sd_now = q_net.net[0].weight.shape[1]  # StrategyQNet 是容器
        ls = self.last_state
        if len(ls) != sd_now:
            ls = (ls[:sd_now] if len(ls) > sd_now
                  else np.pad(ls, (0, sd_now - len(ls))))
        s = torch.tensor(ls, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.tensor([exec_action], dtype=torch.long, device=self.device)
        r = torch.tensor([reward], dtype=torch.float32, device=self.device)
        
        q_net = self.q_nets[self.last_strategy]
        opt = self.optimizers[self.last_strategy]
        
        # TD 目标
        with torch.no_grad():
            if next_state is not None and not done:
                # security MEDIUM：next_state 同样做维度防护（对称于
                # last_state——注释"同理"必须真做：未来调用方若传旧维
                # 会 q_net(ns) shape mismatch 崩溃主循环）
                ns_arr = np.asarray(next_state, dtype=np.float32).flatten()
                if len(ns_arr) != sd_now:
                    ns_arr = (ns_arr[:sd_now] if len(ns_arr) > sd_now
                              else np.pad(ns_arr, (0, sd_now - len(ns_arr))))
                ns = torch.tensor(ns_arr, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
                next_q = q_net(ns).max().item()
                target = reward + self.gamma * next_q
            else:
                target = reward
        
        # TD 误差
        q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        target_t = torch.tensor(target, dtype=q_sa.dtype, device=self.device)
        loss = F.mse_loss(q_sa, target_t)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        opt.step()
        
        # 更新置信度
        self.strategy_update_counts[self.last_strategy] += 1
        self.confidence = self.get_confidence()
        
        # 更新策略权重（Softmax）
        self.strategy_counts[self.last_strategy] += 1
        self.strategy_scores[self.last_strategy] += reward
        avg = self.strategy_scores / np.maximum(self.strategy_counts, 1)
        exp_scores = np.exp(avg - avg.max())
        self.strategy_weights = exp_scores / exp_scores.sum()
    
    def update_matrix(self, my_idx: int, opp_idx: int, reward: float):
        """更新博弈矩阵"""
        self.game_matrix[my_idx, opp_idx] += reward
    
    def grow_state_dim(self, new_dim: int):
        """当 LNN 隐藏层增长时扩展 Q 网络"""
        old_dim = self.state_dim
        self.state_dim = new_dim
        for i in range(self.n_strategies):
            old_weights = {k: v.data.clone().cpu() for k, v in self.q_nets[i].state_dict().items()}
            self.q_nets[i] = StrategyQNet(new_dim, self.n_actions).to(self.device)
            with torch.no_grad():
                for name, param in self.q_nets[i].named_parameters():
                    if name in old_weights:
                        ow = old_weights[name]
                        h = min(param.shape[0], ow.shape[0])
                        w = min(param.shape[1], ow.shape[1]) if len(param.shape) > 1 else param.shape[0]
                        if len(param.shape) > 1:
                            param[:h, :w] = ow[:h, :w].to(self.device)
                        else:
                            param[:h] = ow[:h].to(self.device)
            self.optimizers[i] = torch.optim.AdamW(self.q_nets[i].parameters(), lr=self.lr)
