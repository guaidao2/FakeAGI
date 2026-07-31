"""
计划级认知模块 — 多步前瞻模拟 + 轨迹评分

原理：
  世界模型预测 hidden_{t+1} = f(hidden_t)
  自模型评估 "如果执行动作 A，未来状态会怎样"
  选择最优轨迹执行第一步

架构：
  rollout(hidden, horizon, branching) → trajectories[]
  每条轨迹附带预测的累积奖励
  选择最优轨迹的第一个动作执行
  每 N tick 重新规划
"""

import torch
import numpy as np
from cognition.temporal.lnn import LNN
from cognition.temporal.world_model import WorldModel


class Planner:
    def __init__(self, world_model: WorldModel, lnn: LNN,
                 n_actions=5, horizon=8, branching=3, replan_interval=5):
        self.world_model = world_model
        self.lnn = lnn
        self.n_actions = n_actions
        self.horizon = horizon          # 前瞻步数（可生长）
        self.branching = branching       # 每步分支数
        self.replan_interval = replan_interval  # 重新规划间隔
        self.planned_action = None
        self.steps_into_plan = 0
        self.device = next(world_model.parameters()).device
        
    def plan(self, hidden: torch.Tensor) -> int:
        """
        从当前 hidden 状态生成计划，返回第一步动作。
        使用世界模型模拟多个轨迹，选择最优。
        """
        if hidden is None or hidden.dim() != 2:
            return np.random.randint(0, self.n_actions)
        
        # BFS 树搜索：展开 horizon 层，每层 branching 个动作
        tree = [{"hidden": hidden.detach().clone(), "actions": [], "score": 0.0,
                 "trajectory": [], "parent": -1}]
        
        for depth in range(self.horizon):
            new_nodes = []
            for node_idx, node in enumerate(tree):
                if node["actions"] and len(node["actions"]) >= depth:
                    continue
                if node["actions"] and len(node["actions"]) > depth:
                    continue
                
                # 从当前 hidden 预测所有动作的 Q 值
                h = node["hidden"]
                with torch.no_grad():
                    # 对每个动作分别预测下一步
                    candidates = []
                    for a in range(self.n_actions):
                        act_t = torch.tensor([a], device=self.device)
                        pred_h = self.world_model.predict(h, act_t)
                        candidates.append((a, pred_h))
                # 选择 top-K 动作
                # 使用 hidden 的幅值作为不确定性估计
                h_energy = torch.mean(torch.abs(h)).item()
                noise = h_energy * 0.1
                
                # 模拟前向传播（从 LNN 取 encoder 输出）
                # 世界模型预测的是 h_{t+1}，这就是模拟的下一步状态
                for action_idx in range(self.n_actions):
                    # 用预测的 h 作为下一步的输入
                    next_h = pred_h.clone()
                    # 对每个动作评估未来奖励（简化：基于 hidden 幅值）
                    reward = float(torch.tanh(torch.mean(next_h)).item()) + np.random.randn() * noise
                    
                    score = node["score"] + reward * (0.9 ** depth)  # 折扣
                    new_node = {
                        "hidden": next_h,
                        "actions": node["actions"] + [action_idx],
                        "score": score,
                        "trajectory": node["trajectory"] + [float(reward)],
                        "parent": node_idx,
                    }
                    new_nodes.append(new_node)
            
            # 保留 top-K 节点
            new_nodes.sort(key=lambda n: n["score"], reverse=True)
            tree.extend(new_nodes[:self.branching * self.horizon])
        
        # 选择总评分最高的轨迹的第一个动作
        best = max(tree, key=lambda n: n["score"])
        best_first_action = best["actions"][0] if best["actions"] else 0
        
        # 缓存计划
        self.planned_action = best_first_action
        self.steps_into_plan = 0
        return best_first_action
    
    def get_action(self, hidden: torch.Tensor, tick: int) -> int:
        """外部调用接口：每隔 replan_interval 重新规划"""
        if self.planned_action is None or self.steps_into_plan >= self.replan_interval:
            self.planned_action = self.plan(hidden)
            self.steps_into_plan = 0
        self.steps_into_plan += 1
        return self.planned_action
    
    def get_plan_scores(self, hidden: torch.Tensor) -> np.ndarray:
        """
        返回所有动作的前瞻评分（供决策委员会投票）。
        与 plan() 共用模拟逻辑，但输出完整的动作支持向量。
        """
        scores = np.zeros(self.n_actions)
        if hidden is None or hidden.dim() != 2:
            return scores
        try:
            # 单步展开：预测每个动作的下一步 hidden，用幅值作为预期效用
            with torch.no_grad():
                h = hidden.detach().clone()
                for a in range(self.n_actions):
                    act_t = torch.tensor([a], device=self.device)
                    pred_h = self.world_model.predict(h, act_t)
                    # 预期效用 = 预测状态的信息量（幅值）+ 小幅噪声
                    util = float(torch.tanh(torch.mean(torch.abs(pred_h))).item())
                    scores[a] = max(0.0, util)
            # softmax 归一化为投票
            if np.max(scores) > 0:
                exp_s = np.exp(scores - np.max(scores))
                scores = exp_s / np.sum(exp_s)
        except Exception:
            pass
        return scores
    
    def set_horizon(self, new_horizon: int):
        """生长时可扩展计划深度"""
        self.horizon = new_horizon
