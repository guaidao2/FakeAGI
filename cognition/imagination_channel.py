"""
想象通道 — 反事实轨迹生成 + 好奇探索引导

在安全时（空闲/睡眠），系统回放过去的经验，
替换其中某些动作，生成"如果我当时选了不同的路"的想象轨迹。
这些想象轨迹用于：
1. 训练世界模型（覆盖未见过但可能的因果链）
2. 识别知识缺口（世界模型预测不确定的区域→探索目标）
3. 生成探索策略（"去那里看看会发生什么"）
"""

import torch
import numpy as np


class ImaginationChannel:
    def __init__(self, world_model, replay_buffer_size=200, n_actions=5):
        self.world_model = world_model
        self.replay_buffer = []
        self.max_buffer = replay_buffer_size
        self.n_actions = n_actions
        self.device = next(world_model.parameters()).device
        
        # 想象生成的额外训练数据
        self.counterfactual_buffer = []
    
    def record(self, hidden_before, hidden_after, action_taken):
        """记录一次真实经验"""
        self.replay_buffer.append({
            "h_before": hidden_before.detach().clone().cpu(),
            "h_after": hidden_after.detach().clone().cpu(),
            "action": action_taken,
        })
        if len(self.replay_buffer) > self.max_buffer:
            self.replay_buffer.pop(0)
    
    def imagine_alternatives(self, batch_size=16):
        """生成反事实轨迹：用不同动作替代真实动作"""
        if len(self.replay_buffer) < 10:
            return 0.0
        
        indices = np.random.choice(len(self.replay_buffer), 
                                   min(batch_size, len(self.replay_buffer)), 
                                   replace=False)
        total_loss = 0.0
        count = 0
        
        for idx in indices:
            exp = self.replay_buffer[idx]
            h_before = exp["h_before"].to(self.device)
            real_after = exp["h_after"].to(self.device)
            real_action = exp["action"]
            
            # 选择一个不同于真实动作的反事实动作
            cf_action = np.random.choice([a for a in range(self.n_actions) if a != real_action])
            cf_action_t = torch.tensor([cf_action], device=self.device)
            
            # 预测"如果我选了反事实动作会怎样"
            with torch.no_grad():
                predicted = self.world_model.predict(h_before.unsqueeze(0), cf_action_t)
            
            # 对比真实结果和反事实结果（作为世界模型的额外训练信号）
            predicted_flat = predicted.view(-1)
            target_flat = real_after.unsqueeze(0).view(-1)
            min_len = min(len(predicted_flat), len(target_flat))
            prediction_error = float(torch.mean((predicted_flat[:min_len] - target_flat[:min_len]) ** 2))
            
            # 用反事实数据训练世界模型（维度安全）
            target = real_after.unsqueeze(0)
            if target.shape[-1] != predicted.shape[-1]:
                min_d = min(target.shape[-1], predicted.shape[-1])
                target = target[..., :min_d]
            h_before_in = h_before.unsqueeze(0)
            if h_before_in.shape[-1] != predicted.shape[-1]:
                min_d = min(h_before_in.shape[-1], predicted.shape[-1])
                h_before_in = h_before_in[..., :min_d]
            loss = self.world_model.train_step(
                h_before_in, target, cf_action_t)
            total_loss += loss
            count += 1
            
            self.counterfactual_buffer.append({
                "h_before": h_before.cpu(),
                "h_after": real_after.cpu(),
                "real_action": real_action,
                "cf_action": cf_action,
                "prediction_gap": prediction_error,
            })
            if len(self.counterfactual_buffer) > self.max_buffer:
                self.counterfactual_buffer.pop(0)
        
        return total_loss / max(1, count)
    
    def get_curiosity_goal(self) -> tuple:
        """返回最不确定的探索目标"""
        if not self.counterfactual_buffer:
            return None
        
        # 找出预测差距最大的经验
        most_uncertain = max(self.counterfactual_buffer, 
                            key=lambda x: x["prediction_gap"])
        
        # 如果最大差距仍然很小，说明世界模型已经足够确定
        if most_uncertain["prediction_gap"] < 0.1:
            return None  # 没有探索目标
        
        return most_uncertain
    
    def train_step(self):
        """每一步调用的简化版"""
        return self.imagine_alternatives(batch_size=4)
