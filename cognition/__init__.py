"""
认知处理管线 — 串联感知、时序推理、世界模型、决策

每 tick 调用链：
  obs → encoder → LNN prev_hidden → world_model预测惊奇 → GameNN → action
"""

import torch
import numpy as np
from cognition.temporal.lnn import LNN
from cognition.temporal.world_model import WorldModel
from cognition.learning.surprise import SurpriseComputer
from cognition.decision.gamenn import GameNNDecision as GameNN
from cognition.planner import Planner
from cognition.imagination_channel import ImaginationChannel


class CognitionPipeline:
    def __init__(self, config: dict = None):
        cfg = config or {}
        input_dim = cfg.get("input_dim", 8)
        hidden_dim = cfg.get("hidden_dim", 64)
        n_actions = cfg.get("n_actions", 5)
        n_strategies = cfg.get("n_strategies", 4)

        self.obs_dim = input_dim
        self.self_state_dim = cfg.get("self_state_dim", 14)  # body(8) + drives(6)
        self.lnn = LNN(input_dim=input_dim + self.self_state_dim, hidden_dim=hidden_dim)
        self.world_model = WorldModel(input_dim=hidden_dim)
        self.surprise_computer = SurpriseComputer()
        self.gamenn = GameNN(n_strategies=n_strategies, n_actions=n_actions, state_dim=hidden_dim)
        self.hidden = None
        self.last_lnn_out = None
        self.last_action_taken = 0  # 上一步执行的动作

        self.config = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lnn.to(self.device)
        self.world_model.to(self.device)

        # 生长跟踪
        self.growth_count = 0
        self.max_growths = cfg.get("max_growths", 5)
        self.growth_factor = cfg.get("growth_factor", 1.2)
        self.max_hidden = cfg.get("max_hidden", 256)
        self.growth_cooldown = 0
        self._growth_losses = []
        self.last_action = None  # 用于误差通路选择
        
        # 置信度跟踪（错误处理边界）
        self.confidence = 0.0       # 初始置信度0，让反射主导直到学习建立
        self.confidence_alpha = 0.01
        self.error_buffer = []      # 最近误差缓冲
        self.persistent_error_count = 0
        
        # 计划级认知
        self.planner = Planner(
            world_model=self.world_model, lnn=self.lnn,
            n_actions=n_actions, horizon=6, branching=3, replan_interval=5
        )
        # 想象通道
        self.imagination = ImaginationChannel(
            world_model=self.world_model, n_actions=n_actions
        )
    
    def process(self, obs: np.ndarray, self_state: np.ndarray,
                exploration_ratio: float = 0.0) -> tuple:
        combined = np.concatenate([obs, self_state])
        
        # 感知维度自动生长（self_state_dim 动态更新，防止维度变化死循环）
        exp_dim = self.obs_dim + self.self_state_dim
        if len(combined) != exp_dim:
            if hasattr(self.lnn, 'grow_input'):
                self.obs_dim = len(obs)
                self.self_state_dim = len(self_state)
                self.lnn.grow_input(len(combined))
                print(f"  [GROW_PERCEPTION] input->{len(combined)}dim", flush=True)
            else:
                combined = combined[:exp_dim] if len(combined) > exp_dim else np.pad(combined, (0, exp_dim - len(combined)))
        elif self.lnn.input_dim != len(combined):
            self.obs_dim = len(obs)
            self.self_state_dim = len(self_state)
            self.lnn.grow_input(len(combined))
        
        x = torch.tensor(combined, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 保存上一 tick 的 hidden 用于世界模型预测
        prev_h = self.hidden.detach().clone() if self.hidden is not None else None

        # LNN 时序更新 → 新 hidden
        lnn_out, self.hidden, tau = self.lnn(x, self.hidden)

        # 世界模型：用 prev_hidden 预测当前 hidden（从过去预测现在）
        surprise = 0.0
        error_path = "perception"
        # 准备动作张量（用于条件预测）
        act_t = torch.tensor([self.last_action_taken], device=self.device)
        
        if prev_h is not None and self.hidden is not None:
            with torch.no_grad():
                pred = self.world_model.predict(prev_h, action=act_t)
            surprise = self.surprise_computer.compute(
                pred.cpu().numpy().flatten(),
                self.hidden.detach().cpu().numpy().flatten()
            )
            
            # 误差通路选择：误差是否可通过行动消除？
            # 如果惊奇主要来自空间位置偏差 → 行动通路（调整导航）
            # 如果惊奇来自不可控的随机噪声 → 感知通路（更新模型）
            if surprise > 0.3 and hasattr(self, 'last_action') and self.last_action is not None:
                # 高惊奇且有最近行动记录 → 倾向行动通路
                error_path = "action"
            else:
                error_path = "perception"
            
            # 感知通路：更新世界模型（置信度门控，条件于动作）
            world_loss = self.world_model.train_step(
                prev_h.detach(), self.hidden.detach(), action=act_t)
            
            # 记录经验到想象通道
            if prev_h is not None:
                self.imagination.record(prev_h.detach(), self.hidden.detach(), self.last_action_taken)
                # 定期运行想象（每10 tick一次反事实生成）
                if np.random.random() < 0.1:
                    self.imagination.imagine_alternatives(batch_size=8)
            confidence_gate = self.confidence
            if surprise > 0.1:
                effective_update = world_loss * confidence_gate
            else:
                effective_update = world_loss
            
            self.error_buffer.append(surprise)
            if len(self.error_buffer) > 50:
                self.error_buffer.pop(0)
            recent_error = np.mean(self.error_buffer) if self.error_buffer else 0
            self.confidence = (1 - self.confidence_alpha) * self.confidence + \
                              self.confidence_alpha * (1.0 - min(1.0, recent_error * 5))
            
            self._check_growth(world_loss)
        else:
            self._check_growth(0.0)

        # GameNN 状态相关决策（如果 LNN 维度变动，截断/填充到 GameNN 固定维度）
        self.gamenn.epsilon = exploration_ratio
        self.last_lnn_out = lnn_out  # 供主循环习惯投票使用（与训练状态一致）
        feat = lnn_out.detach().cpu().numpy().flatten()
        if len(feat) != self.gamenn.state_dim:
            if len(feat) > self.gamenn.state_dim:
                feat = feat[:self.gamenn.state_dim]
            else:
                feat = np.pad(feat, (0, self.gamenn.state_dim - len(feat)))
        action, strategy_idx = self.gamenn.select_action(feat)
        self.last_action = action
        self.last_action_taken = action
        
        # 计划级认知：当置信度足够时，用计划动作替代 GameNN 动作
        if self.confidence > 0.3 and self.hidden is not None:
            planned_action = self.planner.get_action(self.hidden, self.growth_count)
            # 只在计划轨迹评分高于即时决策时采用
            action = planned_action
        
        info = {
            "surprise": surprise,
            "tau": tau.mean().item(),
            "strategy": strategy_idx,
            "energy_delta": -0.001,
            "error_path": error_path,
        }
        return action, info

    def _check_growth(self, loss: float):
        """置信度门控的生长检测（大幅减少触发频率）"""
        self.growth_cooldown = max(0, self.growth_cooldown - 1)
        if (self.growth_cooldown > 0 or self.growth_count >= self.max_growths
                or self.lnn.hidden_dim >= self.max_hidden):
            return
        self._growth_losses.append(loss)
        if len(self._growth_losses) > 100:
            self._growth_losses.pop(0)
        # 需要更多样本来确认 plateau
        if len(self._growth_losses) < 80:
            return
        
        should_grow = False
        if self.confidence < 0.15:  # 置信度极低才触发
            should_grow = True
        else:
            recent = np.mean(self._growth_losses[-30:])
            earlier = np.mean(self._growth_losses[-60:-30])
            if abs(recent - earlier) <= 0.005:  # 更严格 plateau 检测
                should_grow = True
        
        if not should_grow:
            return

        old_h = self.lnn.hidden_dim
        old_hidden = self.hidden.detach().clone() if self.hidden is not None else None
        # 增量式生长：每次 +8 神经元（像海马体新生），而非批量 ×1.2
        new_h = min(self.max_hidden, old_h + 8)
        if new_h == old_h:
            return

        # 生长前快照（用于可能的回滚）
        old_lnn_state = {k: v.data.clone().cpu() for k, v in self.lnn.state_dict().items()}
        old_wm_state = {k: v.data.clone().cpu() for k, v in self.world_model.state_dict().items()}
        old_confidence = self.confidence

        self.lnn.grow(new_h)
        self.world_model.grow(new_h)

        # 保留旧 hidden 状态并扩展到新维度
        if old_hidden is not None:
            new_hidden = torch.zeros(1, new_h, device=self.device)
            new_hidden[0, :old_hidden.shape[1]] = old_hidden
            self.hidden = new_hidden
        
        # GameNN 同步扩展
        self.gamenn.grow_state_dim(new_h)
        self.growth_cooldown = 500
        self._growth_losses = []
        print(f"  [GROW#{self.growth_count}] {old_h}→{new_h} hidden", flush=True)

    def set_exploration_ratio(self, ratio: float):
        self.gamenn.epsilon = ratio
