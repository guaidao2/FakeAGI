"""
认知处理管线 — 串联感知、时序推理、世界模型、决策

每 tick 调用链：
  obs → encoder → LNN prev_hidden → world_model预测惊奇 → GameNN → action
"""

import os
import torch
import numpy as np
from cognition.temporal.lnn import LNN

# 加速优化：AGI_QUIET=1 关 verbose 日志（不影响行为——仅 I/O 提速）
_QUIET = os.environ.get("AGI_QUIET", "0") == "1"


def _log(msg):
    if not _QUIET:
        print(msg, flush=True)
from cognition.temporal.world_model import WorldModel
from cognition.temporal.world_experts import MultiExpertWorldModel
from cognition.learning.surprise import SurpriseComputer
from cognition.observation import ObservationAbstraction
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
        # P4: 观测抽象层（原始观测 → 特征通道 → 抽象向量，维度可增长）
        self.obs_abstraction = ObservationAbstraction(raw_dim=input_dim,
                                                      max_channels=cfg.get("max_channels", 16))
        self.lnn = LNN(input_dim=input_dim + self.self_state_dim, hidden_dim=hidden_dim)
        # 薛定谔叠加态世界模型（默认启用；config 可关闭）
        if cfg.get("superposition_world", True):
            from cognition.temporal.superposition_world import SuperpositionWorldModel
            self.world_model = SuperpositionWorldModel(
                input_dim=hidden_dim, n_actions=n_actions,
                n_branches=cfg.get("n_branches", 3),
                max_branches=cfg.get("max_branches", 7))
        else:
            self.world_model = WorldModel(input_dim=hidden_dim)
        # P1b: 多专家世界模型（分情境预测增强，可选启用）
        self.expert_world = MultiExpertWorldModel(input_dim=hidden_dim)
        self.surprise_computer = SurpriseComputer()
        self.gamenn = GameNN(n_strategies=n_strategies, n_actions=n_actions, state_dim=hidden_dim)
        self.hidden = None
        self.last_lnn_out = None
        self.last_action_taken = 0  # 上一步执行的动作

        self.config = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lnn.to(self.device)
        self.world_model.to(self.device)
        self.expert_world.to(self.device)

        # 生长跟踪
        self.growth_count = 0
        self.max_growths = cfg.get("max_growths", 5)
        self.growth_factor = cfg.get("growth_factor", 1.2)
        self.max_hidden = cfg.get("max_hidden", 256)
        self.growth_cooldown = 0
        self._growth_losses = []
        self.last_action = None  # 用于误差通路选择
        self.expert_weights = None  # P1b: MoE 专家激活权重（主循环注入）
        
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
        # P6: 器官生成器（高维输入自动生成感知器官）
        self.organ_generator = None
        if cfg.get("organ_growth", True):
            from cognition.perception import OrganGenerator
            self.organ_generator = OrganGenerator(
                output_dim=cfg.get("organ_output_dim", 8),
                max_organs=cfg.get("max_organs", 3),
                competition_ticks=cfg.get("organ_competition_ticks", 100),
                survival_gate=cfg.get("organ_survival_gate", 0.5))
        self.organ_cache = {}      # modality -> (输入维度, 成熟器官)
        self.organ_active = None   # 当前使用的器官
        self.organ_errors = {}     # organ_id -> 预测误差（竞争期用）
        self._organ_competition_obs = None  # 竞争期候选输出缓存
        # 语言器官（符号接地，config 开启——默认关闭，低维环境零影响）
        self.language = None
        self.language_tokens = None
        self.language_vector = None
        if cfg.get("language", False):
            from cognition.language import SymbolGrounding
            vocab = cfg.get("language_vocab", ["food", "water", "danger", "safe",
                                               "near", "far", "yes", "no"])
            self.language = SymbolGrounding(
                vocab=vocab,
                vocab_size=cfg.get("language_vocab_size", 32),
                embed_dim=cfg.get("language_embed_dim", 8),
                output_dim=cfg.get("language_output_dim", 8))
        self._organ_competition_raw = None  # 竞争期原始输入维度
        self._in_competition = False        # 竞争期标志（obs 被压缩后仍走器官路径）
    
    def process(self, obs: np.ndarray, self_state: np.ndarray,
                exploration_ratio: float = 0.0,
                survival_state: float = 1.0) -> tuple:
        # P6: 器官感知 — 高维输入自动生成器官（低维直通，零行为变化）
        organ_processed = False
        self.organ_active = None
        in_competition = getattr(self, '_in_competition', False)
        if self.organ_generator is not None and (len(obs) >= 16 or in_competition):
            modality = self.organ_generator.infer_modality(
                len(obs) if len(obs) >= 16
                else self._organ_competition_raw or 64)
            organ = self.organ_generator.get_organ(modality)
            # 无成熟器官且无候选 → 生成候选（超量生成；生存门控：
            # 饥饿/威胁高时 survival_state 低 → 不生成新器官）
            if (organ is None
                    and modality not in self.organ_generator.candidates
                    and survival_state >= self.organ_generator.survival_gate):
                self.organ_generator.generate_candidates(modality, len(obs),
                                                         n_candidates=3)
                self._in_competition = True
                self._organ_competition_raw = len(obs)
            if organ is None and modality in self.organ_generator.candidates:
                # 竞争期：候选器官轮流处理输入，**候选输出真正替换观测**
                # → 世界模型对该候选"眼睛"的预测误差即其 fitness（真驱动）
                cand_out = self.organ_generator.evaluate_candidates(
                    modality, obs, self.device)
                if self.organ_generator._active_candidate is not None:
                    organ_processed = True
                    self.organ_active = None
                    self._organ_competition_obs = cand_out
                    self._organ_competition_raw = len(obs)
                    # 候选输出替换观测（固定维度，避免维度死锁）
                    obs = cand_out
                    # 重建观测抽象层（候选输出 8D 语义）
                    if len(obs) != self.obs_abstraction.raw_dim:
                        self.obs_abstraction = ObservationAbstraction(
                            raw_dim=len(obs),
                            max_channels=self.config.get("max_channels", 16))
            if organ is not None:
                try:
                    # 用成熟器官处理观测（保持 device 一致）
                    obs_t = torch.tensor(obs, dtype=torch.float32,
                                         device=self.device).unsqueeze(0)
                    organ = organ.to(self.device)
                    processed = organ(obs_t).detach().cpu().numpy().flatten()
                    # 器官输出维度 ≤ obs 维度 → 替换观测（器官是"更好的眼睛"）
                    if len(processed) <= len(obs):
                        obs = processed
                        self.organ_active = organ
                        organ_processed = True
                        # 器官压缩后重建观测抽象层（raw_dim 语义跟随实际观测）
                        if len(obs) != self.obs_abstraction.raw_dim:
                            self.obs_abstraction = ObservationAbstraction(
                                raw_dim=len(obs),
                                max_channels=self.config.get("max_channels", 16))
                        self._in_competition = False
                except Exception:
                    organ_processed = False

        # P4: 观测抽象层 — 原始观测 → 特征通道 → 抽象向量
        # 若原始观测维度增大（新信号源），自动新增通道（主动生长，非被动补丁）
        if len(obs) > self.obs_abstraction.raw_dim:
            old_dim = self.obs_abstraction.raw_dim
            # 新通道只覆盖新增的维度（old_dim..len(obs)-1）
            self.obs_abstraction.add_channel(
                f"signal_{old_dim}", list(range(old_dim, len(obs))),
                transform="identity")
        abstract_obs = self.obs_abstraction.observe(obs)
        # 抽象维度 > 原 obs_dim → 触发感知生长（协调器会在 main 层同步全链路）
        self.obs_dim = max(self.obs_dim, len(abstract_obs))

        # 语言通道（可选）：token → 语言向量 → 拼入观测（与感知器官同构）
        self.language_vector = None
        if self.language is not None and self.language_tokens is not None:
            try:
                tok_ids = self.language.tokenize(self.language_tokens)
                if tok_ids:
                    lv = self.language.organ.encode(tok_ids)
                    self.language_vector = lv.detach().cpu().numpy().flatten()
            except Exception:
                self.language_vector = None
        if self.language_vector is not None:
            abstract_obs = np.concatenate([abstract_obs, self.language_vector])

        combined = np.concatenate([abstract_obs, self_state])

        # 感知维度自动生长（self_state_dim 动态更新，防止维度变化死循环）
        exp_dim = self.obs_dim + self.self_state_dim
        if len(combined) != exp_dim:
            if hasattr(self.lnn, 'grow_input'):
                self.obs_dim = len(abstract_obs)
                self.self_state_dim = len(self_state)
                self.lnn.grow_input(len(combined))
                _log(f"  [GROW_PERCEPTION] input->{len(combined)}dim")
            else:
                combined = combined[:exp_dim] if len(combined) > exp_dim else np.pad(combined, (0, exp_dim - len(combined)))
        elif self.lnn.input_dim != len(combined):
            self.obs_dim = len(abstract_obs)
            self.self_state_dim = len(self_state)
            self.lnn.grow_input(len(combined))
        
        x = torch.tensor(combined, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 保存上一 tick 的 hidden 用于世界模型预测
        prev_h = self.hidden.detach().clone() if self.hidden is not None else None

        # LNN 时序更新 → 新 hidden
        lnn_out, self.hidden, tau = self.lnn(x, self.hidden)
        # （③ hidden 漂移：诊断 4.74x/500tick——RMS 归一化修复尝试
        #  已回退：强制范数破坏 LNN 递归动力学（E13 26-30→2/30）。
        #  漂移是 LNN 状态增益自适应的表现，非缺陷——保留原行为，
        #  诊断工具 test_hidden_drift.py 保留供观测）

        # 世界模型：用 prev_hidden 预测当前 hidden（从过去预测现在）
        surprise = 0.0
        error_path = "perception"
        # 准备动作张量（用于条件预测）
        act_t = torch.tensor([self.last_action_taken], device=self.device)
        
        if prev_h is not None and self.hidden is not None:
            self._process_tick = getattr(self, '_process_tick', 0) + 1
            with torch.no_grad():
                # 薛定谔叠加态世界模型：多分支预测 → 观测坍缩
                if hasattr(self.world_model, "predict_dist"):
                    preds, amps = self.world_model.predict_dist(prev_h, action=act_t)
                    # 坍缩：用真实观测更新分支振幅，返回残余熵（量子化惊奇）
                    quantum_surprise = self.world_model.collapse_with_predictions(
                        preds, self.hidden.detach(), tick=self._process_tick)
                    pred = sum(a * p for a, p in zip(amps, preds))
                    # 融合：预测误差 + 坍缩熵（两者是不同来源的不确定性，相加保留全部信息）
                    pred_error = self.surprise_computer.compute(
                        pred.cpu().numpy().flatten(),
                        self.hidden.detach().cpu().numpy().flatten())
                    surprise = min(1.0, pred_error + quantum_surprise * 0.3)
                    # 分支分裂（生长）：全局坍缩失败 → 世界模型容量不足
                    if self.world_model.should_split():
                        if self.world_model.split():
                            if getattr(self, 'verbose', False):
                                print(f"[SUPERPOSITION] 分支分裂 → "
                                      f"{len(self.world_model.branches)} 分支", flush=True)
                else:
                    pred = self.world_model.predict(prev_h, action=act_t)
                    surprise = self.surprise_computer.compute(
                        pred.cpu().numpy().flatten(),
                        self.hidden.detach().cpu().numpy().flatten()
                    )
                # P6: 器官竞争期误差注入（候选器官的 fitness 依据）
                if self.organ_generator is not None:
                    # 竞争期：把世界模型误差写入当前被轮换评估的候选
                    # （候选输出已替换 obs → 该误差即候选"眼睛"的感知质量）
                    if getattr(self, '_in_competition', False):
                        cid = self.organ_generator._active_candidate
                        if cid is not None:
                            self.organ_errors[cid] = surprise
                        # 推进竞争期（用原始输入维度推断模态）
                        modality = self.organ_generator.infer_modality(
                            self._organ_competition_raw or 64)
                        if modality in self.organ_generator.candidates:
                            self.organ_generator.competition_step(
                                modality, self.organ_errors)
                            cands = self.organ_generator.candidates[modality]
                            if (cands and
                                    cands[0].age >= self.organ_generator.competition_ticks):
                                best = self.organ_generator.settle_competition(modality)
                                # 结算后清理 organ_errors（防泄漏）
                                self.organ_errors.clear()
                                self._organ_competition_obs = None
                                self._in_competition = False
                                if best is not None:
                                    self.organ_cache[modality] = (len(obs), best)
                                    if getattr(self, 'verbose', False):
                                        print(f"[ORGAN] 竞争结算: 保留 {best.describe()} "
                                              f"fitness={best.fitness:.4f} "
                                              f"(凋亡 {self.organ_generator.pruned_count})",
                                              flush=True)
                    elif self.organ_active is not None:
                        # 成熟器官：记录误差（后续可驱动结构生长）
                        self.organ_errors[self.organ_active.organ_id] = surprise
            
            # 误差通路选择：误差是否可通过行动消除？
            # 如果惊奇主要来自空间位置偏差 → 行动通路（调整导航）
            # 如果惊奇来自不可控的随机噪声 → 感知通路（更新模型）
            if surprise > 0.3 and hasattr(self, 'last_action') and self.last_action is not None:
                # 高惊奇且有最近行动记录 → 倾向行动通路
                error_path = "action"
            else:
                error_path = "perception"
            
            # 感知通路：更新世界模型（置信度门控，条件于动作）
            # B3：稳态门控——应激高/能量低时冻结慢学习（保护已有表征）
            gate = 1.0
            dv_target = None
            try:
                b = self.agi.body if hasattr(self, 'agi') and self.agi else None
                if b is not None:
                    # nit：clamp 上下限（gate ∈ [0,1]）
                    gate = min(1.0, max(0.0, 1.0 - b.stress * 2.0
                                        - max(0.0, 0.3 - b.energy) * 3.0))
                    # A 步骤：价值目标（收敛调优——ΔV 稀疏[95%≈0]导致
                    # value_head 学恒 0；改预测**价值水平 V**=归一化
                    # 身体状态——稠密平滑目标（"预测身体"哲学）
                    # review blocking：energy∈[0,2] water∈[0,1]——
                    # (energy+water)/2∈[0,1.5] clamp 后健康态恒 1.0
                    # （学恒 1 饱和伪象）；正确归一化 energy/2
                    v = (b.energy / 2.0 + b.water) / 2.0
                    v_target = torch.tensor(
                        [float(max(0.0, min(1.0, v)))],
                        dtype=torch.float32,
                        device=next(self.world_model.parameters()).device)
                    dv_target = v_target  # 复用 dv_target 参数名（V 水平）
            except Exception:
                pass
            world_loss = self.world_model.train_step(
                prev_h.detach(), self.hidden.detach(), action=act_t,
                gate=gate, dv_target=dv_target)
            
            # P1b: 多专家世界模型分情境训练（若 MoE 激活可用）
            try:
                ew = getattr(self, 'expert_weights', None)
                if ew and len(ew) > 0 and hasattr(self, 'expert_world'):
                    self.expert_world.ensure_expert(max(ew.keys()) + 1)
                    self.expert_world.train_step(
                        prev_h.detach(), self.hidden.detach(),
                        action=act_t, expert_weights=ew)
            except Exception:
                pass
            
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
            world_loss = 0.0  # A：else 分支初始化（info 组装用）
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
            # A 步骤接线修复：world_loss 必须进 info——
            # 否则 main.py info.get("world_loss", 0.5) 恒取默认值（信号空转）
            "world_loss": world_loss,
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
        try:
            self.expert_world.grow(new_h)
        except Exception:
            pass

        # 保留旧 hidden 状态并扩展到新维度
        if old_hidden is not None:
            new_hidden = torch.zeros(1, new_h, device=self.device)
            new_hidden[0, :old_hidden.shape[1]] = old_hidden
            self.hidden = new_hidden
        
        # GameNN 同步扩展
        self.gamenn.grow_state_dim(new_h)
        self.growth_cooldown = 500
        self._growth_losses = []
        _log(f"  [GROW#{self.growth_count}] {old_h}→{new_h} hidden")

    def set_exploration_ratio(self, ratio: float):
        self.gamenn.epsilon = ratio
