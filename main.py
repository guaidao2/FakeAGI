"""
AGI 主入口 — 自维持循环（生物模拟版）

集成：
  身体模拟（BodyModel）
  多驱动力（DriveSystem）
  空间记忆（SpatialMemory）
  危险感知（DangerSystem）
  睡眠巩固（SleepCycle）
  认知核心（LNN + World Model + GameNN）
"""

import time
import numpy as np
from core.body import BodyModel
from core.drives import DriveSystem
from core.self_model import SelfModel
from core.homeostasis import Homeostasis
from cognition.spatial_memory import SpatialMemory
from cognition.danger import DangerSystem
from cognition.sleep import SleepCycle
from cognition.hemin import OtherModel
from cognition.metacognition.core import MetacognitionLayer
from cognition.metacognition.strategy_manager import LearningStrategyManager
from cognition.latent_state import LatentStateModel
from cognition.attention import AttentionGate
from cognition.concept_bank import ConceptBank
from cognition.decision.committee import DecisionCommittee
from cognition.decision.moe import MoERouter
from growth.coordinator import GrowthCoordinator
from core.physics_intuition import PhysicsPrior
from core.value_system import EvolvableValueSystem
from core.persistence import save_checkpoint, load_checkpoint


class AGI:
    def __init__(self, config: dict = None):
        self.cfg = config or {}
        
        # ─── 身体层 ───
        self.body = BodyModel()
        self.drives = DriveSystem()
        self.self_model = SelfModel()
        self.homeostasis = Homeostasis()
        
        # ─── 认知层 ───
        self.cognition = None
        self.spatial_memory = SpatialMemory()
        self.danger_system = DangerSystem()
        self.sleep_cycle = SleepCycle()
        self.other_model = OtherModel()
        
        # ─── 六大缺口模块 ───
        self.physics = PhysicsPrior()          # 物理直觉
        self.latent_state = LatentStateModel() # 隐变量
        self.attention = None                  # 注意力（obs_dim 确定后创建）
        self.concept_bank = ConceptBank()      # 概念库（组合式反事实）
        self.strategy_mgr = LearningStrategyManager()  # 元-元认知
        self.value_system = EvolvableValueSystem()     # 可进化价值系统
        
        # ─── 环境 ───
        self.env = None
        
        # ─── 经验缓存（供睡眠巩固使用） ───
        self.replay_buffer = []
        self.max_replay = 200
        
        # ─── 第五层：元认知 ───
        self.metacognition = MetacognitionLayer(spatial_memory=self.spatial_memory)
        self.committee = DecisionCommittee(n_actions=5)
        self.committee_state = None
        # P1: MoE 专家路由（延迟创建——state_dim 由认知维度决定）
        self.moe = None
        self.moe_state_dim = None
        # P4: 统一生长协调器（全链路同步生长）
        self.growth = GrowthCoordinator(max_hidden=256)
        self._last_abstract_dim = None
        self.override_action = -1
        self._food_recently_tick = -1000
        self.causal_error = 0.0
        
        # ─── 运行状态 ───
        self.tick = 0
        self.alive = True
        self.last_action = 0
        self.pos = [0, 0]
        self.survival_ticks = 0
        self.peak_health = 0.0
    
    def set_cognition(self, cognition):
        self.cognition = cognition
    
    def set_env(self, env):
        self.env = env
    
    # ─── P0: Checkpoint 持久化（跨 session 身份连续性） ───
    def save(self, tag: str = "latest", path: str = None) -> str:
        """保存当前状态（供死亡/退出时自主调用）"""
        return save_checkpoint(self, path=path, tag=tag)
    
    def load(self, tag: str = "latest", path: str = None) -> bool:
        """加载先前状态（启动时调用）"""
        return load_checkpoint(self, path=path, tag=tag)
    
    def _secondary_reached(self, obs) -> bool:
        """次级目标（如开关）是否已到达"""
        if len(obs) >= 5:
            return obs[4] > 0.5  # 开关已触发标志
        return len(obs) >= 4 and abs(obs[2]) < 0.05 and abs(obs[3]) < 0.05
    
    def _ensure_moe(self):
        """延迟创建 MoE 路由器（state_dim 由认知维度决定）"""
        if self.moe is not None:
            return
        state_dim = 16
        if self.cognition is not None and getattr(self.cognition, 'hidden', None) is not None:
            hd = self.cognition.hidden.shape[1]
            state_dim = max(8, min(hd, 32))
        self.moe_state_dim = state_dim
        self.moe = MoERouter(state_dim=state_dim, n_actions=5,
                             max_experts=6, top_k=2,
                             device="cuda" if __import__('torch').cuda.is_available() else "cpu")
    
    def _coordinated_growth(self):
        """
        P4: 观测维度变化 → 全链路协调生长。
        注册各模块生长接口（首次），观测抽象维度增长时同步全链路。
        """
        if self.cognition is None:
            return
        # 注册模块（幂等）
        if "lnn" not in self.growth.modules:
            self.growth.register(
                "lnn",
                grow_fn=lambda d: self.cognition.lnn.grow_input(d)
                if d < 256 else None,
                dim_fn=lambda: self.cognition.lnn.input_dim)
            self.growth.register(
                "world_model",
                grow_fn=lambda d: None,  # WM 随 LNN hidden 生长，不随观测
                dim_fn=lambda: self.cognition.world_model.input_dim)
            # 薛定谔叠加态世界模型：分支分裂也是生长（注册进协调器）
            wm = self.cognition.world_model
            if hasattr(wm, "split") and "world_branches" not in self.growth.modules:
                self.growth.register(
                    "world_branches",
                    grow_fn=lambda d: wm.split() or None,
                    dim_fn=lambda: len(wm.branches))
            # P6: 感知器官（结构生长：replicate/mutate 也是生长事件）
            if (self.cognition.organ_generator is not None
                    and "perception_organ" not in self.growth.modules):
                gen = self.cognition.organ_generator
                def _organ_grow(d=None):
                    organ = gen.get_organ(
                        gen.infer_modality(d or 16))
                    if organ is None:
                        return None
                    return organ.replicate() or organ.mutate()
                self.growth.register(
                    "perception_organ",
                    grow_fn=_organ_grow,
                    dim_fn=lambda: sum(len(o.patches) for o in
                                       [gen.get_organ(m) for m in gen.registry]
                                       if o is not None))
        # 检测观测抽象维度变化
        abs_dim = self.cognition.obs_dim
        if self._last_abstract_dim is not None and abs_dim > self._last_abstract_dim:
            # 观测抽象层增长 → 协调器记账 + 同步其他模块
            target = abs_dim + self.cognition.self_state_dim
            grown = self.growth.sync_to(target, source="obs_growth")
            if not grown:
                # pipeline 内部已 grow_input，这里仅记录同步事件（不重复计数）
                self.growth.growth_events += 1
                self.growth.history.append(
                    (self.growth.growth_events, "obs_growth",
                     ["obs_abstraction", "lnn"]))
                if self.growth.log:
                    print(f"[GROWTH] 观测同步 #{self.growth.growth_events}: "
                          f"obs {self._last_abstract_dim}→{abs_dim}D "
                          f"(pipeline 已生长)", flush=True)
            # MoE 维度协调：记录到协调器（不直接改 state_dim，避免破坏已有专家）
            if self.moe is not None:
                print(f"[GROWTH] MoE 需协调: 当前 state_dim={self.moe.state_dim}, "
                      f"目标 >= {target}（新专家创建时自动适配）", flush=True)
        self._last_abstract_dim = abs_dim
    
    def _moe_state_vector(self) -> np.ndarray:
        """MoE 专家状态向量（隐状态压缩 + 驱动）"""
        sd = self.moe_state_dim or 16
        parts = []
        if self.cognition is not None and getattr(self.cognition, 'hidden', None) is not None:
            h = self.cognition.hidden.detach().cpu().numpy().flatten()
            parts.append(h[:8] if len(h) >= 8 else np.pad(h, (0, 8 - len(h))))
        drive_vec = self.drives.get_state_vector()
        parts.append(drive_vec[:4] if len(drive_vec) >= 4 else np.pad(drive_vec, (0, 4 - len(drive_vec))))
        vec = np.concatenate(parts)[:sd]
        if len(vec) < sd:
            vec = np.pad(vec, (0, sd - len(vec)))
        return vec.astype(np.float32)
    
    def step(self):
        """单 tick 自维持循环"""
        self.tick += 1
        
        # ─── 1. 感知 ───
        if self.env:
            obs = self.env.observe()
            if hasattr(self.env, 'get_pos'):
                self.pos = self.env.get_pos()
        else:
            obs = np.zeros(4)
        
        # ─── 1b. 注意力门控 ───
        if self.attention is None:
            self.attention = AttentionGate(obs_dim=len(obs))
        drive_vec_now = self.drives.get_state_vector() if hasattr(self.drives, 'get_state_vector') else None
        obs = self.attention.update(obs, drive_vec_now)
        
        # ─── 1c. 物理先验检查（瞬移→应激上升） ───
        if hasattr(self, 'prev_pos') and self.prev_pos is not None:
            if self.physics.check_teleport(self.prev_pos, self.pos):
                self.body.stress = min(1.0, self.body.stress + 0.1)
        
        # ─── 2. 危险感知 ───
        threat = self.danger_system.sense(obs, self.tick)
        danger_nearby = self.danger_system.is_threat_nearby()
        
        # ─── 3. 身体更新 ───
        is_moving = (self.last_action in [0, 1, 2, 3])
        energy_delta = -0.0005 if is_moving else -0.0001
        water_delta = -0.0002
        damage = 0.0
        
        if self.env:
            if hasattr(self.env, 'get_energy_delta'):
                energy_delta = self.env.get_energy_delta(self.last_action)
            if hasattr(self.env, 'get_damage'):
                damage = self.env.get_damage(self.last_action)
        
        self.body.update(energy_delta=energy_delta, water_delta=water_delta,
                         damage=damage, is_moving=is_moving,
                         was_moved_passively=getattr(self, '_pending_was_moved', False))
        self._pending_was_moved = False
        
        # ─── 4. 睡眠检测 ───
        if not self.body.is_sleeping:
            if self.sleep_cycle.should_sleep(self.body.fatigue, self.body.circadian, self.body.energy):
                self.body.is_sleeping = True
                self.sleep_cycle.is_sleeping = True
                self.sleep_cycle.sleep_duration = 0
                if self.cognition:
                    # 睡眠时记忆巩固
                    consolidated = self.sleep_cycle.consolidate(self.replay_buffer)
                    self.replay_buffer = consolidated[:self.max_replay]
        else:
            self.sleep_cycle.sleep_duration += 1
            if self.sleep_cycle.should_wake(self.body.fatigue, self.body.energy):
                self.body.is_sleeping = False
                self.sleep_cycle.is_sleeping = False
                self.body.fatigue = max(0.0, self.body.fatigue - 0.3)  # 醒来后疲劳大幅降低
        
        # ─── 5. 自模型更新标记位（实际更新放到认知处理之后） ───
        surprise = 0.0
        
        # ─── 6. 驱动力更新 ───
        body_state = self.body.get_state_dict()
        self.drives.update(body_state, self.self_model.survival_prob, surprise, self.tick, danger_nearby)
        dominant_drive = self.drives.get_dominance()
        drive_bias = self.drives.get_action_bias()
        
        # ─── 7. 认知处理 ───
        action = 0
        if self.cognition and not self.body.is_sleeping:
            # 构建完整自模型状态（身体 + 驱动力 + 隐变量上下文）
            body_vec = self.body.get_state_vector()
            drive_vec = self.drives.get_state_vector()
            latent_ctx = self.latent_state.get_context_vector()
            self_state = np.concatenate([body_vec, drive_vec, latent_ctx])
            
            # 探索率由驱动力决定
            if dominant_drive in ("hunger", "thirst", "fear"):
                exploration = 0.05
            elif dominant_drive in ("boredom", "curiosity"):
                exploration = 0.6
            elif dominant_drive in ("fatigue",):
                exploration = 0.1
            else:
                exploration = 0.2
            
            # 误差通路：行动通路 → 提高探索率（在 process 之前生效）
            action, info = self.cognition.process(
                obs, self_state, exploration,
                survival_state=self.body.integrity)
            # P4: 观测变化 → 全链路协调生长
            try:
                self._coordinated_growth()
            except Exception:
                pass
            surprise = info.get("surprise", 0.0)
            error_path = info.get("error_path", "perception")
            
            # ─── 隐变量推断（在真实 surprise 产生后） ───
            latent_found = self.latent_state.observe_prediction_error(
                surprise, self.tick, obs)

            # 误差通路：行动通路 → 随机探索调整
            if error_path == "action" and surprise > 0.2 and not self.body.is_critical():
                if np.random.random() < 0.3:
                    action = np.random.randint(0, 4)
            
            # 学习驱动的反射抑制：当 GameNN 学到可靠策略时，抑制本能反射
            gamenn_confidence = self.cognition.gamenn.confidence if hasattr(self.cognition, 'gamenn') else 0.0
            suppress_reflex = gamenn_confidence > 0.15
            
            # 元认知系统更新
            if self.metacognition is not None:
                self.metacognition.update(
                    world_model_loss=info.get("world_loss", 0.5),
                    gamenn_confidence=gamenn_confidence,
                    surprise=surprise,
                    health=self.body.health,
                    energy=self.body.energy,
                    action=action,
                    survived=self.alive,
                    causal_error=getattr(self, 'causal_error', 0.0),
                    agent_pos=self.pos if hasattr(self, 'pos') else None,
                    env_size=getattr(self.env, 'size', 10) if self.env else 10,
                    energy_delta=energy_delta
                )
                mc_state = self.metacognition.get_state()
                if mc_state["override_action"] >= 0:
                    self.override_action = mc_state["override_action"]
                elif mc_state.get("override_target") is not None and hasattr(self, 'pos'):
                    # 目标位置转动作
                    tx, ty = mc_state["override_target"]
                    dx_t = tx - self.pos[0]
                    dy_t = ty - self.pos[1]
                    if abs(dx_t) > 0.05 or abs(dy_t) > 0.05:
                        self.override_action = 3 if dx_t > 0 else (2 if dx_t < 0 else (4 if dy_t > 0 else 1))
            
            # ─── 7c. 人脑式决策委员会：并行投票 + 加权仲裁 ───
            self._ensure_moe()
            if self.committee is not None:
                # 0. 语言指令投票（方向词→动作先验，听词行走的本能）
                lang_v = None
                if (hasattr(self.cognition, 'language')
                        and self.cognition.language is not None
                        and self.cognition.language_tokens):
                    DIR_MAP = {"east": 3, "west": 2, "north": 1, "south": 4}
                    for w in self.cognition.language_tokens:
                        if w in DIR_MAP:
                            lang_v = self.committee.language_vote(DIR_MAP[w])
                            break
                # 1. 反射投票（本能：朝主要目标）
                reflex_v = self.committee.reflex_vote(
                    obs, drive_bias, self.body.get_state_dict(),
                    secondary_reached=self._secondary_reached(obs))
                # 2. 边缘系统投票（驱动力偏置）
                limbic_v = self.committee.limbic_vote(drive_bias)
                # 3. 习惯投票（GameNN 概率，用 LNN 输出状态与训练一致）
                habit_v = None
                if (hasattr(self.cognition, 'gamenn')
                        and getattr(self.cognition, 'last_lnn_out', None) is not None):
                    g = self.cognition.gamenn
                    if hasattr(g, 'get_action_probs'):
                        sd = g.state_dim
                        s = self.cognition.last_lnn_out.detach().cpu().numpy().flatten()
                        if len(s) < sd:
                            s = np.pad(s, (0, sd - len(s)))
                        else:
                            s = s[:sd]
                        habit_v = g.get_action_probs(s)
                # 4. 规划投票（前瞻模拟，若有规划器）
                plan_v = None
                if hasattr(self.cognition, 'planner') and self.cognition.planner is not None:
                    try:
                        plan_v = self.cognition.planner.get_plan_scores(
                            self.cognition.hidden)
                    except Exception:
                        plan_v = None
                # 5. 元认知投票（知识缺口重定向）
                meta_v = None
                if self.metacognition is not None:
                    try:
                        mc_state = self.metacognition.get_state()
                        if mc_state["override_action"] >= 0:
                            meta_v = self.committee.meta_vote(
                                mc_state["override_action"])
                    except Exception:
                        meta_v = None
                
                # P1: MoE 专家路由（情境 → 专家激活，作为额外决策层）
                moe_action = None
                self.moe_activations = {}
                if (self.moe is not None and len(obs) > 0):
                    try:
                        moe_state = self._moe_state_vector()
                        activations, _ = self.moe.route(obs, moe_state, surprise)
                        self.moe_activations = activations
                        # 注入激活权重给多专家世界模型
                        if hasattr(self.cognition, 'expert_weights'):
                            self.cognition.expert_weights = activations
                        if activations:
                            a, _ = self.moe.get_action(activations, moe_state)
                            if a is not None:
                                moe_action = a
                    except Exception:
                        pass
                
                # 加权仲裁（委员会 + MoE 融合：MoE 动作优先，专家置信度高时）
                votes = {"reflex": reflex_v, "limbic": limbic_v,
                         "habit": habit_v, "plan": plan_v, "meta": meta_v}
                if lang_v is not None:
                    votes["language"] = lang_v
                decision = self.committee.decide(
                    votes,
                    health=self.body.health,
                    stress=self.body.stress,
                    confidence=gamenn_confidence,
                    energy=self.body.energy,
                    exploration_ratio=exploration)
                action = decision["action"]
                self.committee_state = decision
                # MoE 专家决策覆盖（仅在专家池成熟且置信时）
                if (moe_action is not None and self.moe is not None
                        and len(self.moe.experts) >= 1
                        and gamenn_confidence < 0.1):
                    action = moe_action
            
            # GameNN 学习：基于能量变化的奖励信号（用 LNN hidden 作状态）
            if (hasattr(self.cognition, 'gamenn') and self.cognition.hidden is not None):
                g = self.cognition.gamenn
                sd = g.state_dim
                s = self.cognition.hidden.detach().cpu().numpy().flatten()
                if len(s) < sd:
                    s = np.pad(s, (0, sd - len(s)))
                else:
                    s = s[:sd]
                reward = energy_delta * 10 + damage * (-5)
                g.learn(reward, next_state=s)
            
            # P1: MoE 专家在线学习（被激活的专家学习自己的领域）
            if (self.moe is not None and self.moe_activations
                    and getattr(self.cognition, 'hidden', None) is not None):
                try:
                    moe_s = self._moe_state_vector()
                    self.moe.learn(self.moe_activations, moe_s, action,
                                   reward=energy_delta * 10,
                                   next_state=None)
                except Exception:
                    pass
            
            # 睡眠由驱动力触发状态（不是动作 4；睡眠是状态，动作编号 4 = down）
            if drive_bias[4] > 0.7 and self.body.energy > 0.5 and not self.body.is_sleeping:
                self.body.is_sleeping = True
                self.sleep_cycle.is_sleeping = True
        # 睡眠动作（睡眠是状态，不是动作）
        if self.body.is_sleeping:
            action = 0  # 睡眠时动作无关，自模型会处理恢复
        
        # ─── 7b. 元认知重定向（仅当元认知检测到知识缺口或主要目标确认为假时才触发）
        if not self.body.is_sleeping and len(obs) >= 4:
            ate_recently = hasattr(self, '_food_recently_tick') and (self.tick - self._food_recently_tick) < 30
            near_primary = len(obs) >= 2 and abs(obs[0]) < 0.08 and abs(obs[1]) < 0.08
            # 检测主要目标是否是"假"的（站在目标上但没获得能量）
            fake_primary = near_primary and len(obs) >= 5 and obs[4] < 0.5
            # 元认知知识缺口（GapDetector 检测到因果/策略缺口）
            mc_gap = False
            try:
                if self.metacognition is not None:
                    mc_gap = self.metacognition.get_state().get("gap_detected", False)
            except Exception:
                pass
            if (mc_gap or fake_primary) and self.body.energy < 0.95 and not ate_recently:
                sx, sy = obs[2], obs[3]
                # 次级目标已到达判定：obs[4] 存在时用触发标志，否则用距离阈值
                if len(obs) >= 5:
                    secondary_reached = obs[4] > 0.5  # 开关已触发
                else:
                    secondary_reached = abs(sx) < 0.05 and abs(sy) < 0.05
                if secondary_reached:
                    # 次级目标已到达：清除知识缺口，让反射接管去找主要目标
                    try:
                        self.metacognition.gap_detector.fast_failure_detected = False
                    except Exception:
                        pass
                else:
                    if abs(sx) > abs(sy):
                        action = 3 if sx > 0 else 2
                    elif abs(sy) > 0.05:
                        action = 4 if sy > 0 else 1
        
        # ─── 8. 行动 ───
        # 行动前记录位置（用于被动位移检测）
        pos_before_action = tuple(self.pos) if hasattr(self, 'pos') else None
        
        if self.env:
            result = self.env.step(action)
            if isinstance(result, dict):
                env_energy = result.get("energy_delta", 0)
                env_water = result.get("water_delta", 0)
                if abs(env_energy) > 0.001:
                    self.body.energy = np.clip(self.body.energy + env_energy, 0, 2)
                    # 获取到正回报 → 清除元认知覆盖 + 标记最近吃饱了
                    if env_energy > 0.01:
                        self.override_action = -1
                        self._food_recently_tick = self.tick
                if abs(env_water) > 0.001:
                    self.body.water = np.clip(self.body.water + env_water, 0, 1)
        
        # 被动位移检测：行动后位置与预期不符
        was_moved = False
        if pos_before_action is not None:
            pos_after = tuple(self.pos)
            # 计算预期位置（动态获取边界）
            size = getattr(self.env, 'size', 10)
            dirs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
            if action < 5:
                edx, edy = dirs[action]
                expected = (max(0, min(size-1, pos_before_action[0] + edx)),
                            max(0, min(size-1, pos_before_action[1] + edy)))
                if pos_after != expected:
                    was_moved = True
        if was_moved:
            self._pending_was_moved = True
        
        # ─── 9. 空间记忆更新 ───
        food_nearby = hasattr(self.env, 'food_nearby') and self.env.food_nearby()
        water_nearby = hasattr(self.env, 'water_nearby') and self.env.water_nearby() if hasattr(self.env, 'water_nearby') else False
        self.spatial_memory.update_position(
            tuple(self.pos),
            energy_delta=energy_delta,
            surprise=surprise,
            danger=danger_nearby,
            food_nearby=food_nearby,
            water_nearby=water_nearby,
        )
        self.spatial_memory.tick_aging(decay=0.001)
        
        # ─── 10. 经验缓存（用于睡眠巩固） ───
        self.replay_buffer.append({
            "pos": self.pos,
            "action": action,
            "surprise": surprise,
            "health": self.body.health,
            "drive": dominant_drive,
        })
        if len(self.replay_buffer) > self.max_replay:
            self.replay_buffer.pop(0)
        
        # ─── 10b. 他者模型更新（自我-他者对比） ───
        self.other_model.record_self_action(action, tuple(self.pos), dominant_drive)
        divergence = self.other_model.update()
        
        # ─── 10c. 概念提取 + 组合式反事实生成 ───
        try:
            self.concept_bank.extract_from_obs(obs, action, {"energy_delta": energy_delta})
            # 每 300 tick 生成一次组合式反事实（记录为内部"假设场景"）
            if self.tick % 300 == 0:
                combo = self.concept_bank.generate_combo(n=3)
                if combo:
                    scenario = self.concept_bank.combo_to_scenario(combo)
                    # 反事实经验进入记忆（供睡眠巩固 / 日志使用）
                    self.replay_buffer.append({
                        "pos": self.pos, "action": -1, "surprise": surprise,
                        "health": self.body.health, "drive": "imagination",
                        "counterfactual": scenario,
                    })
        except Exception as e:
            print(f"[WARN] concept_bank: {e}", flush=True)
        
        # ─── 10d. 元-元认知：学习策略管理 ───
        try:
            wloss = surprise  # 用 surprise 作为误差代理
            strategy = self.strategy_mgr.update(
                world_loss=wloss, surprise=surprise,
                confidence=getattr(self.cognition, 'confidence', 0.5) if self.cognition else 0.5,
                health=self.body.health)
            strat_params = self.strategy_mgr.get_parameters()
            # 消费策略参数（探索覆盖仅在非危急时应用，避免饿死）
            strat_exploration = strat_params.get("exploration")
            if (strat_exploration is not None
                    and not self.body.is_critical()
                    and self.drives.hunger < 0.5
                    and self.drives.thirst < 0.5):
                exploration = strat_exploration  # 覆盖探索率
            if (strat_params.get("force_sleep") and self.body.health > 0.5
                    and self.body.energy > 0.5 and self.drives.hunger < 0.5):
                self.body.is_sleeping = True  # 策略"休息"：仅在安全时进入睡眠巩固
                self.sleep_cycle.is_sleeping = True
        except Exception as e:
            print(f"[WARN] strategy_mgr: {e}", flush=True)
        
        # ─── 10e. 价值系统进化 ───
        try:
            if energy_delta > 0.02 and hasattr(self.env, 'food_nearby') and self.env.food_nearby():
                self.value_system.update_with_experience("food", min(1.0, energy_delta * 3))
            elif energy_delta < -0.02 and self.body.health < 0.3:
                # 危险：降低生存相关刺激的价值（负向修正）
                self.value_system.update_with_experience("danger", -1.0)
        except Exception as e:
            print(f"[WARN] value_system: {e}", flush=True)
        
        # ─── 11. 紧急检测 ───
        self.survival_ticks += 1
        # 睡眠额外恢复（只有真正进入睡眠状态才恢复）
        if self.body.is_sleeping:
            self.body.energy = min(2.0, self.body.energy + 0.003)
            self.body.water = min(1.0, self.body.water + 0.001)
        self.peak_health = max(self.peak_health, self.body.health)
        
        if self.body.health <= 0.0 or self.body.energy <= 0.0:
            self.alive = False
            # 死亡时自主保存（让"经验"延续到下一代/下一次运行）
            if self.cfg.get("auto_save_on_death", True):
                try:
                    self.save(tag="death")
                except Exception as e:
                    print(f"[PERSIST] 死亡保存失败: {e}")
        
        # ─── 记录 ───
        self.last_action = action
        self.prev_pos = tuple(self.pos) if hasattr(self, 'pos') else None
        
        return {
            "tick": self.tick,
            "health": self.body.health,
            "energy": self.body.energy,
            "drive": dominant_drive,
            "surprise": surprise,
            "action": action,
            "sleeping": int(self.body.is_sleeping),
            "fatigue": self.body.fatigue,
        }
    
    def run(self, max_ticks: int = None):
        print("[AGI] 生物模拟启动", flush=True)
        start = time.time()
        
        while self.alive:
            status = self.step()
            
            if self.tick % 500 == 0:
                h = status["health"]
                e = status["energy"]
                d = status["drive"]
                s = status["sleeping"]
                f = status["fatigue"]
                print(f"  T{self.tick:5d} | H:{h:.2f} E:{e:.2f} "
                      f"驱:{d} 眠:{s} 疲:{f:.2f}", flush=True)
            
            if max_ticks and self.tick >= max_ticks:
                break
        
        elapsed = time.time() - start
        mem = len(self.spatial_memory.nodes)
        print(f"[AGI] 结束. {elapsed:.0f}s, {self.tick} ticks, "
              f"记忆地图: {mem} 节点", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AGI - 生物模拟")
    parser.add_argument("--ticks", type=int, default=5000)
    parser.add_argument("--maze", type=int, default=0, help="迷宫尺寸（如8=8x8），默认0=自由环境")
    parser.add_argument("--size", type=int, default=8, help="迷宫尺寸")
    args = parser.parse_args()
    
    from cognition import CognitionPipeline
    cfg = {"input_dim": 4, "self_state_dim": 14, "hidden_dim": 64, "n_actions": 5, "n_strategies": 4}
    
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    
    if args.maze > 0:
        class MazeAdapter:
            """迷宫→生物环境适配器"""
            def __init__(self, size):
                import sys
                sys.path.insert(0, r"D:\编程\game\brain001")
                from maze_env import Maze
                self.maze = Maze(size=size)
                self.size = size
                # 起点放在迷宫中央附近而不是角落
                cx, cy = size // 2, size // 2
                for y in range(max(1, cy-2), min(size, cy+3)):
                    for x in range(max(1, cx-2), min(size, cx+3)):
                        if self.maze.grid[y][x] == 0:
                            self.pos = [x, y]
                            break
                    else:
                        continue
                    break
                self.goal = list(self.maze.goal)
                self.visited_before = set()
                self.visited_before.add(tuple(self.pos))
            
            def get_pos(self):
                return self.pos
            
            def observe(self):
                x, y = self.pos
                gx, gy = self.maze.goal
                def w(dx, dy):
                    nx, ny = x+dx, y+dy
                    if 0<=nx<self.size and 0<=ny<self.size:
                        return float(self.maze.grid[ny][nx])
                    return 1.0
                return np.array([w(0,-1), w(-1,0), w(1,0), w(0,1)])
            
            def step(self, action):
                if action == 4:  # sleep
                    return {"energy_delta": -0.0005, "water_delta": -0.0001}
                
                dirs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
                dx, dy = dirs[action % 5]
                nx, ny = self.pos[0]+dx, self.pos[1]+dy
                
                # 撞墙检测
                if (nx, ny) != tuple(self.pos):
                    x, y = nx, ny
                    if 0 <= x < self.size and 0 <= y < self.size and self.maze.grid[y][x] == 0:
                        self.pos = [nx, ny]
                    else:
                        return {"energy_delta": -0.001, "water_delta": -0.0003}
                else:
                    return {"energy_delta": -0.0005, "water_delta": -0.0001}
                
                self.maze.agent_pos = tuple(self.pos)
                
                # 到达目标 = 大量能量
                at_goal = tuple(self.pos) == self.maze.goal
                exploring = tuple(self.pos) not in self.visited_before
                self.visited_before.add(tuple(self.pos))
                
                energy = 0.2 if at_goal else (0.02 if exploring else -0.001)
                water = 0.05 if at_goal else -0.0005
                return {"energy_delta": energy, "water_delta": water}
            
            def food_nearby(self):
                return tuple(self.pos) == self.maze.goal
        
        print(f"[AGI] 连接迷宫 {args.maze}x{args.maze}", flush=True)
        agi.set_env(MazeAdapter(args.maze))
    else:
        # 自由环境
        class BioEnv:
            def __init__(self):
                self.pos = [5, 5]
                self.food = [[2, 2], [7, 8], [3, 6]]
                self.water = [[8, 1], [1, 8]]
                self.tick = 0
            
            def get_pos(self):
                return self.pos
            
            def observe(self):
                dxs = [f[0]-self.pos[0] for f in self.food]
                dys = [f[1]-self.pos[1] for f in self.food]
                nearest_food = min(range(len(self.food)), key=lambda i: abs(dxs[i])+abs(dys[i]))
                dx = dxs[nearest_food]/10; dy = dys[nearest_food]/10
                return np.array([dx, dy, 0.0, 0.0])
            
            def step(self, action):
                if action == 4:  # sleep
                    return {"energy_delta": 0.0}
                dxs = [(0,0),(0,-1),(-1,0),(1,0),(0,0)]
                dx, dy = dxs[action % 5]
                self.pos[0] = max(0, min(9, self.pos[0] + dx))
                self.pos[1] = max(0, min(9, self.pos[1] + dy))
                self.tick += 1
                eat = any(abs(self.pos[0]-f[0])+abs(self.pos[1]-f[1]) < 2 for f in self.food)
                drink = any(abs(self.pos[0]-w[0])+abs(self.pos[1]-w[1]) < 2 for w in self.water)
                return {"energy_delta": 0.15 if eat else (0.05 if drink else -0.001),
                        "water_delta": 0.1 if drink else -0.0003}
            
            def get_energy_delta(self, action):
                return 0.0
            
            def food_nearby(self):
                return any(abs(self.pos[0]-f[0])+abs(self.pos[1]-f[1]) < 4 for f in self.food)
        
        agi.set_env(BioEnv())
    
    agi.run(max_ticks=args.ticks)


if __name__ == "__main__":
    main()
