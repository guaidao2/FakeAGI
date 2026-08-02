"""
探索目标生成器 — 元认知层

把知识缺口转化为可执行的子目标：
  1. 世界模型缺口 → "去没去过的地方，观察那里的结构"
  2. 策略缺口 → "尝试不同的动作序列，即使它不满足当前需求"
  3. 因果缺口 → "重复带因果关联的动作序列，观察结果是否一致"
  4. 惊奇缺口 → "回到上次高惊奇的位置重新观察"

每个目标有：类型、目标位置、目标动作、持续时间
"""

import numpy as np


class ExplorationGoal:
    """一个可执行的探索目标"""
    def __init__(self, kind: str, priority: float = 0.5,
                 target_pos=None, target_action=None,
                 duration: int = 50, context: str = ""):
        self.kind = kind          # "explore_novel" / "try_random" / "verify_causal" / "revisit"
        self.priority = priority  # 0~1
        self.target_pos = target_pos  # [x, y] or None
        self.target_action = target_action  # int or None
        self.duration = duration      # 计划持续多少 tick
        self.remaining = duration
        self.context = context
        self.activated = False
    
    def tick(self):
        self.remaining -= 1
        return self.remaining > 0
    
    def __repr__(self):
        return f"[GOAL {self.kind} p={self.priority:.2f} rem={self.remaining}]"


class GoalGenerator:
    def __init__(self, spatial_memory=None):
        self.spatial_memory = spatial_memory
        self.current_goal = None
    
    def generate(self, gap, agent_pos: list = None, env_size: int = 10) -> ExplorationGoal:
        """根据缺口生成目标"""
        if gap.kind == "world_model" or gap.kind == "surprise":
            # 找到未探索的方向（空间记忆引导——信息增益；随机回退）
            if self.spatial_memory is not None and hasattr(self.spatial_memory, 'get_exploration_target'):
                target = self.spatial_memory.get_exploration_target(
                    agent_pos=agent_pos, env_size=env_size)
                if target is not None:
                    return ExplorationGoal("explore_novel", gap.score * 0.8,
                                         target_pos=target, duration=80,
                                         context="explore_unvisited")
            # 无空间记忆或候选信息增益不足时，随机选择一个远程方向
            if agent_pos:
                dx = np.random.choice([-1, 0, 1]) * env_size * 0.3
                dy = np.random.choice([-1, 0, 1]) * env_size * 0.3
                target_x = np.clip(agent_pos[0] + dx, 0, env_size - 1)
                target_y = np.clip(agent_pos[1] + dy, 0, env_size - 1)
                return ExplorationGoal("explore_novel", gap.score * 0.8,
                                     target_pos=[int(target_x), int(target_y)],
                                     duration=60, context="random_explore")
            return ExplorationGoal("explore_novel", gap.score * 0.5, duration=40,
                                 context="random_walk")
        
        elif gap.kind == "strategy":
            # 策略缺口：尝试一个罕见的动作
            rare_action = np.random.randint(0, 5)
            return ExplorationGoal("try_random", gap.score * 0.7,
                                 target_action=int(rare_action), duration=30,
                                 context="try_rare_action")
        
        elif gap.kind == "causal":
            loc = gap.context.get("location", None)
            if gap.context.get("type") == "no_reward":
                rand_dir = np.random.randint(1, 5)
                return ExplorationGoal("try_random", gap.score,
                                     target_action=int(rand_dir), duration=80,
                                     context="break_no_reward_loop")
            if loc:
                return ExplorationGoal("verify_causal", gap.score,
                                     target_pos=loc, duration=100,
                                     context="verify_causal_chain")
            return ExplorationGoal("verify_causal", gap.score * 0.6, duration=60,
                                 context="generic_causal_check")
        
        return None  # 无合适的探索目标
