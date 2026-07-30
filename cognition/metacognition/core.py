"""
元认知层 — 第五级认知

整体循环：
  每 tick:
    1. GapDetector 从认知管线收集信号 → 输出最显著的知识缺口
    2. GoalGenerator 将缺口转换为探索目标
    3. CuriosityScheduler 决定"现在探索还是利用"
    4. 若是探索模式，GoalGenerator 输出 override_action / override_target
    5. SelfAssessment 每 N tick 做一次综合自我评估
    
  输出给主循环的接口：
    explore_mode: bool       → 是否处于探索模式
    override_action: int     → 探索模式下覆盖的动作（或 -1 表示不覆盖）
    override_target: [x,y]   → 探索目标位置（供导航系统使用）
    exploration_ratio: float → 建议的探索率
"""

import numpy as np
from cognition.metacognition import GapDetector, KnowledgeGap
from cognition.metacognition.goal_gen import GoalGenerator, ExplorationGoal
from cognition.metacognition.scheduler import CuriosityScheduler
from cognition.metacognition.assess import SelfAssessment


class MetacognitionLayer:
    def __init__(self, spatial_memory=None):
        self.gap_detector = GapDetector()
        self.goal_generator = GoalGenerator(spatial_memory=spatial_memory)
        self.scheduler = CuriosityScheduler()
        self.assessor = SelfAssessment()
        
        self.current_goal = None
        self.explore_mode = False
        self.override_action = -1
        self.override_target = None
        self.tick = 0
        self.give_up_count = {}
    
    def update(self, world_model_loss: float, gamenn_confidence: float,
               surprise: float, health: float, energy: float, 
               action: int, survived: bool, 
               causal_error: float = None, agent_pos: list = None,
               env_size: int = 10, energy_delta: float = 0.0):
        self.tick += 1
        
        # 1. 收集信号到缺口检测器
        self.gap_detector.update(world_model_loss, gamenn_confidence, surprise, causal_error,
                               energy_delta=energy_delta, agent_pos=agent_pos,
                               energy_level=energy)
        
        # 2. 记录到自我评估
        self.assessor.record(health, energy, surprise, action, survived)
        
        # 3. 检测当前是否有显著缺口
        gap = self.gap_detector.detect()
        gap_exists = gap is not None
        
        # 4. 检查当前目标是否需要放弃
        goal_active = self.current_goal is not None and self.current_goal.remaining > 0
        if self.current_goal is not None:
            goal_key = f"{self.current_goal.kind}_{self.tick // 100}"
            if self.scheduler.should_give_up(goal_key):
                self.current_goal = None
                self.explore_mode = False
        
        # 5. 如果没有活跃目标且有缺口 → 生成新目标
        if not goal_active and gap_exists:
            self.current_goal = self.goal_generator.generate(
                gap, agent_pos=agent_pos, env_size=env_size)
            if self.current_goal is not None:
                self.explore_mode = True
        
        # 6. 更新好奇心调度器
        self.scheduler.update(health, gap_exists, goal_active, gamenn_confidence)
        
        # 7. 探索/利用决策
        if self.explore_mode:
            # 在探索模式下，是否真的探索取决于调度器
            self.scheduler.should_explore()
        
        # 8. 生成输出
        self._compute_outputs(agent_pos)
        
        # 9. 定期自我评估
        if self.tick % 500 == 0:
            self.assessor.assess()
    
    def _compute_outputs(self, agent_pos: list = None):
        """计算给主循环的 override 信号"""
        self.override_action = -1
        self.override_target = None
        
        if self.current_goal is None or not self.explore_mode:
            self.exploration_ratio = self.scheduler.get_balance()
            return
        
        if self.current_goal.target_pos is not None and agent_pos is not None:
            self.override_target = self.current_goal.target_pos
        
        if self.current_goal.target_action is not None:
            self.override_action = self.current_goal.target_action
        
        self.current_goal.remaining -= 1
        if self.current_goal.remaining <= 0:
            self.current_goal = None
            self.explore_mode = False
        
        self.exploration_ratio = self.scheduler.get_balance()
    
    def get_state(self) -> dict:
        return {
            "explore_mode": self.explore_mode,
            "override_action": self.override_action,
            "override_target": self.override_target,
            "exploration_ratio": self.scheduler.get_balance(),
            "competence": self.assessor.competence,
            "trend": self.assessor.trend,
            "gap_detected": self.gap_detector.detect() is not None,
            "goal": str(self.current_goal) if self.current_goal else "none",
        }
