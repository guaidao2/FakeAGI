"""诊断：TestEnv 下 agent 实际轨迹/动作（概念驱动前置调试）"""
import numpy as np
import torch
from main import AGI


class TestEnv:
    def __init__(self, size=10):
        self.size = size
        self.tick = 0
        self.pos = [5, 5]
        self.food_pos = [1, 1]
        self.water_pos = [8, 1]

    def observe(self):
        return np.array([
            (self.food_pos[0]-self.pos[0])/self.size,
            (self.food_pos[1]-self.pos[1])/self.size,
            (self.water_pos[0]-self.pos[0])/self.size,
            (self.water_pos[1]-self.pos[1])/self.size], dtype=np.float32)

    def get_pos(self):
        return self.pos

    def step(self, a):
        self.tick += 1
        dxs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
        dx, dy = dxs[a % 5]
        self.pos[0] = max(0, min(self.size-1, self.pos[0]+dx))
        self.pos[1] = max(0, min(self.size-1, self.pos[1]+dy))
        near_food = abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1]) < 3
        ed = 0.2 if near_food else -0.0015
        wd = 0.15 if abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1]) < 2 else -0.002
        return {"energy_delta": ed, "water_delta": wd}


np.random.seed(1001)
torch.manual_seed(1001)
env = TestEnv()
agi = AGI()
agi.set_env(env)
from cognition import CognitionPipeline
agi.set_cognition(CognitionPipeline({}))
agi._concept_drive_enabled = True
# 纯净验证：禁用元认知覆盖 + 信息寻求（聚焦反射+概念引导）
agi.metacognition = None
agi._info_seek_enabled = False
from collections import Counter
acts = Counter()
dist_min = 99
for t in range(400):
    obs_before = env.observe()
    a = 0
    try:
        acts[a] += 1
    except Exception:
        pass
    agi.step()
    dist = abs(env.pos[0]-env.food_pos[0])+abs(env.pos[1]-env.food_pos[1])
    dist_min = min(dist_min, dist)
    if t < 30 or (t >= 50 and t < 70) or (t > 100 and t < 130):
        m = agi.concept_bank.match_concept(env.observe())
        print(f"t={t:3d} pos={env.pos} e={agi.body.energy:.2f} "
              f"match={m} stay={getattr(agi,'_concept_stay',0)} "
              f"food_d={abs(env.pos[0]-env.food_pos[0])+abs(env.pos[1]-env.food_pos[1])}")
    if not agi.alive:
        print(f"DIED at t={t} pos={env.pos} dist={dist}")
        break
print(f"min_dist={dist_min} final_pos={env.pos} energy={agi.body.energy:.2f}")
