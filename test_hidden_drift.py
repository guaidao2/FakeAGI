"""hidden 漂移诊断：LNN hidden 范数随时间变化（独立课题）"""
import numpy as np
import torch
from main import AGI
from cognition import CognitionPipeline


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
        ed = 0.2 if (near_food and a == 0) else -0.0015
        wd = 0.15 if abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1]) < 2 else -0.002
        return {"energy_delta": ed, "water_delta": wd}


np.random.seed(1001)
torch.manual_seed(1001)
env = TestEnv()
agi = AGI()
agi.set_env(env)
agi.set_cognition(CognitionPipeline({}))
agi.metacognition = None
agi._info_seek_enabled = False

norms = []
for t in range(500):
    agi.step()
    if agi.cognition.hidden is not None:
        h = agi.cognition.hidden.detach().cpu().numpy().flatten()
        norms.append(float(np.linalg.norm(h)))
    if not agi.alive:
        break

norms = np.array(norms)
if len(norms) > 50:
    early = np.mean(norms[:len(norms)//3])
    mid = np.mean(norms[len(norms)//3:2*len(norms)//3])
    late = np.mean(norms[2*len(norms)//3:])
    drift = late / max(early, 1e-9)
    print(f"hidden 范数: early={early:.2f} mid={mid:.2f} late={late:.2f} "
          f"（{len(norms)} ticks）")
    print(f"漂移比: {drift:.2f}x")
    print(f"  判定: {'OK（<1.3x 无显著漂移）' if drift < 1.3 else 'FAIL（显著漂移——' + f'{drift:.2f}' + 'x 需稳定化）'}")
