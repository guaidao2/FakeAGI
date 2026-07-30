"""
实验6：规则变化适应

场景：水源在 (7,7) 稳定 2000 tick
然后水源突然移到 (2,2)
→ 通过：系统在 500 tick 内适应新水源位置
→ 不通过：一直跑去旧位置
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from main import AGI
from cognition import CognitionPipeline

def test():
    print("实验6: 规则变化适应", flush=True)
    
    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    
    class ChangeEnv:
        def __init__(self):
            self.pos = [5, 5]
            self.water_pos = [7, 7]
            self.phase = 1
        def get_pos(self): return self.pos
        def observe(self):
            wx, wy = self.water_pos
            return np.array([(wx-self.pos[0])/10, (wy-self.pos[1])/10, 0.0, 0.0])
        def step(self, a):
            if a == 4: return {'energy_delta': -0.0002, 'water_delta': -0.0001}
            dirs=[(0,0),(0,-1),(-1,0),(1,0),(0,1)]; dx,dy=dirs[a%5]
            self.pos[0]=max(0,min(9,self.pos[0]+dx)); self.pos[1]=max(0,min(9,self.pos[1]+dy))
            at_water = abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1])<2
            return {'energy_delta': 0.05 if at_water else -0.001,
                    'water_delta': 0.15 if at_water else -0.0005}
        def food_nearby(self): return False
    
    env = ChangeEnv()
    agi.set_env(env)
    
    # 阶段1：水源在 (7,7)
    water_found_old = 0
    for t in range(2000):
        agi.step()
        if abs(agi.pos[0]-7)+abs(agi.pos[1]-7) < 2:
            water_found_old += 1
    
    # 阶段2：水源移到 (2,2)
    env.water_pos = [2, 2]
    agi.body.water = 0.3  # 制造缺水压力
    
    first_found_new = None
    old_visits_after = 0
    for t in range(2000):
        agi.step()
        if first_found_new is None and abs(agi.pos[0]-2)+abs(agi.pos[1]-2) < 2:
            first_found_new = t
        if abs(agi.pos[0]-7)+abs(agi.pos[1]-7) < 2:
            old_visits_after += 1
    
    print(f"  阶段1: 找到旧水源 {water_found_old}次")
    print(f"  阶段2: 首次找到新水源 at t={first_found_new}")
    print(f"  阶段2: 仍去旧位置 {old_visits_after}次")
    
    if first_found_new is not None and first_found_new < 500:
        print(f"  判定: OK 通过 — 快速适应规则变化 ({first_found_new}tick)")
    elif first_found_new is not None:
        print(f"  判定: WARN 边缘 — 但适应较慢 ({first_found_new}tick)")
    else:
        print(f"  判定: NO 不通过 — 2000 tick 内未适应")

if __name__ == "__main__":
    test()
