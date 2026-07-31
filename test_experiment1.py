"""
实验1：新奇环境适应

场景：先在 8x8 自由环境跑 500 tick，然后突然换成 16x16
→ 通过：500 tick 内适应，不饿死
→ 不通过：卡在旧路径模式中饿死
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from main import AGI
from cognition import CognitionPipeline

def test():
    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    
    results = {}
    
    # 阶段1：8x8 环境
    class Env8:
        def __init__(self): self.pos=[3,3]; self.food=[7,7]
        def get_pos(self): return self.pos
        def observe(self):
            dx=(self.food[0]-self.pos[0])/8; dy=(self.food[1]-self.pos[1])/8
            return np.array([dx,dy,0.,0.])
        def step(self,a):
            dirs=[(0,0),(0,-1),(-1,0),(1,0),(0,1)]; dx,dy=dirs[a%5]
            self.pos[0]=max(0,min(7,self.pos[0]+dx)); self.pos[1]=max(0,min(7,self.pos[1]+dy))
            at=abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])<2
            return {'energy_delta':0.3 if at else -0.001,'water_delta':0.05 if at else -0.0002}
        def food_nearby(self): return abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])<4
    
    agi.set_env(Env8())
    for t in range(500): agi.step()
    e8 = agi.body.energy
    results['phase1_energy'] = e8
    
    # 阶段2：突然换到 16x16
    class Env16:
        def __init__(self): self.pos=[0,0]; self.food=[15,15]
        def get_pos(self): return self.pos
        def observe(self):
            dx=(self.food[0]-self.pos[0])/16; dy=(self.food[1]-self.pos[1])/16
            return np.array([dx,dy,0.,0.])
        def step(self,a):
            dirs=[(0,0),(0,-1),(-1,0),(1,0),(0,1)]; dx,dy=dirs[a%5]
            self.pos[0]=max(0,min(15,self.pos[0]+dx)); self.pos[1]=max(0,min(15,self.pos[1]+dy))
            at=abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])<2
            return {'energy_delta':0.3 if at else -0.001,'water_delta':0.05 if at else -0.0002}
        def food_nearby(self): return abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])<4
    
    agi.set_env(Env16())
    agi.body.energy = min(1.0, e8)
    start = time.time()
    for t in range(1000):
        status = agi.step()
        if abs(agi.pos[0]-15)+abs(agi.pos[1]-15) < 2:
            results['phase2_reached_at'] = t
            break
    else:
        results['phase2_reached_at'] = None
    
    results['phase2_energy'] = agi.body.energy
    results['phase2_alive'] = agi.alive
    
    # 判定
    print(f"实验1: 新奇环境适应")
    print(f"  阶段1 (8x8) 后能量: {results['phase1_energy']:.2f}")
    print(f"  阶段2 (16x16) 到达: {results['phase2_reached_at']}")
    print(f"  阶段2 后能量: {results['phase2_energy']:.2f}")
    print(f"  存活: {results['phase2_alive']}")
    if results['phase2_reached_at'] is not None:
        print(f"  判定: OK 通过 — {results['phase2_reached_at']} tick 内适应新环境")
    elif results['phase2_alive']:
        print(f"  判定: WARN 边缘 — 存活但未找到食物")
    else:
        print(f"  判定: NO 不通过 — 环境切换后死亡")

if __name__ == "__main__":
    test()
