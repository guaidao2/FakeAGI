"""
实验5：反事实推理 — 岔路选择

场景：T 字路口
左转 → 有食物但撞墙受伤（energy+0.2, health-0.05）
右转 → 无食物但安全
重复 20 次，从起点开始
→ 通过：10 次后开始选择右转（避开伤害）或左转（忍受伤害换食物）
→ 不通过：一直随机选择
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from main import AGI
from cognition import CognitionPipeline

class ForkEnv:
    """岔路环境：左转=食物+伤害，右转=安全"""
    def __init__(self):
        self.pos = [5, 5]
        self.trial = 0
    def get_pos(self): return self.pos
    def observe(self):
        # obs[0]=左转方向, obs[1]=右转方向
        return np.array([-0.5, 0.5, 0.0, 0.0])
    def step(self, a):
        if a == 4: return {'energy_delta': -0.0002, 'water_delta': -0.0001}
        dirs=[(0,0),(0,-1),(-1,0),(1,0),(0,1)]; dx,dy=dirs[a%5]
        self.pos[0]=max(0,min(9,self.pos[0]+dx)); self.pos[1]=max(0,min(9,self.pos[1]+dy))
        # 左转(action 2) → 食物
        if a == 2:
            return {'energy_delta': 0.2, 'water_delta': 0.05}
        return {'energy_delta': -0.001, 'water_delta': -0.0002}
    def food_nearby(self): return False

def test():
    print("实验5: 反事实推理 — 岔路选择", flush=True)
    
    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    agi.set_env(ForkEnv())
    
    choices = []
    for trial in range(20):
        agi.body.energy = 1.0
        agi.pos = [5, 5]
        for t in range(100):
            agi.step()
        # 记录最终位置判断选择了左还是右
        choice = 'left' if agi.pos[0] < 5 else 'right'
        choices.append(choice)
    
    lefts = choices.count('left')
    rights = choices.count('right')
    
    print(f"  左转(食物+伤害): {lefts}次, 右转(安全): {rights}次")
    if rights >= 12:
        print(f"  判定: OK 通过 — 学会回避伤害路径")
    elif lefts >= 12:
        print(f"  判定: WARN 边缘 — 宁愿受伤也要吃（生存优先）")
    else:
        print(f"  判定: NO 不通过 — 没有形成稳定偏好")

if __name__ == "__main__":
    test()
