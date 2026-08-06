"""
实验4：因果推理 — 隐藏规则

场景：迷宫中有一个"开关格"和一个"锁门格"
先去开关格→锁门格变为通路
不去开关格→锁门格是墙
系统需要学会两步因果链
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from main import AGI
from cognition import CognitionPipeline

def test():
    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    print("实验4: 因果推理 — 隐藏规则", flush=True)
    
    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    
    # 自定义环境：开关在(2,2)，锁门在(7,7)，食物在(9,9)
    # 必须先踩开关，锁门才打开
    class CausalEnv:
        def __init__(self):
            self.pos = [0, 0]
            self.switch_triggered = False
        def get_pos(self): return self.pos
        def observe(self):
            sx, sy = 2, 2; gx, gy = 5, 5
            to_switch = np.array([(sx-self.pos[0])/10, (sy-self.pos[1])/10])
            to_food = np.array([(gx-self.pos[0])/10, (gy-self.pos[1])/10])
            # 6D: 食物方向(2) + 开关方向(2) + 开关状态(1) + 锁门提示(1)
            return np.array([to_food[0], to_food[1],
                             to_switch[0], to_switch[1],
                             float(self.switch_triggered),
                             1.0 if not self.switch_triggered else 0.0])
        def step(self, a):
            # 注意：动作 4 = 向下移动（睡眠由 main loop 的 is_sleeping 状态处理）
            dirs=[(0,0),(0,-1),(-1,0),(1,0),(0,1)]
            dx,dy=dirs[a%5]
            self.pos[0]=max(0,min(9,self.pos[0]+dx)); self.pos[1]=max(0,min(9,self.pos[1]+dy))
            at_switch = abs(self.pos[0]-2)+abs(self.pos[1]-2)<2
            if at_switch: self.switch_triggered = True
            # 食物：只有在开关触发后才出现
            at_food = abs(self.pos[0]-5)+abs(self.pos[1]-5)<2 and self.switch_triggered
            return {'energy_delta':0.3 if at_food else -0.001,
                    'water_delta':0.05 if at_food else -0.0002}
        def food_nearby(self):
            return abs(self.pos[0]-5)+abs(self.pos[1]-5)<4 and self.switch_triggered
    
    agi.set_env(CausalEnv())
    
    max_ticks = 5000
    reached_switch = False
    reached_food = False
    
    for t in range(max_ticks):
        status = agi.step()
        if 40 <= t <= 55:
            print(f"  T{t}: pos={agi.pos} e={agi.body.energy:.3f} sleepy={agi.body.is_sleeping}", flush=True)
        if t % 1000 == 0 and t > 0:
            print(f"  T{t}: pos={agi.pos} e={agi.body.energy:.3f} sleepy={agi.body.is_sleeping} fatigue={agi.body.fatigue:.2f}", flush=True)
        if abs(agi.pos[0]-2)+abs(agi.pos[1]-2) < 2:
            if not reached_switch:
                print(f"  首次踩到开关: t={t}", flush=True)
            reached_switch = True
        if reached_switch and abs(agi.pos[0]-5)+abs(agi.pos[1]-5) < 2:
            reached_food = True
            print(f"  到达食物: t={t} (学会两步因果链)", flush=True)
            break
    
    print(f"  踩开关: {reached_switch}, 吃到食物: {reached_food}")
    if reached_switch and reached_food:
        print(f"  判定: OK 通过 — 系统学会【踩开关→门开→食物】因果链")
    elif reached_switch:
        print(f"  判定: WARN 边缘 — 找到开关但没连到食物")
    else:
        print(f"  判定: NO 不通过 — 没发现隐藏规则")

if __name__ == "__main__":
    test()
