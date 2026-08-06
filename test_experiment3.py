"""
实验3：空间记忆利用测试（修复版）

场景：10x10 自由环境，食物固定在 (9,9)
系统从 (0,0) 出发，到达后重置回 (0,0)
跑 10 轮（空间记忆跨轮持续积累）
→ 通过：第10轮路径比第1轮短 30% 以上
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from main import AGI
from cognition import CognitionPipeline


class Env:
    """自由环境，食物固定位置，动作 4 = 向下移动"""
    def __init__(self):
        self.pos = [0, 0]
    def get_pos(self): return self.pos
    def observe(self):
        dx = (9 - self.pos[0]) / 10; dy = (9 - self.pos[1]) / 10
        return np.array([dx, dy, 0.0, 0.0])
    def step(self, a):
        dirs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
        dx, dy = dirs[a % 5]
        self.pos[0] = max(0, min(9, self.pos[0] + dx))
        self.pos[1] = max(0, min(9, self.pos[1] + dy))
        at = abs(self.pos[0]-9) + abs(self.pos[1]-9) < 2
        return {"energy_delta": 0.3 if at else -0.001,
                "water_delta": 0.05 if at else -0.0002}
    def food_nearby(self):
        return abs(self.pos[0]-9) + abs(self.pos[1]-9) < 4


def test():
    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    rounds = 10
    max_steps = 1000
    
    print("实验3: 空间记忆利用测试")
    print(f"起点(0,0)->食物(9,9), {rounds}轮, 上限{max_steps}步/轮")
    
    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    
    results = []
    
    for r in range(rounds):
        env = Env()
        agi.set_env(env)
        agi.body.energy = 1.0
        agi.body.water = 1.0
        
        for tick in range(max_steps):
            agi.step()
            if abs(agi.pos[0]-9)+abs(agi.pos[1]-9) < 2:
                results.append({
                    "round": r+1, "steps": tick+1, "reached": True,
                    "nodes": len(agi.spatial_memory.nodes),
                })
                print(f"  轮{r+1:2d}: OK  {tick+1:4d}步, 记忆{len(agi.spatial_memory.nodes):2d}节点")
                break
        else:
            results.append({
                "round": r+1, "steps": max_steps, "reached": False,
                "nodes": len(agi.spatial_memory.nodes),
            })
            print(f"  轮{r+1:2d}: NO  {max_steps}步, 未到达")
    
    # 分析
    reached = [r for r in results if r["reached"]]
    if len(reached) < 2:
        print(f"\n  判定: NO — 仅{len(reached)}/10轮到达，无法评估")
        return
    
    first = reached[0]["steps"]
    last = reached[-1]["steps"]
    improvement = (1 - last / first) * 100
    
    print(f"\n{'='*40}")
    print(f"  第1次: {first}步")
    print(f"  第10次: {last}步")
    print(f"  改进: {improvement:.1f}%")
    print(f"  判定: {'OK 通过' if improvement >= 30 else 'WARN 边缘' if improvement >= 10 else 'NO 不通过'}")


if __name__ == "__main__":
    test()
