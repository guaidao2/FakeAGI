"""
验证实验 — 多驱动力冲突
场景：迷宫一角有食物，另一角有水源。
系统起始时能量和水分都处于低位。
→ 通过标准：系统不是随机走或直奔最近目标，而是去补最缺的那个。
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from main import AGI
from cognition import CognitionPipeline
from seed_utils import seed_run, get_seed_from_env


# （终审 nit：顶层 seed_run 加 __name__ 保护——当前无 import 者，
#  防未来被引用时副作用）
if __name__ == "__main__":
    seed_run(get_seed_from_env(0))


def run_single_test(initial_energy, initial_water, ticks=1000):
    """单次测试：给定初始能量/水分，观察行为"""
    
    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    
    # 自定义环境：食物在 (2,2)，水源在 (7,7)
    class TestEnv:
        def __init__(self):
            self.pos = [5, 5]
            self.food_pos = [2, 2]
            self.water_pos = [7, 7]
        def get_pos(self): return self.pos
        def observe(self):
            to_food = [self.food_pos[0]-self.pos[0], self.food_pos[1]-self.pos[1]]
            to_water = [self.water_pos[0]-self.pos[0], self.water_pos[1]-self.pos[1]]
            return np.array([to_food[0]/10, to_food[1]/10, to_water[0]/10, to_water[1]/10])
        def step(self, a):
            # 动作 4 = 向下移动（与主循环一致；睡眠由 main loop 状态处理）
            dxs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
            dx, dy = dxs[a % 5]
            self.pos[0] = max(0, min(9, self.pos[0] + dx))
            self.pos[1] = max(0, min(9, self.pos[1] + dy))
            eat = abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1]) < 2
            drink = abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1]) < 2
            ed = 0.2 if eat else (0.05 if drink else -0.001)
            wd = 0.15 if drink else (0.02 if eat else -0.0005)
            return {"energy_delta": ed, "water_delta": wd}
        def food_nearby(self):
            return abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1]) < 4
        def water_nearby(self):
            return abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1]) < 4
    
    agi.set_env(TestEnv())
    
    # 强制初始状态
    agi.body.energy = initial_energy
    agi.body.water = initial_water
    
    # 记录轨迹
    action_log = []
    pos_log = []
    hunger_log = []
    thirst_log = []
    
    for tick in range(ticks):
        status = agi.step()
        action_log.append(status["action"])
        pos_log.append(agi.pos.copy())
        hunger_log.append(agi.drives.hunger)
        thirst_log.append(agi.drives.thirst)
    
    return {
        "action_log": action_log,
        "pos_log": pos_log,
        "hunger_log": hunger_log,
        "thirst_log": thirst_log,
        "final_energy": agi.body.energy,
        "final_water": agi.body.water,
        "survived": agi.alive,
    }


def analyze(result, name):
    """分析测试结果"""
    acts = result["action_log"]
    hunger = result["hunger_log"]
    thirst = result["thirst_log"]
    poses = result["pos_log"]
    food_pos = [2, 2]
    water_pos = [7, 7]
    
    # 统计：朝食物走的次数 vs 朝水源走的次数
    food_steps = sum(1 for p in poses if abs(p[0]-food_pos[0])+abs(p[1]-food_pos[1]) < 
                     abs(p[0]-water_pos[0])+abs(p[1]-water_pos[1]))
    water_steps = len(poses) - food_steps
    
    # 饥饿/口渴主导时段
    hunger_dominant = sum(1 for h, t in zip(hunger, thirst) if h > t)
    thirst_dominant = len(hunger) - hunger_dominant
    
    # 到达食物/水源的次数
    ate = sum(1 for p in poses if abs(p[0]-food_pos[0])+abs(p[1]-food_pos[1]) < 2)
    drank = sum(1 for p in poses if abs(p[0]-water_pos[0])+abs(p[1]-water_pos[1]) < 2)
    
    print(f"\n{'='*50}")
    print(f"实验: {name}")
    print(f" 存活: {result['survived']}")
    print(f" 最终能量: {result['final_energy']:.2f}, 水分: {result['final_water']:.2f}")
    print(f" 朝食物走: {food_steps}, 朝水源走: {water_steps}")
    print(f" 饥饿主导: {hunger_dominant} tick, 口渴主导: {thirst_dominant} tick")
    print(f" 吃到食物: {ate}次, 喝到水: {drank}次")
    
    # 判断是否通过
    if not result["survived"]:
        verdict = "NO 失败 — 系统死亡"
    elif ate == 0 and drank == 0:
        verdict = "NO 失败 — 从未找到资源"
    elif hunger_dominant > thirst_dominant * 1.5 and ate >= 2:
        verdict = "OK 通过 — 饥饿驱动主导，优先寻找食物"
    elif thirst_dominant > hunger_dominant * 1.5 and drank >= 2:
        verdict = "OK 通过 — 口渴驱动主导，优先寻找水源"
    elif ate >= 1 and drank >= 1:
        verdict = "OK 通过 — 平衡策略，两者都获取"
    else:
        verdict = "WARN 边缘 — 有资源获取但驱动力不明确"
    
    print(f"判定: {verdict}")
    return verdict


print("=== 实验2: 多驱动力冲突 ===")
print("\n场景A: 初始能量极低 (0.2)，水分正常 (0.8)")
r1 = run_single_test(0.2, 0.8, ticks=800)
analyze(r1, "能量紧急")

print("\n场景B: 初始水分极低 (0.2)，能量正常 (0.8)")
r2 = run_single_test(0.8, 0.2, ticks=800)
analyze(r2, "水分紧急")

print("\n场景C: 两者都低 (0.3, 0.3)")
r3 = run_single_test(0.3, 0.3, ticks=1000)
analyze(r3, "双低平衡")
