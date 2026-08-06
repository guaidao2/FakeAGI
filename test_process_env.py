"""
过程选择环境（ProcessEnv）— 问路 vs 扫掠

设计（DESIGN_PROCESS_SELECTION.md C1）：
  - 食物方向隐藏（信息隐藏，同 E12-S2）
  - 问路："food" → 环境回应方向词，但**有噪声**（失败率可配）
  - 问路 intrinsic 成本：问路占 1 tick（该 tick 无法移动 = 机会成本）
  - 环境响应失败 = 答非所问（给错方向）→ 系统落差未消解 → 可靠性下降

参数：
  - ask_noise: 响应失败率（0=有问必答，1=永远失败——N2）
"""
import numpy as np
from seed_utils import seed_run, get_seed_from_env
seed_run(get_seed_from_env(0))



class ProcessEnv:
    def __init__(self, size=10, ask_noise=0.3, seed=0):
        self.size = size
        self.ask_noise = ask_noise
        np.random.seed(seed)
        self.pos = [5, 5]
        self.food = [0, 0]
        self.eaten = 0
        self.steps = 0
        self.asked = 0          # 问路次数
        self.ask_correct = 0    # 问路得到正确方向的次数

    def get_pos(self): return self.pos

    def observe(self):
        """信息隐藏：空观测（食物方向隐藏）"""
        return np.array([0.0, 0.0])

    def respond(self, word: str) -> tuple:
        """问路响应（有噪声）：说 food → (方向词, 是否答对)"""
        self.asked += 1
        if word != "food":
            return [], False
        if np.random.random() < self.ask_noise:
            # 答非所问：随机给一个方向（误导）
            d = np.random.choice(["east", "west", "north", "south"])
            return ["food", d], False
        # 正确方向
        dx = self.food[0] - self.pos[0]
        dy = self.food[1] - self.pos[1]
        if abs(dx) >= abs(dy):
            d = "east" if dx > 0 else ("west" if dx < 0 else ("south" if dy > 0 else "north"))
        else:
            d = "south" if dy > 0 else "north"
        self.ask_correct += 1
        return ["food", d], True

    def step(self, a):
        """动作：0=stay 1=up 2=left 3=right 4=down"""
        self.steps += 1
        dirs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
        dx, dy = dirs[a % 5]
        self.pos[0] = max(0, min(self.size-1, self.pos[0]+dx))
        self.pos[1] = max(0, min(self.size-1, self.pos[1]+dy))
        d = abs(self.pos[0]-self.food[0]) + abs(self.pos[1]-self.food[1])
        if d < 2:
            ed = 0.3
            self.eaten += 1
            # 吃到后重置（防虚高）
            self.food = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        else:
            ed = -0.002
        return {'energy_delta': ed, 'water_delta': -0.0002}

    def get_energy_delta(self, a):
        d = abs(self.pos[0]-self.food[0]) + abs(self.pos[1]-self.food[1])
        return 0.3 if d < 2 else -0.002

    def get_damage(self, a): return 0.0

    def food_nearby(self):
        return abs(self.pos[0]-self.food[0]) + abs(self.pos[1]-self.food[1]) < 2
