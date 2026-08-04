"""
概念门控模式对照实验（护栏裁决——③ 动机选择器 vs 护栏字面）

问题：概念驱动（匹配可消耗物→action=0 停留）是"价值直接驱动动作"——
DESIGN_CONCEPTS §4 护栏"价值只喂学习系统不直接驱动动作"的张力点。
③ 公理允许"情感/动机=预测误差加权机制（选择器）"——概念驱动可解释为
动机选择器（公理内）或护栏违反（字面）。

实验（n=10 seeds，完全复用 test_concept_driven 已验证环境——
吃需停留 a==0、固定食物、1200 tick）：
  A. direct 模式：概念匹配 → 直接 action=0（当前实现——价值直驱）
  B. hint 模式：概念匹配信号作为 GameNN 观测额外维（价值只喂学习
     系统）——动作由委员会加权（habit 投票经 GameNN TD 学习）

判据：
  A. hint 食物 >= direct 食物（价值经学习系统可行——护栏字面可守）
     direct > hint（直控必要——③ 动机选择器裁决）→ 均如实报告
  B. 两种模式均存活（无死锁）
"""
import os
import sys
import numpy as np
import torch

from main import AGI
from cognition import CognitionPipeline


class TestEnv:
    """4D 观测：食物方向 (dx, dy) + 水方向 (dx, dy)——与 test_concept_driven
    完全一致（吃需停留 a==0——概念引导=饥饿+匹配→停留→吃到）"""
    def __init__(self, size=10):
        self.size = size
        self.tick = 0
        self.pos = [5, 5]
        self.food_pos = [1, 1]
        self.water_pos = [8, 1]

    def observe(self):
        dx = self.food_pos[0] - self.pos[0]
        dy = self.food_pos[1] - self.pos[1]
        wx = self.water_pos[0] - self.pos[0]
        wy = self.water_pos[1] - self.pos[1]
        return np.array([dx / self.size, dy / self.size,
                         wx / self.size, wy / self.size], dtype=np.float32)

    def get_pos(self):
        return self.pos

    def step(self, a):
        self.tick += 1
        dxs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dxs[a % 5]
        self.pos[0] = max(0, min(self.size - 1, self.pos[0] + dx))
        self.pos[1] = max(0, min(self.size - 1, self.pos[1] + dy))
        near_food = (abs(self.pos[0] - self.food_pos[0]) +
                     abs(self.pos[1] - self.food_pos[1]) < 3)
        eat = near_food and a == 0
        drink = (abs(self.pos[0] - self.water_pos[0]) +
                 abs(self.pos[1] - self.water_pos[1]) < 2)
        ed = 0.2 if eat else -0.0015
        wd = 0.15 if drink else -0.002
        return {"energy_delta": ed, "water_delta": wd}


def run(mode, seed=0, ticks=1200):
    os.environ["CONCEPT_GATE_MODE"] = mode
    np.random.seed(1000 + seed)
    torch.manual_seed(1000 + seed)
    env = TestEnv()
    agi = AGI()
    agi.set_env(env)
    agi.set_cognition(CognitionPipeline({}))
    agi.metacognition = None
    agi._info_seek_enabled = False
    foods = 0
    for _ in range(ticks):
        before = agi.body.energy
        agi.step()
        if agi.body.energy > before + 0.01:
            foods += 1
        if not agi.alive:
            break
    return {"food": foods, "alive": agi.alive, "tick": agi.tick}


def main():
    print("=" * 56)
    print("概念门控模式对照（护栏裁决：direct vs hint）")
    print("=" * 56)
    seeds = list(range(10))
    dir_r = [run("direct", s) for s in seeds]
    hint_r = [run("hint", s) for s in seeds]
    df = np.mean([r["food"] for r in dir_r])
    hf = np.mean([r["food"] for r in hint_r])
    dalive = sum(1 for r in dir_r if r["alive"])
    halive = sum(1 for r in hint_r if r["alive"])
    print(f"  A(direct 直控): 食物 {df:.1f}±{np.std([r['food'] for r in dir_r]):.1f}"
          f" | 存活 {dalive}/10")
    print(f"  B(hint 喂GameNN): 食物 {hf:.1f}±{np.std([r['food'] for r in hint_r]):.1f}"
          f" | 存活 {halive}/10")
    if hf >= df:
        verdict = "hint≥direct——价值经学习系统可行（护栏字面可守）"
    else:
        verdict = "direct>hint——价值直驱必要（③ 动机选择器裁决成立）"
    print(f"  裁决: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
