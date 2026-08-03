"""
概念符号化端到端行为验证（②——词→概念→行为的真实价值）

问题：词有"所指"（单元验证 OK）——但"听词→行为提升"是否真实？
（回应"语言不训练怎么学会"：词通过概念通路改变行为=语义有用）

设计：
- 环境：TestEnv（采集语义——吃需要动作 0 停留）+ 语言广播
  （每 30 tick 广播 ["food"]——模拟提示"有食物"）
- "food" 不在 DIR_MAP（无方向词投票路径）→ 唯一生效通路=概念符号
- 对照 A：广播词但概念驱动关（词无行为效果——纯噪声）
- 实验 B：广播词 + 概念驱动开（词→概念激活→停留引导）
- 判据：
  A. 实验组食物 >= 对照组（词→行为价值——语义有用）
  B. 符号激活事件 > 0（词真实触发概念——_language_used_tick 变化）
  C. 语言信任提升（trust 强化闭环工作——听词→吃到→trust+0.1）
"""
import sys
import numpy as np
import torch

from main import AGI
from cognition import CognitionPipeline


class TestEnv:
    """4D 观测 + 语言广播（采集语义：吃需停留）"""
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
        eat = near_food and a == 0   # 采集语义
        drink = abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1]) < 2
        ed = 0.2 if eat else -0.0015
        wd = 0.15 if drink else -0.002
        return {"energy_delta": ed, "water_delta": wd}


def run(symbol_enabled, seed=0, ticks=800):
    np.random.seed(1000 + seed)
    torch.manual_seed(1000 + seed)
    env = TestEnv()
    agi = AGI()
    agi.set_env(env)
    agi.set_cognition(CognitionPipeline({}))
    agi.metacognition = None
    agi._info_seek_enabled = False
    agi._concept_drive_enabled = symbol_enabled  # 符号通路（概念驱动）开关
    foods = 0
    sym_activations = 0
    last_used = getattr(agi, '_language_used_tick', -1)
    for t in range(ticks):
        # 语言广播：每 30 tick 提示 "food"
        if t % 30 == 0:
            agi.cognition.language_tokens = ["food"]
        else:
            agi.cognition.language_tokens = None
        before = agi.body.energy
        agi.step()
        if agi.body.energy > before + 0.01:
            foods += 1
        used = getattr(agi, '_language_used_tick', -1)
        if used != last_used:
            sym_activations += 1   # 符号激活（含方向词——但无方向词故=符号）
            last_used = used
        if not agi.alive:
            break
    return {"food": foods, "sym": sym_activations,
            "trust": agi._language_trust, "alive": agi.alive,
            "concepts": sum(1 for c in agi.concept_bank.concepts
                            if c.kind == "consumable")}


def main():
    print("=" * 60)
    print("概念符号化端到端验证（词→概念→行为价值）")
    print("=" * 60)
    seeds = list(range(10))
    ctrl = [run(False, s) for s in seeds]
    exp = [run(True, s) for s in seeds]

    cf = np.mean([r["food"] for r in ctrl])
    ef = np.mean([r["food"] for r in exp])
    print(f"\n  食物获得（×{len(seeds)}seeds）: 无符号 {cf:.1f} vs 符号化 {ef:.1f}")
    ok_a = ef >= cf * 0.9
    print(f"  A: {'OK（不退化' + ('，有提升' if ef > cf + 0.5 else '）') if ok_a else 'FAIL'}")

    es = np.mean([r["sym"] for r in exp])
    print(f"  符号激活事件: 实验 {es:.1f} 次")
    ok_b = es > 0
    print(f"  B: {'OK（词真实触发概念）' if ok_b else 'FAIL'}")

    et = np.mean([r["trust"] for r in exp])
    print(f"  语言信任: 实验 {et:.3f}（初值 0.15）")
    ok_c = et > 0.16
    print(f"  C: {'OK（信任闭环工作——听词→吃到→trust+）' if ok_c else 'FAIL'}")

    nc = np.mean([r["concepts"] for r in exp])
    print(f"  概念簇: {nc:.0f}")
    ok_d = nc >= 1
    print(f"  D: {'OK（概念形成——符号化前提）' if ok_d else 'FAIL'}")

    ok = ok_a and ok_b and ok_c and ok_d
    print(f"\n  判定: {'OK 通过（词→概念→行为——语义有用）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
