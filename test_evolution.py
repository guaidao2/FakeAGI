"""
P3: 种群演化循环 — 选择压力驱动的跨代进化

流程（每代）：
  1. 种群初始化：N 个个体，加载经验 DNA 作为先验（变异：扰动）
  2. 每个个体在自己的环境中"活着"（自维持循环）
  3. 个体死亡 → 记录 fitness（生存时长/峰值健康）
  4. 代结束 → 从种群中按 fitness 加权提取经验 DNA
  5. DNA 变异 → 下一代种群初始化
  6. 重复

验证指标：
  - 跨代平均存活时长上升（学习曲线）
  - DNA 规则/技能数量累计（知识积累）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from main import AGI
from cognition import CognitionPipeline
from core.dna import load_dna, save_dna, extract_dna, apply_dna


class EvolutionEnv:
    """演化测试环境：食物/水源随机分布，有危险区"""
    def __init__(self, size=10, seed=None):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.pos = [self.rng.integers(0, size), self.rng.integers(0, size)]
        self.food = [self.rng.integers(0, size), self.rng.integers(0, size)]
        self.water = [self.rng.integers(0, size), self.rng.integers(0, size)]
        self.danger_zone = self.rng.integers(0, size, size=2)
        self.food_qty = 3.0   # 食物有限量（会耗尽）
        self.water_qty = 3.0  # 水有限量
        self.tick = 0

    def get_pos(self):
        return self.pos

    def observe(self):
        return np.array([
            (self.food[0] - self.pos[0]) / self.size,
            (self.food[1] - self.pos[1]) / self.size,
            (self.water[0] - self.pos[0]) / self.size,
            (self.water[1] - self.pos[1]) / self.size,
            (self.danger_zone[0] - self.pos[0]) / self.size,
            (self.danger_zone[1] - self.pos[1]) / self.size,
            self.food_qty / 3.0,   # 食物剩余量（可观测）
            self.water_qty / 3.0,  # 水剩余量
        ], dtype=np.float32)

    def step(self, a):
        self.tick += 1
        if a == 4:
            # 睡眠：main.py 会 +0.003 恢复，这里 -0.0035 抵消并净消耗
            # （演化环境中睡眠不能无限续命，必须持续觅食）
            return {"energy_delta": -0.0035, "water_delta": -0.0008}
        dirs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
        dx, dy = dirs[a % 5]
        self.pos[0] = max(0, min(self.size-1, self.pos[0]+dx))
        self.pos[1] = max(0, min(self.size-1, self.pos[1]+dy))
        eat = abs(self.pos[0]-self.food[0]) + abs(self.pos[1]-self.food[1]) < 2 and self.food_qty > 0
        drink = abs(self.pos[0]-self.water[0]) + abs(self.pos[1]-self.water[1]) < 2 and self.water_qty > 0
        danger = abs(self.pos[0]-self.danger_zone[0]) + abs(self.pos[1]-self.danger_zone[1]) < 2
        ed = 0.20 if eat else (0.04 if drink else -0.002)
        wd = 0.12 if drink else (0.02 if eat else -0.001)
        if eat:
            self.food_qty -= 0.15  # 食物被吃会耗尽
        if drink:
            self.water_qty -= 0.12
        if danger:
            ed -= 0.15  # 危险区致命消耗
            wd -= 0.08
        # 随机灾害：每 150 tick 一次能量突降（环境不可预测性）
        if self.tick % 150 == 0:
            ed -= 0.05
        return {"energy_delta": ed, "water_delta": wd}

    def food_nearby(self):
        return abs(self.pos[0]-self.food[0]) + abs(self.pos[1]-self.food[1]) < 4

    def water_nearby(self):
        return abs(self.pos[0]-self.water[0]) + abs(self.pos[1]-self.water[1]) < 4


def run_life(max_ticks=1500, dna_loaded=False, seed=None) -> AGI:
    """让一个个体活一生，返回 AGI（含 fitness）"""
    agi = AGI(config={"auto_save_on_death": True})
    agi.set_cognition(CognitionPipeline({
        "input_dim": 8, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    # DNA 先验（变异：价值加扰动）
    if dna_loaded:
        apply_dna(agi)
        for name, entry in agi.value_system.secondary_values.items():
            entry["value"] = float(np.clip(entry["value"] + np.random.normal(0, 0.05), 0, 1))

    env = EvolutionEnv(size=10, seed=seed)
    agi.set_env(env)

    for _ in range(max_ticks):
        if not agi.alive:
            break
        agi.step()
    return agi


def evolve(generations=4, population=3, max_ticks=1200):
    """主演化循环（固定环境 seed：同代个体面对相同环境，跨代可对比）"""
    print("=" * 50)
    print("P3: 种群演化 — 跨代进化测试")
    print("=" * 50)

    history = []
    dna = load_dna()
    for gen in range(1, generations + 1):
        print(f"\n── 第 {gen} 代 ──")
        survivors = []
        # 同代个体共享相同环境 seed（公平对比），跨代用不同 seed（环境漂移）
        env_seed = 1000 + gen * 100
        for i in range(population):
            agi = run_life(max_ticks=max_ticks, dna_loaded=(gen > 1),
                           seed=env_seed)
            survivors.append(agi.survival_ticks)
            print(f"  个体{i+1}: 存活 {agi.survival_ticks} tick, "
                  f"峰值健康 {agi.peak_health:.2f}, "
                  f"生长 {agi.cognition.growth_count if agi.cognition else 0} 次")
            # 死亡时保存 checkpoint + 提取 DNA
            agi.save(tag=f"gen{gen}_ind{i+1}")
            extract_dna(agi, generation=gen)

        avg = float(np.mean(survivors))
        history.append(avg)
        print(f"  第 {gen} 代平均存活: {avg:.0f} tick")

    # 结果
    print("\n" + "=" * 50)
    print("演化曲线（平均存活时长/代）:")
    for g, h in enumerate(history, 1):
        marker = "↑" if g > 1 and h > history[g-2] else "→"
        print(f"  第 {g} 代: {h:.0f} tick {marker}")
    improved = len(history) >= 2 and history[-1] > history[0]
    print(f"判定: {'OK 通过 — 跨代改善' if improved else 'WARN 边缘 — 无显著改善'}")
    return history


if __name__ == "__main__":

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    evolve()
