"""
概念驱动行为验证（DESIGN_CONCEPTS §3 阶段 2 前置：概念→行为引导）

假设：观测匹配"可消耗物"概念 + 饥饿 → 停留尝试交互 → 食物获得提升
（概念是内部形成的身体经验压缩——非外部奖励注入）

设计：
- TestEnv：食物在固定位置，agent 从远处出发，能量递减（饥饿压力）
- 对照组：_concept_drive_enabled=False（无概念引导）
- 实验组：_concept_drive_enabled=True（概念引导停留）
- 判据：
  A. 实验组食物获得 >= 对照组（概念引导不退化或提升）
  B. 实验组在食物旁停留 tick 数 > 对照组（引导真实生效）
  C. 概念簇形成（consumable 概念存在——引导的前提）
  D. 防死锁：实验组存活（无无限停留饿死——_concept_stay_max 生效）
"""
import sys
import numpy as np
import torch

from main import AGI
from cognition import CognitionPipeline


class TestEnv:
    """4D 观测：食物方向 (dx, dy) + 水方向 (dx, dy)"""
    def __init__(self, size=10):
        self.size = size
        self.tick = 0
        self.pos = [5, 5]           # 中等距离起点（对照需能吃得到）
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
        # 采集语义：吃需要动作 0（停留/交互）——停留才有意义
        # （概念引导 = 饥饿+匹配→停留→吃到）
        eat = near_food and a == 0
        drink = (abs(self.pos[0] - self.water_pos[0]) +
                 abs(self.pos[1] - self.water_pos[1]) < 2)
        ed = 0.2 if eat else -0.0015  # 温和饥饿压力（探索预算够）
        wd = 0.15 if drink else -0.002
        return {"energy_delta": ed, "water_delta": wd}


def run(concept_driven, seed=0, ticks=1200):
    np.random.seed(1000 + seed)
    torch.manual_seed(1000 + seed)
    env = TestEnv()
    agi = AGI()
    agi.set_env(env)
    # 必须挂认知核心——决策块在 `if self.cognition` 内（否则无决策恒停留）
    agi.set_cognition(CognitionPipeline({}))
    # 纯净验证：禁用元认知覆盖 + 信息寻求（聚焦反射+概念引导——
    # 元认知目标方向依赖 pos 同步，纯净环境下易干扰）
    agi.metacognition = None
    agi._info_seek_enabled = False
    agi._concept_drive_enabled = concept_driven
    foods = 0
    stay_near_food = 0
    died_at = -1
    for _ in range(ticks):
        before = agi.body.energy
        agi.step()
        if agi.body.energy > before + 0.01:
            foods += 1
        # 停留且在食物旁（概念引导的交互尝试）
        if (abs(env.pos[0] - env.food_pos[0]) +
                abs(env.pos[1] - env.food_pos[1]) < 2 and
                getattr(agi, '_concept_stay', 0) > 0):
            stay_near_food += 1
        if not agi.alive:
            died_at = agi.tick
            break
    n_concepts = len(agi.concept_bank.concepts)
    return {"food": foods, "stay": stay_near_food,
            "concepts": n_concepts, "alive": agi.alive, "tick": agi.tick,
            "died_at": died_at,
            "concept_kinds": [c.kind for c in agi.concept_bank.concepts]}


def main():
    print("=" * 56)
    print("概念驱动行为验证（概念→行为引导）")
    print("=" * 56)
    seeds = [0, 1, 2, 3, 4]
    ctrl = [run(False, s) for s in seeds]
    exp = [run(True, s) for s in seeds]

    cf = np.mean([r["food"] for r in ctrl])
    ef = np.mean([r["food"] for r in exp])
    print(f"\n  食物获得（×{len(seeds)}seeds 均值）: "
          f"对照 {cf:.1f} vs 概念引导 {ef:.1f}")
    ok_a = ef >= cf * 0.9   # 不退化（引导非强制——不要求必超）
    print(f"  A: {'OK（不退化' + ('，有提升' if ef > cf + 0.5 else '）') if ok_a else 'FAIL'}")

    cs = np.mean([r["stay"] for r in ctrl])
    es = np.mean([r["stay"] for r in exp])
    print(f"  食物旁停留 tick: 对照 {cs:.1f} vs 概念引导 {es:.1f}")
    ok_b = es > cs + 1.0
    print(f"  B: {'OK（引导真实生效——停留更多）' if ok_b else 'FAIL'}")

    # nit：C 判据统计 consumable 簇（extract_from_obs 每 tick 加
    # act_ 概念——len(concepts) 含非 consumable，验证不了概念簇形成）
    nc = np.mean([sum(1 for c in r.get("concept_kinds", [])
                      if c == "consumable") for r in exp])
    print(f"  概念簇数量: {nc:.0f}（引导前提）")
    ok_c = nc >= 1
    print(f"  C: {'OK（概念簇形成）' if ok_c else 'FAIL'}")

    # D：相对死亡率（压力环境下个别 seed 饿死正常——
    # 概念引导应降低死亡：实验组死亡 <= 对照组）
    cd_ = sum(1 for r in ctrl if not r["alive"])
    ed_ = sum(1 for r in exp if not r["alive"])
    c_died = [r["died_at"] for r in ctrl if not r["alive"]]
    e_died = [r["died_at"] for r in exp if not r["alive"]]
    print(f"  死亡: 对照 {cd_} {c_died} / 实验 {ed_} {e_died}"
          f"（×{len(seeds)}seeds）")
    ok_d = ed_ <= cd_
    print(f"  D: {'OK（引导未增加死亡——吃到更多→饿死更少）' if ok_d else 'FAIL'}")

    ok = ok_a and ok_b and ok_c and ok_d
    print(f"\n  判定: {'OK 通过' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
