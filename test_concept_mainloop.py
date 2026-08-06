"""
概念层接主循环验证（DESIGN_CONCEPTS §3 阶段 1 接入）

验证：
  A. 主循环跑后 concept_bank 形成 consumable 簇（价值锚聚类真实工作）
  B. 簇有区分度（不同 V 上升观测聚成不同簇/或一个稳定簇）
  C. 无回归：主循环正常运行（存活）
"""
import sys
import numpy as np
from main import AGI
from cognition import CognitionPipeline


def main():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))

    class TestEnv:
        def __init__(self):
            self.pos = [5, 5]
            self.food_pos = [2, 2]
            self.water_pos = [7, 7]
        def get_pos(self): return self.pos
        def observe(self):
            return np.array([(self.food_pos[0]-self.pos[0])/10,
                             (self.food_pos[1]-self.pos[1])/10,
                             (self.water_pos[0]-self.pos[0])/10,
                             (self.water_pos[1]-self.pos[1])/10])
        def step(self, a):
            dxs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
            dx, dy = dxs[a % 5]
            self.pos[0] = max(0, min(9, self.pos[0] + dx))
            self.pos[1] = max(0, min(9, self.pos[1] + dy))
            eat = abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1]) < 2
            drink = abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1]) < 2
            ed = 0.2 if eat else (0.05 if drink else -0.001)
            wd = 0.15 if drink else (0.02 if eat else -0.0005)
            return {"energy_delta": ed, "water_delta": wd}
    agi.set_env(TestEnv())

    for _ in range(600):
        agi.step()

    consumables = [c for c in agi.concept_bank.concepts
                   if c.kind == "consumable"]
    print(f"  A: consumable 簇数={len(consumables)}")
    for c in consumables:
        print(f"     {c.name} freq={c.freq}")
    ok_a = len(consumables) >= 1
    print(f"     {'OK（价值锚聚类在主循环真实形成概念）' if ok_a else 'FAIL'}")

    # B：簇区分度（多个簇时向量不同；单簇则验证 freq>1 有学习）
    if len(consumables) >= 2:
        d = np.linalg.norm(consumables[0].vector - consumables[1].vector)
        ok_b = d > 0.05
        print(f"  B: 簇间距离={d:.3f} "
              f"{'OK（有区分度）' if ok_b else 'FAIL'}")
    else:
        ok_b = consumables and consumables[0].freq >= 2
        print(f"  B: 单簇 freq={consumables[0].freq if consumables else 0} "
              f"{'OK（簇在学习）' if ok_b else 'FAIL'}")

    # C：无回归
    ok_c = agi.alive
    print(f"  C: 存活={agi.alive} {'OK' if ok_c else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    verdict = ("OK（概念层接入主循环：价值锚聚类真实形成可消耗物概念）"
               if ok else "FAIL")
    print("\n判定: " + verdict)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
