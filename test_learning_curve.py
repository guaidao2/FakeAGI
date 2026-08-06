"""
D：量化诊断——装配后系统到底在学吗（DESIGN_CONCEPTS §8 后续）

B1-B3 接线后，测量学习信号：
  A. world_loss 随 tick 是否下降（之前恒 0.5 兜底=信号空转）
  B. GameNN 置信度是否上升（策略是否真的学会）
  C. 存活/觅食（行为层面）
判定：world_loss 显著下降 = 引擎有燃料；否则装配了但没学起来
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

    n_ticks = 1000
    wl_hist, conf_hist, food_hist = [], [], []
    foods = 0
    for t in range(n_ticks):
        before = agi.body.energy
        agi.step()
        if agi.body.energy > before + 0.01:
            foods += 1
        if (t + 1) % 100 == 0:
            wl = getattr(agi, '_last_world_loss', None)
            if wl is None:
                # 从 curiosity 或认知取 world_loss 近似
                wl = (agi.curiosity._loss_hist[-1]
                      if agi.curiosity and agi.curiosity._loss_hist else None)
            conf = 0.0
            try:
                if hasattr(agi.cognition, 'gamenn'):
                    conf = agi.cognition.gamenn.get_confidence()
            except Exception:
                pass
            wl_hist.append(wl if wl is not None else float('nan'))
            conf_hist.append(conf)
            food_hist.append(foods)
            print(f"  t={t+1:4d} | world_loss={wl if wl is not None else 'N/A':>6} "
                  f"| conf={conf:.3f} | foods={foods}")

    # 判定：后半 world_loss 显著低于前半（排除 NaN）
    valid = [w for w in wl_hist if w == w]  # NaN 过滤
    early = np.mean(wl_hist[:len(wl_hist)//2])
    late = np.mean(wl_hist[len(wl_hist)//2:])
    print(f"\n  world_loss: early={early:.4f} late={late:.4f} "
          f"({early/max(late,1e-9):.2f}x)")
    ok_a = late < early * 0.8 if late > 0 else False
    ok_b = conf_hist[-1] > conf_hist[0] + 0.01
    print(f"  置信度: 首={conf_hist[0]:.3f} 末={conf_hist[-1]:.3f}")
    print(f"  食物: {foods} 次 / {n_ticks} tick")
    print(f"\nA(world_loss 下降)={'OK（引擎有燃料）' if ok_a else 'FAIL（信号仍空转——A 步骤必要）'} "
          f"B(置信度上升)={'OK' if ok_b else 'FAIL'}")
    return 0 if (ok_a or ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
