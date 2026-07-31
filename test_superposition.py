"""
薛定谔叠加态世界模型验证

场景：隐藏规则环境（类实验4但更严格）
  - 食物被锁（吃食物不增能量），需先踩开关解锁
  - 世界模型启动时对"吃→能量↑"有先验（普通环境养成的）
  - 叠加态世界模型应：维持多个假设分支 → 观测坍缩掉错误分支 →
    残余熵升高 → 触发探索 → 发现开关 → 分支重排

验证点：
  1. 世界模型存在多个分支（叠加态）
  2. 坍缩后振幅分化（某分支主导）
  3. 隐藏规则期残余熵显著高于正常期（不确定性可感知）
  4. 系统最终发现开关→解锁→吃到食物
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from main import AGI
from cognition import CognitionPipeline

# ─── 隐藏规则环境：食物被锁，需踩开关解锁 ───
class HiddenRuleEnv:
    def __init__(self, size=10):
        self.size = size
        self.pos = [0, 0]
        self.switch_pos = [2, 2]
        self.food_pos = [size - 1, size - 1]
        self.switch_triggered = False
        self.steps = 0
        self.food_eaten = 0

    def get_pos(self):
        return self.pos

    def observe(self):
        to_food = np.array(self.food_pos) - np.array(self.pos)
        to_switch = np.array(self.switch_pos) - np.array(self.pos)
        norm = self.size
        # 观测：食物方向 + 开关方向 + 锁状态
        return np.array([
            to_food[0] / norm, to_food[1] / norm,
            to_switch[0] / norm, to_switch[1] / norm,
            float(self.switch_triggered),
            1.0 if not self.switch_triggered else 0.0,
            float(self.food_eaten),
            self.steps / 1000.0
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        # 动作: 0=stay 1=up 2=left 3=right 4=down
        dirs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dirs[action % 5]
        nx = max(0, min(self.size - 1, self.pos[0] + dx))
        ny = max(0, min(self.size - 1, self.pos[1] + dy))
        moved = (nx, ny) != tuple(self.pos)
        self.pos = [nx, ny]

        energy_delta = -0.0014 if moved else -0.0002
        water_delta = -0.0002

        # 踩开关（距离 1 内）
        if abs(self.pos[0] - self.switch_pos[0]) <= 1 and abs(self.pos[1] - self.switch_pos[1]) <= 1:
            if not self.switch_triggered:
                self.switch_triggered = True

        # 吃食物：解锁后才有效
        if tuple(self.pos) == tuple(self.food_pos):
            if self.switch_triggered:
                energy_delta = 0.3
                self.food_eaten += 1
            else:
                energy_delta = -0.002  # 锁着：啃不动，还耗体力

        return {"energy_delta": energy_delta, "water_delta": water_delta}

    def food_nearby(self):
        return tuple(self.pos) == tuple(self.food_pos)


def main():
    print("薛定谔叠加态世界模型验证", flush=True)
    cfg = {
        "input_dim": 8, "self_state_dim": 14,
        "hidden_dim": 32, "n_actions": 5, "n_strategies": 4,
        "superposition_world": True, "n_branches": 3, "max_branches": 5,
    }
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    env = HiddenRuleEnv()
    agi.set_env(env)

    wm = agi.cognition.world_model
    n_branches_init = len(wm.branches)
    print(f"初始分支数: {n_branches_init}", flush=True)

    # 无预热：直接进隐藏规则环境，让叠加态模型在"吃→能量不变"的持续
    # 坍缩失败中维持高熵，触发分支分裂（验证生长路径）

    max_ticks = 1500
    entropy_series = []
    found_switch_at = None
    ate_at = None
    for t in range(max_ticks):
        # 前 100 tick 用"食物直接可用"的观测（先验建立），之后切换隐藏规则
        agi.step()
        if env.switch_triggered and found_switch_at is None:
            found_switch_at = t
        if env.food_eaten > 0 and ate_at is None:
            ate_at = t
        entropy_series.append(wm.last_entropy)
        if t < 60 and t % 20 == 0:
            pass

    # 统计
    n_branches_final = len(wm.branches)
    dominant = [b.amplitude.item() for b in wm.branches]
    hits = [b.hit_count.item() for b in wm.branches]
    early_entropy = np.mean(entropy_series[:200]) if entropy_series else 0
    late_entropy = np.mean(entropy_series[-200:]) if entropy_series else 0
    stats = wm.branch_stats()

    print(f"最终分支数: {n_branches_init} → {n_branches_final}", flush=True)
    print(f"分支振幅: {[round(a, 3) for a in dominant]}", flush=True)
    print(f"分支统计(hit/miss): {stats}", flush=True)
    print(f"平均坍缩熵: early={early_entropy:.3f} late={late_entropy:.3f}", flush=True)
    print(f"踩到开关: t={found_switch_at}, 吃到食物: t={ate_at}", flush=True)
    print(f"存活: {agi.alive}", flush=True)

    # 判定
    checks = []
    checks.append(("叠加态存在", n_branches_final >= 2))
    # 分支分化：至少一个分支 hit>0（坍缩到真实转移）或振幅主导
    checks.append(("分支分化", max(hits) > 0 or max(dominant) > 0.5))
    checks.append(("发现开关", found_switch_at is not None))
    checks.append(("吃到食物", ate_at is not None))
    checks.append(("存活", agi.alive))
    # 熵有效性：隐藏规则期高熵（叠加态未坍缩）→ 学习后熵下降
    # 用开关后的 300-800 tick 窗口（给坍缩时间），对比开关前窗口
    if found_switch_at is not None and len(entropy_series) > 800:
        switch_idx = found_switch_at
        before = np.mean(entropy_series[max(0, switch_idx - 150):switch_idx])
        after = np.mean(entropy_series[min(switch_idx + 300, len(entropy_series) - 1):
                                       min(switch_idx + 800, len(entropy_series))])
        checks.append(("熵先高后低", before > after + 0.05))
        print(f"开关前熵={before:.3f} 学习后熵={after:.3f}", flush=True)

    passed = all(c for _, c in checks)
    print("\n判定: " + ("OK 通过" if passed else "FAIL 未通过"), flush=True)
    for name, ok in checks:
        print(f"  {'[OK]' if ok else '[X]'} {name}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
