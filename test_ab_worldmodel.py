"""
A/B 对比：确定性世界模型 vs 薛定谔叠加态世界模型

同一 HiddenRuleEnv（食物被锁→踩开关解锁），同一 AGI 配置，
仅切换 superposition_world。各跑 N 轮，统计：
  - 学习速度：首次吃到食物的 tick（越早越快）
  - 认知区分度：熵/惊喜度 在"隐藏规则期 vs 学习后"的差（越大越能感知不确定性）
  - 可增长性：分支是否分裂（叠加态独有）
  - 健壮性：多轮成功率
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from main import AGI
from cognition import CognitionPipeline


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
        dirs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dirs[action % 5]
        nx = max(0, min(self.size - 1, self.pos[0] + dx))
        ny = max(0, min(self.size - 1, self.pos[1] + dy))
        moved = (nx, ny) != tuple(self.pos)
        self.pos = [nx, ny]
        energy_delta = -0.0014 if moved else -0.0002
        water_delta = -0.0002
        if abs(self.pos[0] - self.switch_pos[0]) <= 1 and abs(self.pos[1] - self.switch_pos[1]) <= 1:
            if not self.switch_triggered:
                self.switch_triggered = True
        if tuple(self.pos) == tuple(self.food_pos):
            if self.switch_triggered:
                energy_delta = 0.3
                self.food_eaten += 1
            else:
                energy_delta = -0.002
        return {"energy_delta": energy_delta, "water_delta": water_delta}

    def food_nearby(self):
        return tuple(self.pos) == tuple(self.food_pos)


def run_once(superposition: bool, max_ticks: int = 1200, seed: int = 0):
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    cfg = {
        "input_dim": 8, "self_state_dim": 14,
        "hidden_dim": 32, "n_actions": 5, "n_strategies": 4,
        "superposition_world": superposition,
        "n_branches": 3, "max_branches": 5,
    }
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    env = HiddenRuleEnv()
    agi.set_env(env)
    ate_at = None
    surprise_series = []
    last_buf_len = 0
    for t in range(max_ticks):
        agi.step()
        wm = agi.cognition.world_model
        # 收集新出现的惊喜度（error_buffer 每 tick 追加）
        eb = agi.cognition.error_buffer
        if len(eb) > last_buf_len:
            surprise_series.extend(eb[last_buf_len:])
            last_buf_len = len(eb)
        if env.food_eaten > 0 and ate_at is None:
            ate_at = t
        if ate_at is not None and t > ate_at + 150:
            break
    n_branches = len(agi.cognition.world_model.branches) if hasattr(agi.cognition.world_model, 'branches') else 1
    # 认知区分度：学习前(前1/3) vs 学习后(最后1/3) 的惊喜度差
    s = np.array(surprise_series) if surprise_series else np.zeros(10)
    third = max(1, len(s) // 3)
    early_s = np.mean(s[:third])
    late_s = np.mean(s[-third:])
    # 熵序列：从 collapse_history 取（带 tick 时序）
    if hasattr(wm, 'collapse_history') and wm.collapse_history:
        ents = [e for _, _, e in wm.collapse_history]
        ent = np.array(ents)
        t3 = max(1, len(ent) // 3)
        early_e = np.mean(ent[:t3])
        late_e = np.mean(ent[-t3:])
    else:
        early_e = late_e = 0.0
    return {
        "ate_at": ate_at,
        "surprise_drop": early_s - late_s,
        "entropy_drop": early_e - late_e,
        "n_branches": n_branches,
        "alive": agi.alive,
    }


def main():
    trials = 3
    print("=" * 60, flush=True)
    print("A/B: 确定性世界模型 vs 薛定谔叠加态世界模型", flush=True)
    print("=" * 60, flush=True)

    results = {"det": [], "sup": []}
    for seed in range(trials):
        r_det = run_once(False, seed=seed)
        r_sup = run_once(True, seed=seed)
        results["det"].append(r_det)
        results["sup"].append(r_sup)
        print(f"seed={seed} | 确定: ate={r_det['ate_at']} 分支={r_det['n_branches']} "
              f"惊喜降={r_det['surprise_drop']:.3f} | "
              f"叠加: ate={r_sup['ate_at']} 分支={r_sup['n_branches']} "
              f"惊喜降={r_sup['surprise_drop']:.3f} 熵降={r_sup['entropy_drop']:.3f}", flush=True)

    print("\n── 汇总 ──", flush=True)
    det_ate = [r["ate_at"] for r in results["det"] if r["ate_at"] is not None]
    sup_ate = [r["ate_at"] for r in results["sup"] if r["ate_at"] is not None]
    det_succ = len(det_ate) / trials
    sup_succ = len(sup_ate) / trials
    print(f"成功率:   确定性 {det_succ*100:.0f}%  vs  叠加态 {sup_succ*100:.0f}%", flush=True)
    print(f"学习速度: 确定性 avg={np.mean(det_ate) if det_ate else 'N/A'}"
          f"  vs  叠加态 avg={np.mean(sup_ate) if sup_ate else 'N/A'}", flush=True)
    print(f"分支数:   确定性 1（固定）  vs  叠加态 "
          f"{[r['n_branches'] for r in results['sup']]}（可增长）", flush=True)
    det_drop = np.mean([r["surprise_drop"] for r in results["det"]])
    sup_drop = np.mean([r["surprise_drop"] for r in results["sup"]])
    print(f"认知区分度(惊喜下降): 确定性 {det_drop:.3f}  vs  叠加态 {sup_drop:.3f}", flush=True)
    sup_ed = np.mean([r["entropy_drop"] for r in results["sup"]])
    print(f"叠加态独有: 坍缩熵下降 {sup_ed:.3f}（确定性无此信号）", flush=True)


if __name__ == "__main__":
    main()
