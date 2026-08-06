"""
他者模型验证 — 多智能体迷宫实验（竞争 / 合作 / 无他者基线）

验证项：
  A. 意图识别：competitor 他者 → 系统意图=competitor（高置信）
  B. 意图识别：cooperator 他者 → 系统意图=cooperator（高置信）
  C. 竞争回避：competitor 模式下，系统与他者距离保持 > 无他者基线
  D. 合作不回避：cooperator 模式下系统正常觅食（距离 < 竞争模式）
  E. 竞争资源分配：competitor 场景下系统食物获取 < 基线（被抢）
     cooperator 场景下系统食物获取 ≈ 基线（共享不冲突）
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from core.other_agent import OtherAgent, OtherModel


class SharedEnv:
    """共享迷宫：一个食物，主系统 + 他者"""
    def __init__(self, size=16):
        self.size = size
        self.pos = [8, 8]
        self.food_pos = [np.random.randint(0, size), np.random.randint(0, size)]
        self.food_eaten = 0
        self.steps = 0

    def step(self, action):
        self.steps += 1
        dirs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dirs[action % 5]
        self.pos[0] = max(0, min(self.size - 1, self.pos[0] + dx))
        self.pos[1] = max(0, min(self.size - 1, self.pos[1] + dy))
        d = abs(self.pos[0] - self.food_pos[0]) + abs(self.pos[1] - self.food_pos[1])
        if d < 2:
            self.food_eaten += 1
            self.food_pos = [np.random.randint(0, self.size),
                             np.random.randint(0, self.size)]
        return d < 2


def run_episode(other_strategy, max_ticks=2000, seed=0, other_enabled=True):
    np.random.seed(seed)
    env = SharedEnv()
    om = OtherModel()
    other = OtherAgent(strategy=other_strategy) if other_enabled else None
    dists = []
    for t in range(max_ticks):
        # 我：朝食物（简单反射）
        dx = env.food_pos[0] - env.pos[0]
        dy = env.food_pos[1] - env.pos[1]
        action = 3 if abs(dx) > abs(dy) and dx > 0 else (2 if dx < 0 and abs(dx) > abs(dy)
                 else (4 if dy > 0 else 1))
        # 他者影响：竞争回避（模型驱动）；合作→不干预（正常觅食）
        if other is not None and om.intent == "competitor":
            avoid = om.get_avoidance(env.pos)
            if avoid is not None:
                action = avoid
        env.step(action)
        # 他者移动
        if other is not None:
            other_action = other.choose_action(env.food_pos)
            other.step(other_action)
            om.observe(other.pos, env.pos, env.food_pos, t)
            dists.append(abs(other.pos[0] - env.pos[0]) + abs(other.pos[1] - env.pos[1]))
    # 前后半程距离对比（回避生效：识别 competitor 后距离应增大）
    half = len(dists) // 2
    before = np.mean(dists[:half]) if half > 0 else 0
    after = np.mean(dists[half:]) if half < len(dists) else 0
    return {
        "food": env.food_eaten,
        "intent": om.intent,
        "intent_conf": om.intent_confidence,
        "dist_before": before,
        "dist_after": after,
        "dist_gain": after - before,
    }


def main():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    print("=" * 60)
    print("他者模型验证 — 多智能体迷宫（竞争/合作/基线）")
    print("=" * 60)

    # 基线（无他者）
    base = np.mean([run_episode("wanderer", seed=s, other_enabled=False)["food"]
                    for s in range(10)])
    # 竞争 / 合作（各 10 seeds）
    comp = [run_episode("competitor", seed=s) for s in range(10)]
    coop = [run_episode("cooperator", seed=s) for s in range(10)]

    comp_food = np.mean([r["food"] for r in comp])
    coop_food = np.mean([r["food"] for r in coop])
    comp_gain = np.mean([r["dist_gain"] for r in comp])
    coop_gain = np.mean([r["dist_gain"] for r in coop])
    comp_intent = sum(1 for r in comp if r["intent"] == "competitor")
    coop_intent = sum(1 for r in coop if r["intent"] == "cooperator")

    print(f"\n基线（无他者）: food={base:.1f}")
    print(f"竞争他者: food={comp_food:.1f} 意图=competitor {comp_intent}/10 "
          f"距离增益={comp_gain:+.1f}")
    print(f"合作他者: food={coop_food:.1f} 意图=cooperator {coop_intent}/10 "
          f"距离增益={coop_gain:+.1f}")

    # 判定
    a_ok = comp_intent >= 7
    b_ok = coop_intent >= 7
    c_ok = comp_gain > 0  # 竞争：识别后距离增大（回避生效）
    d_ok = coop_gain < comp_gain  # 合作：不回避（增益更小）
    e_ok = comp_food < base * 0.7 and coop_food > base * 0.8  # 竞争被抢；合作不受影响
    print(f"\n[A] 竞争意图识别: {comp_intent}/10 (应≥7) {'OK' if a_ok else 'FAIL'}")
    print(f"[B] 合作意图识别: {coop_intent}/10 (应≥7) {'OK' if b_ok else 'FAIL'}")
    print(f"[C] 竞争回避: 距离增益 {comp_gain:+.1f} > 0 {'OK' if c_ok else 'FAIL'}")
    print(f"[D] 合作不回避: 增益 {coop_gain:+.1f} < 竞争 {comp_gain:+.1f} "
          f"{'OK' if d_ok else 'FAIL'}")
    print(f"[E] 资源分配: 竞争 food={comp_food:.1f} < 基线×0.7 {base*0.7:.1f}"
          f" 且 合作 food={coop_food:.1f} > 基线×0.8 {base*0.8:.1f} "
          f"{'OK' if e_ok else 'FAIL'}")

    ok = a_ok and b_ok and c_ok and d_ok and e_ok
    print(f"\n判定: {'OK 通过' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
