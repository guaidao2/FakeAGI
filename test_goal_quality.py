"""
元认知目标质量验证（空间记忆引导 vs 随机基线）

修复前：goal_gen.py 检查 get_exploration_target 但从未实现 →
空间记忆引导从未生效，恒回退随机方向（randint）。

验证：
  A. 目标类型分布：多数目标为 explore_unvisited（引导）而非
     random_explore（随机）——修复前 100% 随机
  B. 目标位置质量：引导目标平均 visit_count 显著低于随机基线
     （信息增益：去没去过的地方）
  C. 未访问目标存在：地图部分探索时目标含未访问位置
"""
import sys
import numpy as np
from cognition.spatial_memory import SpatialMemory
from cognition.metacognition.goal_gen import GoalGenerator, ExplorationGoal


def make_memory(visited_frac=0.4, seed=0):
    """构造 8x8 地图，访问 visited_frac 比例位置"""
    rng = np.random.RandomState(seed)
    mem = SpatialMemory()
    for x in range(8):
        for y in range(8):
            if rng.random() < visited_frac:
                mem.update_position(
                    (x, y), energy_delta=-0.001,
                    surprise=rng.random() * 0.5)
    return mem


def run(seed=0):

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    np.random.seed(seed)  # nit 修复：get_exploration_target 用全局 rng——固定可复现
    rng = np.random.RandomState(seed)
    mem = make_memory(visited_frac=0.4, seed=seed)
    gen = GoalGenerator(spatial_memory=mem)
    agent_pos = [3, 3]
    kinds = []
    targets_visit = []
    unvisited_targets = 0
    n = 100
    for _ in range(n):
        # world_model 缺口（触发 explore_novel 分支）
        gap = type("G", (), {"kind": "world_model", "score": 0.8})()
        goal = gen.generate(gap, agent_pos=agent_pos, env_size=8)
        if goal is None:
            continue
        kinds.append(goal.context)
        if goal.target_pos is not None:
            tp = tuple(goal.target_pos)
            node = mem.nodes.get(tp)
            targets_visit.append(node.visit_count if node else 0)
            if node is None:
                unvisited_targets += 1

    # 随机基线：同地图随机方向目标
    base_visit = []
    for _ in range(n):
        dx = rng.choice([-1, 0, 1]) * 8 * 0.3
        dy = rng.choice([-1, 0, 1]) * 8 * 0.3
        tx = int(np.clip(agent_pos[0] + dx, 0, 7))
        ty = int(np.clip(agent_pos[1] + dy, 0, 7))
        node = mem.nodes.get((tx, ty))
        base_visit.append(node.visit_count if node else 0)

    guided_avg = np.mean(targets_visit) if targets_visit else 0
    base_avg = np.mean(base_visit)
    guide_ratio = (kinds.count("explore_unvisited") / max(len(kinds), 1))
    unvisited_ratio = unvisited_targets / max(len(targets_visit), 1)

    print(f"  [seed{seed}] 引导目标 {len(kinds)}/100 | "
          f"explore_unvisited={guide_ratio*100:.0f}% "
          f"random_explore={(1-guide_ratio)*100:.0f}%")
    print(f"          目标平均熟悉度={guided_avg:.2f} "
          f"vs 随机基线={base_avg:.2f} | "
          f"未访问目标={unvisited_ratio*100:.0f}%")

    ok_a = guide_ratio > 0.5          # 引导为主（修复前 0%）
    ok_b = guided_avg < base_avg * 0.8  # 目标熟悉度显著低于随机
    ok_c = unvisited_ratio > 0.1      # 有未访问目标
    return ok_a, ok_b, ok_c


if __name__ == "__main__":
    results = [run(s) for s in [0, 1, 2]]
    a = all(r[0] for r in results)
    b = all(r[1] for r in results)
    c = all(r[2] for r in results)
    print(f"\nA(引导为主)={'OK' if a else 'FAIL'} "
          f"B(目标熟悉度<随机)={'OK' if b else 'FAIL'} "
          f"C(含未访问)={'OK' if c else 'FAIL'}")
    ok = a and b and c
    verdict = ("OK（空间记忆引导真实生效：目标去未探索区域，"
               "信息增益高于随机基线）" if ok else "FAIL")
    print("判定: " + verdict)
    sys.exit(0 if ok else 1)
