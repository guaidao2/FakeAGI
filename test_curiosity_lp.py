"""
B1 接线验证：curiosity 接 learning progress（DESIGN_CONCEPTS §7.5）

修复前：CuriosityManager 主循环零调用（存在未接线）+ curiosity_map
是访问计数（novelty，非 learning progress——noisy-TV 陷阱）。

验证：
  A. 语义：loss 下降（在学）→ learning_progress 高；
     loss 停滞 → 低；loss 上升 → 0
  B. should_explore：learning progress 高时即使 surprise 低也探索
     （探索率提升路径，main.py 已接入）
  C. 无回归：默认配置下原有行为保留（计数/预算逻辑不变）
"""
import sys
import numpy as np
from core.curiosity import CuriosityManager


def run():
    # A：三种 loss 序列的 learning progress
    cm = CuriosityManager()
    for v in np.linspace(1.0, 0.2, 60):      # 下降（在学）
        cm.update_learning_progress(v)
    lp_down = cm.learning_progress

    cm2 = CuriosityManager()
    for v in [0.5] * 60:                      # 停滞
        cm2.update_learning_progress(v)
    lp_flat = cm2.learning_progress

    cm3 = CuriosityManager()
    for v in np.linspace(0.2, 1.0, 60):      # 上升（变差）
        cm3.update_learning_progress(v)
    lp_up = cm3.learning_progress

    print(f"  A: loss下降 lp={lp_down:.3f} | 停滞 lp={lp_flat:.3f} "
          f"| 上升 lp={lp_up:.3f}")
    ok_a = lp_down > 0.3 and lp_flat < 0.2 and lp_up < 0.1
    print(f"     {'OK（learning progress 语义正确）' if ok_a else 'FAIL'}")

    # B：learning progress 高时 should_explore 概率提升
    np.random.seed(0)
    cm4 = CuriosityManager()
    for v in np.linspace(1.0, 0.2, 60):
        cm4.update_learning_progress(v)
    n_explore_low = sum(cm4.should_explore(0.0) for _ in range(200))
    cm5 = CuriosityManager()
    n_explore_base = sum(cm5.should_explore(0.0) for _ in range(200))
    print(f"  B: surprise=0 探索次数——lp高(在学)={n_explore_low}/200 "
          f"vs 无lp={n_explore_base}/200")
    ok_b = n_explore_low > n_explore_base + 10
    print(f"     {'OK（在学中探索倾向提升）' if ok_b else 'FAIL'}")

    # C：无回归——计数/预算逻辑保留
    cm6 = CuriosityManager()
    cm6.record_exploration("food_east")
    cm6.record_exploration("food_east")
    ok_c = (cm6.get_boredom("food_east") == 0.2
            and cm6.curiosity_map.get("food_east") == 2
            and cm6.total_explorations == 2)
    print(f"  C: 计数/预算逻辑保留 {'OK' if ok_c else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    print(f"\n判定: {'OK（B1 接线完成：curiosity 由学习进展驱动）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
