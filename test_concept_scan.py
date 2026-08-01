"""
路线 B 收尾 — D 外观权重收敛参数扫描（离线调参，非正式实验）

目标：找 L2 × 训练轮数组合，使 D（|w_app| < 0.3）收敛且 B（果实零样本
最差 seed ≥ 40/50）保持。

预注册约束（不变）：
  - 训练数据分布不变：food+water 两种表面（绝不引入 fruit）
  - 判定标准不变：B ≥ 40/50、D < 0.3
  - 只调超参：L2 ∈ {0.01, 0.03, 0.05}、轮数 ∈ {500, 1000}

输出：组合表（B 最差 seed / D 外观权重），选定默认参数写入
test_concept_transfer.py。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_concept_transfer import train_mixed, eval_policy, LinearQ


def scan():
    np.random.seed(42)
    print("=" * 66)
    print("路线 B 收尾 — 参数扫描（L2 × 轮数 → D 收敛且 B 保持）")
    print("=" * 66)
    print(f"{'L2':>6} {'轮数':>5} | {'B最差seed':>9} | {'D|w_app|':>8} | 判定")
    print("-" * 66)
    best = None
    for l2 in (0.01, 0.03, 0.05):
        for ep in (500, 1000):
            agent = train_mixed(episodes_per=ep // 2, seed=0, l2=l2)
            # B：3 独立评估布局集（seed 42/7/2026）
            b_vals = []
            for s in (42, 7, 2026):
                np.random.seed(s)
                b_vals.append(eval_policy(agent, "fruit", trials=50, seed=500 + s))
            b_min = min(b_vals)
            w_app = float(np.abs(agent.theta[:, 10]).max())
            d_ok = w_app < 0.3
            b_ok = b_min >= 40
            verdict = "★选" if (d_ok and b_ok) else ""
            print(f"{l2:>6.2f} {ep:>5} | {b_min:>7}/50 | {w_app:>8.3f} | "
                  f"{'B' if b_ok else 'b'}{'D' if d_ok else 'd'}{verdict}")
            if d_ok and b_ok and (best is None or w_app < best[2]):
                best = (l2, ep, w_app, b_min)
    print("-" * 66)
    if best:
        print(f"选定: L2={best[0]}, 轮数={best[1]}（|w_app|={best[2]:.3f}, "
              f"B 最差 {best[3]}/50）")
        return 0
    print("未找到 D 收敛且 B 保持的组合——需调整扫描范围")
    return 1


if __name__ == "__main__":
    sys.exit(scan())
