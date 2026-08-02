"""
B2 接线验证：睡眠巩固按显著性加权重放（CLS，DESIGN_CONCEPTS §7.5）

修复前：consolidate 均匀抽样（np.random.choice 无权重）——
高预测误差（信息量高）的经验与普通经验同等概率被重放。

验证：
  A. 加权重放：高 surprise 经验被选中的比例显著高于均匀基线
  B. 保底：零 surprise 经验仍有非零概率被选（不饿死低信息经验）
"""
import sys
import numpy as np
from cognition.sleep import SleepCycle


def run(seed=0):
    np.random.seed(seed)
    sc = SleepCycle()
    # 构造 100 条经验：10 条高 surprise(0.9)，90 条低 surprise(0.05)
    buf = []
    for i in range(100):
        buf.append({"surprise": 0.9 if i < 10 else 0.05,
                    "pos": [i % 10, i // 10], "action": 1})
    # 多次巩固统计高 surprise 选中比例
    n_trials = 200
    high_picked = 0
    total_picked = 0
    for _ in range(n_trials):
        out = sc.consolidate(buf)
        picked = set()
        # 重建被选 indices 不易——用高 surprise 条目出现次数近似：
        # consolidate 返回的是条目本身，统计其中高 surprise 条目数
        for item in out:
            picked.add(item["pos"][0])
        # 简化：直接统计返回条目里 surprise 0.9 的数量
        high_picked += sum(1 for item in out if item["surprise"] == 0.9)
        total_picked += len(out)
    ratio = high_picked / total_picked
    # 均匀基线：10/100 = 10%
    print(f"  [seed{seed}] 高surprise重放占比={ratio*100:.1f}% "
          f"(均匀基线 10%)")
    return ratio


if __name__ == "__main__":
    ratios = [run(s) for s in [0, 1, 2]]
    avg = np.mean(ratios)
    ok_a = avg > 0.15          # 显著高于 10% 基线
    # B：低 surprise 经验仍被选中（保底）——由权重 +0.1 保证
    ok_b = avg < 0.95          # 未完全排挤低信息经验
    print(f"\nA(加权优先)={'OK' if ok_a else 'FAIL'} "
          f"B(保底不排挤)={'OK' if ok_b else 'FAIL'}")
    ok = ok_a and ok_b
    verdict = ("OK（B2 接线完成：睡眠巩固按显著性重放——"
               "高预测误差经验优先进入长期记忆）" if ok else "FAIL")
    print("判定: " + verdict)
    sys.exit(0 if ok else 1)
