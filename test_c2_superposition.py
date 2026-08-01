"""
C2 验证：叠加态可靠性估计器 vs 标量估计器
核心测试：被教歪可逆性——环境先连续失败（教歪），后恢复成功，
叠加态估计器应能回升（多假设不单点信任），标量估计器应卡死（对照）。

验证项：
  A. 连续失败 → 叠加态估计下降（坍缩到低假设）
  B. 恢复成功 → 叠加态估计回升（被教歪可逆——C2 核心）
  C. 恢复后权重分布保留低假设分支（"可能再变坏"记忆——标量单点跳变没有）
  D. 与 ProcessSelector 集成：叠加态模式下 N2/N4 行为正常
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from core.process_selector import SuperpositionEstimator, ProcessEstimator, ProcessSelector


def main():
    print("=" * 60)
    print("C2 验证：叠加态可靠性估计器（被教歪可逆性）")
    print("=" * 60)

    # A+B. 叠加态：连续失败 10 次 → 恢复成功 20 次
    sup = SuperpositionEstimator("ask", prior=0.5)
    for _ in range(10):
        sup.update(success=False)
    after_fail = sup.reliability
    for _ in range(20):
        sup.update(success=True)
    after_recover = sup.reliability

    a_ok = after_fail < 0.3
    b_ok = after_recover > after_fail + 0.2
    print(f"\n[A] 连续失败→叠加态下降: {0.5:.2f}→{after_fail:.2f} "
          f"(应<0.3) {'OK' if a_ok else 'FAIL'}")
    print(f"[B] 恢复成功→叠加态回升: {after_fail:.2f}→{after_recover:.2f} "
          f"(应+0.2) {'OK' if b_ok else 'FAIL'}")

    # C. 叠加态特有：恢复后权重分布仍保留低假设分支（"可能再变坏"的记忆）
    #    标量是单点跳变（0.0→1.0），叠加态是分布坍缩（多假设并存）
    #    置信度地板 = 最大权重×1% → 低假设相对权重 ≥ 1%（0.005 容差）
    weights = sup.weights
    low_branch = weights[2]  # 0.1 假设分支的权重
    c_ok = low_branch >= 0.005  # 低假设未完全归零——保留不确定性记忆
    print(f"[C] 叠加态恢复后权重分布: {[round(float(w),3) for w in weights]} "
          f"(低假设分支 {low_branch:.3f} 应≥0.005——保留'可能再变坏'记忆) "
          f"{'OK' if c_ok else 'FAIL'}")

    # D. ProcessSelector 集成：叠加态模式下 N2（fail）少问、N4（perfect）多问
    def sim(mode, trials=200):
        sel = ProcessSelector()  # 默认叠加态
        asks = 0
        for _ in range(trials):
            gap = 0.5
            choice = sel.choose(gap=gap, tick=0)
            if choice == "ask":
                asks += 1
                # 环境响应
                if mode == "fail":
                    sel.update_outcome("ask", success=False)
                else:
                    sel.update_outcome("ask", success=True)
            else:
                sel.update_outcome("sweep", success=np.random.random() < 0.3)
        return asks

    asks_fail = sim("fail")
    asks_perfect = sim("perfect")
    d_ok = asks_fail < asks_perfect * 0.3
    print(f"[D] 选择器集成: fail ask={asks_fail} vs perfect ask={asks_perfect} "
          f"(应 fail 少) {'OK' if d_ok else 'FAIL'}")

    ok = a_ok and b_ok and c_ok and d_ok
    print(f"\n判定: {'OK 通过' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
