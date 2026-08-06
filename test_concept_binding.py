"""
概念预测绑定验证（DESIGN_CONCEPTS §3 阶段 2：概念预测自己的价值）

假设：概念 = 观测簇 × 价值绑定——概念的价值预测（出现时 V 的 EMA）
应显著优于"随机/恒定"基线（概念学到了"这个可消耗物值多少"）。

设计：
- 场景 A：食物价值高（每次吃 +0.2 energy → V 高）——概念 value_ema 应高
- 场景 B：危险/低价值（假设"毒果"——吃了 V 降）——概念 value_ema 应低
- 判据：
  A. 高价值概念预测 > 0.55（学到"值钱"）
  B. 低价值概念预测 < 0.45（学到"不值钱"）——概念区分价值
  C. 预测 vs 实际 V 相关性（预测有价值——r > 0.3）
"""
import sys
import numpy as np

from cognition.concept_bank import ConceptBank


def run_scenario(obs_list, v_list):
    """喂 (obs, v) 序列给价值锚聚类，返回概念库"""
    cb = ConceptBank()
    for obs, v in zip(obs_list, v_list):
        cb.add_value_anchored(np.asarray(obs, dtype=np.float32), True, v=v)
    return cb


def main():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    print("=" * 56)
    print("概念预测绑定验证（概念→价值预测）")
    print("=" * 56)

    # 场景 A：高价值食物（obs 特征 A 类，V 高 0.7-0.9）
    rng = np.random.RandomState(0)
    obs_a = [rng.rand(4) * 0.3 for _ in range(30)]       # 相似观测（同簇）
    v_a = [0.7 + rng.rand() * 0.2 for _ in range(30)]    # V 高
    # 场景 B：低价值（obs 特征 B 类，V 低 0.2-0.4）
    obs_b = [0.8 + rng.rand(4) * 0.3 for _ in range(30)]
    v_b = [0.2 + rng.rand() * 0.2 for _ in range(30)]

    cb = ConceptBank()
    for obs, v in zip(obs_a, v_a):
        cb.add_value_anchored(np.asarray(obs, dtype=np.float32), True, v=v)
    for obs, v in zip(obs_b, v_b):
        cb.add_value_anchored(np.asarray(obs, dtype=np.float32), True, v=v)

    # 测试匹配：A 类观测 → 高价值概念；B 类 → 低价值概念
    _, _, ma, pa = cb.match_concept(np.asarray(obs_a[0]), threshold=1.5)
    _, _, mb, pb = cb.match_concept(np.asarray(obs_b[0]), threshold=1.5)

    print(f"  A 类观测 → 匹配={ma} 预测V={pa:.3f}")
    ok_a = ma and pa > 0.55
    print(f"  A: {'OK（高价值概念预测高——学到「值钱」）' if ok_a else 'FAIL'}")

    print(f"  B 类观测 → 匹配={mb} 预测V={pb:.3f}")
    ok_b = mb and pb < 0.45
    print(f"  B: {'OK（低价值概念预测低——概念区分价值）' if ok_b else 'FAIL'}")

    # C：跨概念价值区分（概念预测是概念级 EMA——同概念内恒定，
    # 组内相关无意义；跨概念：高价值概念预测显著高于低价值，
    # 且各自样本量充足（value_count ≥ 10——EMA 已收敛））
    # review warn：StopIteration 防护（next 找不到时判 FAIL 而非崩）
    try:
        ca = next(c for c in cb.concepts if c.value_count > 0 and
                  abs(c.vector[0] - obs_a[0][0]) < 0.5)
        cb2 = next(c for c in cb.concepts if c is not ca)
        print(f"  C: 高价值概念 {ca.value_ema:.3f} (n={ca.value_count}) "
              f"vs 低价值概念 {cb2.value_ema:.3f} (n={cb2.value_count})")
        ok_c = (ca.value_ema - cb2.value_ema > 0.2
                and ca.value_count >= 10 and cb2.value_count >= 10)
        print(f"     {'OK（跨概念价值区分稳定——EMA 已收敛）' if ok_c else 'FAIL'}")
    except StopIteration:
        ok_c = False
        print("  C: FAIL（概念不足 2 个——无法跨概念比较）")

    ok = ok_a and ok_b and ok_c
    print(f"\n  判定: {'OK 通过（概念预测绑定成立）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
