"""
负价值锚验证（⑧ 对称压缩——"什么会死"与"什么能活"同等重要）

假设：伤害/V 下降观测 → danger 概念簇（与 consumable 对称）。
概念驱动停留前检查 danger——威胁区不逗留。

设计：
- 喂伤害观测（特征 A 类 + v_down）→ danger 簇
- 喂正价值观测（特征 B 类 + v_up）→ consumable 簇
- 判据：
  A. 伤害观测 → danger 概念形成
  B. 正价值观测 → consumable 概念（对称——两类共存）
  C. 区分：伤害类观测匹配 danger 而非 consumable
  D. 对称性：两类概念计数一致（各 ≥1）
"""
import sys
import numpy as np

from cognition.concept_bank import ConceptBank


def main():
    print("=" * 56)
    print("负价值锚验证（danger 概念——⑧ 对称压缩）")
    print("=" * 56)
    rng = np.random.RandomState(0)
    cb = ConceptBank()

    # 伤害观测（特征 A 类——如威胁区方向）
    obs_d = [rng.rand(4) * 0.3 for _ in range(30)]
    # 正价值观测（特征 B 类——如食物方向）
    obs_f = [0.6 + rng.rand(4) * 0.3 for _ in range(30)]

    for obs in obs_d:
        cb.add_danger_anchored(np.asarray(obs, dtype=np.float32), True)
    for obs in obs_f:
        cb.add_value_anchored(np.asarray(obs, dtype=np.float32), True, v=0.8)

    kinds = [c.kind for c in cb.concepts]
    print(f"  概念: {[(c.kind, c.name) for c in cb.concepts]}")

    # A：danger 概念形成
    ok_a = "danger" in kinds
    print(f"  A: {'OK（伤害观测→danger 概念形成）' if ok_a else 'FAIL'}")

    # B：consumable 共存（对称）
    ok_b = "consumable" in kinds
    print(f"  B: {'OK（consumable 共存——正负对称）' if ok_b else 'FAIL'}")

    # C：区分（伤害类观测匹配 danger 而非 consumable——阈值 0.35
    # 与主循环一致；默认 1.5 过宽会双匹配）
    _, _, m_d, _ = cb.match_concept(np.asarray(obs_d[0]), kind="danger",
                                    threshold=0.35)
    _, _, m_f, _ = cb.match_concept(np.asarray(obs_d[0]), kind="consumable",
                                    threshold=0.35)
    print(f"  C: 伤害观测 → danger 匹配={m_d} consumable 匹配={m_f}")
    ok_c = m_d and not m_f
    print(f"     {'OK（伤害观测归 danger——不误入 consumable）' if ok_c else 'FAIL'}")

    # D：对称性（两类各 ≥1）
    nd = sum(1 for k in kinds if k == "danger")
    nf = sum(1 for k in kinds if k == "consumable")
    print(f"  D: danger={nd} consumable={nf}")
    ok_d = nd >= 1 and nf >= 1
    print(f"     {'OK（正负锚对称——两类共存）' if ok_d else 'FAIL'}")

    ok = ok_a and ok_b and ok_c and ok_d
    print(f"\n  判定: {'OK 通过（负价值锚成立——⑧ 对称压缩）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
