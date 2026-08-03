"""
概念组合验证（① 阶段 4——概念图：可消耗物抽象）

概念图 = 多个子概念共享抽象（食物簇 ∪ 水源簇 → "可消耗物"）。
抽象匹配：任一子概念类型匹配 → 抽象激活（跨子概念泛化）。

设计：
- 喂食物簇（obs 特征 A 类，V 高）+ 水源簇（obs 特征 B 类，V 高）
- 注册抽象 "consumable" → ["consumable"]（默认——两类都 kind=consumable
  则同簇；要真组合需不同 kind——用自定义 kind 模拟）
- 判据：
  A. 食物类观测 → "可消耗物"抽象激活
  B. 水源类观测 → 同一抽象激活（跨子概念泛化——组合的本质）
  C. 抽象返回价值预测（抽象带价值——不是空壳）
"""
import sys
import numpy as np

from cognition.concept_bank import ConceptBank


def main():
    print("=" * 56)
    print("概念组合验证（概念图——可消耗物抽象）")
    print("=" * 56)
    rng = np.random.RandomState(0)

    # 自定义 kind：food_source / water_source（同属 consumable 抽象）
    cb = ConceptBank()
    obs_f = [rng.rand(4) * 0.3 for _ in range(20)]        # 食物类观测
    obs_w = [0.6 + rng.rand(4) * 0.3 for _ in range(20)]  # 水源类观测（不同特征区）
    v_f = [0.7 + rng.rand() * 0.2 for _ in range(20)]
    v_w = [0.6 + rng.rand() * 0.2 for _ in range(20)]

    # 直接构造不同 kind 的概念（模拟食物/水源两个子概念）
    from cognition.concept_bank import Concept
    cb.concepts.append(Concept("food_c", "food_source", obs_f[0].copy()))
    cb.concepts.append(Concept("water_c", "water_source", obs_w[0].copy()))
    for obs, v in zip(obs_f, v_f):
        cb.concepts[0].update_value(v)
    for obs, v in zip(obs_w, v_w):
        cb.concepts[1].update_value(v)
    # 注册抽象：可消耗物 = 食物源 ∪ 水源
    cb.add_abstract_group("consumable", ["food_source", "water_source"])

    # A：食物类观测 → 抽象激活
    m_f, vp_f = cb.match_abstract(np.asarray(obs_f[0]), "consumable")
    print(f"  A: 食物观测 → 抽象匹配={m_f} v={vp_f:.3f}")
    ok_a = m_f and vp_f > 0.55
    print(f"     {'OK（食物类激活可消耗物抽象）' if ok_a else 'FAIL'}")

    # B：水源类观测 → 同一抽象激活（跨子概念泛化——组合的本质）
    m_w, vp_w = cb.match_abstract(np.asarray(obs_w[0]), "consumable")
    print(f"  B: 水源观测 → 抽象匹配={m_w} v={vp_w:.3f}")
    ok_b = m_w and vp_w > 0.55
    print(f"     {'OK（水源类激活同一抽象——跨子概念泛化）' if ok_b else 'FAIL'}")

    # C：抽象带价值（非空壳——价值预测来自子概念）
    ok_c = vp_f > 0 and vp_w > 0
    print(f"  C: 抽象价值预测 f={vp_f:.3f} w={vp_w:.3f}")
    print(f"     {'OK（抽象带价值——可驱动决策）' if ok_c else 'FAIL'}")

    # D：不相关观测不激活（区分——危险类不激活可消耗物）
    obs_d = [2.0 + rng.rand(4) * 0.5 for _ in range(5)]
    m_d, _ = cb.match_abstract(np.asarray(obs_d[0]), "consumable")
    print(f"  D: 远特征观测 → 抽象匹配={m_d}")
    ok_d = not m_d
    print(f"     {'OK（不相关观测不激活——抽象有边界）' if ok_d else 'FAIL'}")

    ok = ok_a and ok_b and ok_c and ok_d
    print(f"\n  判定: {'OK 通过（概念组合成立——跨子概念抽象）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
