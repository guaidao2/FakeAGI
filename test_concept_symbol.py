"""
概念符号化验证（DESIGN_CONCEPTS §3 阶段 3：词↔概念绑定）

假设：概念激活时同时出现的词 → 共现绑定（Hebbian）——
词从此有"所指"（绑定过的概念），概念从此有"名字"。

设计：
- 场景 A：喂 (obs, v) 序列 + 伴随词 ["food"] → 概念绑定 "food"
- 场景 B：新词 ["water"]（未共现）→ 不绑定（区分学习）
- 判据：
  A. 绑定学习：概念符号含 "food"（共现绑定生效）
  B. 词→概念：activate_by_symbol("food") 返回概念 + 预测值
  C. 语言接地：未绑定词 "water" 不激活（区分）
  D. 词→行为：符号激活返回的 value_pred 与概念一致（可驱动引导）
"""
import sys
import numpy as np

from cognition.concept_bank import ConceptBank


def main():
    print("=" * 56)
    print("概念符号化验证（词-概念绑定——语言接地）")
    print("=" * 56)

    rng = np.random.RandomState(0)
    cb = ConceptBank()
    obs_a = [rng.rand(4) * 0.3 for _ in range(30)]
    v_a = [0.7 + rng.rand() * 0.2 for _ in range(30)]

    # 共现学习：概念激活（add_value_anchored 命中）+ 伴随词 ["food"]
    last_name = ""
    for obs, v in zip(obs_a, v_a):
        name = cb.add_value_anchored(np.asarray(obs, dtype=np.float32),
                                     True, v=v)
        if name:
            last_name = name
            cb.bind_symbols(name, ["food"])   # 概念激活时听到 "food"

    print(f"  概念: {[c.name for c in cb.concepts]}")

    # A：绑定学习（review nit：用绑定目标 last_name 而非 concepts[0]）
    c = next(c for c in cb.concepts if c.name == last_name)
    print(f"  A: 概念 {c.name} 符号 = {c.symbols}")
    ok_a = "food" in c.symbols
    print(f"     {'OK（共现绑定生效——概念学到词）' if ok_a else 'FAIL'}")

    # B：词→概念
    cname, vpred, found = cb.activate_by_symbol("food")
    print(f"  B: activate_by_symbol('food') → {cname} v={vpred:.3f}")
    ok_b = found and cname == c.name and vpred > 0.55
    print(f"     {'OK（词激活概念——有「所指」）' if ok_b else 'FAIL'}")

    # C：语言接地区分（未绑定词不激活）
    _, _, found_w = cb.activate_by_symbol("water")
    print(f"  C: activate_by_symbol('water') → found={found_w}")
    ok_c = not found_w
    print(f"     {'OK（未共现词不激活——区分学习）' if ok_c else 'FAIL'}")

    # D：词→行为（符号激活 value_pred 与概念一致——可驱动引导）
    _, vpred2, _ = cb.activate_by_symbol("food")
    ok_d = abs(vpred2 - c.predict_value()) < 1e-6
    print(f"  D: 符号激活 v={vpred2:.3f} vs 概念 v={c.predict_value():.3f}")
    print(f"     {'OK（符号→价值一致——词可驱动引导）' if ok_d else 'FAIL'}")

    ok = ok_a and ok_b and ok_c and ok_d
    print(f"\n  判定: {'OK 通过（概念符号化成立——语言接地）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
