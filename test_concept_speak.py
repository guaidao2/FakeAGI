"""
语言生成验证（①——概念→词：看到概念→说出绑定符号）

假设：概念绑定词后，概念激活能"说对词"（生成方向与理解反向）。

设计：
- 喂 (obs, v) + 绑定 "food" → speak(食物类 obs) → "food"
- 判据：
  A. 生成准确：speak(食物类) → "food"（说对）
  B. 沉默正确：speak(陌生观测) → 不开口（无匹配不说）
  C. 双向闭环：理解（听 food→概念）与生成（看概念→说 food）一致
"""
import sys
import numpy as np

from cognition.concept_bank import ConceptBank


def main():
    print("=" * 56)
    print("语言生成验证（概念→词——说）")
    print("=" * 56)
    rng = np.random.RandomState(0)
    cb = ConceptBank()

    obs_f = [rng.rand(4) * 0.3 for _ in range(30)]
    v_f = [0.7 + rng.rand() * 0.2 for _ in range(30)]
    last_name = ""
    for obs, v in zip(obs_f, v_f):
        name = cb.add_value_anchored(np.asarray(obs, dtype=np.float32),
                                     True, v=v)
        if name:
            last_name = name
            cb.bind_symbols(name, ["food"])

    # A：生成准确
    w, cname, spoke = cb.speak(np.asarray(obs_f[0]))
    print(f"  A: speak(食物类观测) → '{w}' ({cname})")
    ok_a = spoke and w == "food"
    print(f"     {'OK（概念激活→说出绑定词——生成准确）' if ok_a else 'FAIL'}")

    # B：沉默正确（陌生观测——远离概念区）
    obs_strange = [2.0 + rng.rand(4) * 0.5 for _ in range(5)]
    w2, _, spoke2 = cb.speak(np.asarray(obs_strange[0]))
    print(f"  B: speak(陌生观测) → spoke={spoke2}")
    ok_b = not spoke2
    print(f"     {'OK（无匹配不说——沉默正确）' if ok_b else 'FAIL'}")

    # C：双向闭环（理解与生成一致）
    cname2, vp, found = cb.activate_by_symbol("food")
    w3, cname3, spoke3 = cb.speak(np.asarray(obs_f[0]))
    ok_c = found and spoke3 and cname2 == cname3
    print(f"  C: 听 food→{cname2}；看概念→说 '{w3}'（{cname3}）")
    print(f"     {'OK（理解与生成闭环——同一概念）' if ok_c else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    print(f"\n  判定: {'OK 通过（语言生成成立——概念会说词）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
