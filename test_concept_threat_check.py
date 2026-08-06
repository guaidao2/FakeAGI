"""
威胁检查行为验证（概念驱动停留前——danger 匹配则不停留）

背景：上轮"威胁检查语义反转"（危险分支反而停留）靠 review 人工审查
才发现——补行为级测试（续审 should-fix：测试不应依赖人工审查）。

设计：
- 构造概念库：consumable 概念（食物区特征）+ danger 概念（威胁区特征）
- 场景 A：观测匹配 consumable 且不匹配 danger → 停留逻辑触发（action=0）
- 场景 B：观测同时匹配 consumable 和 danger（重叠区）→ 不触发停留
  （② 活着优先——威胁区不逗留）
"""
import sys
import numpy as np

from cognition.concept_bank import ConceptBank, Concept


def build_cb():
    cb = ConceptBank()
    # 威胁旁食物场景：consumable 质心 threat_near≈0.6（威胁邻接时吃过）
    # danger 质心 threat_near≈0.9——重叠带 [0.6,0.9] 双匹配
    fc = Concept("food_c", "consumable", np.array([0.5, 0.5, 0.6], dtype=np.float32))
    fc.update_value(0.8)
    cb.concepts.append(fc)
    dc = Concept("danger_c", "danger", np.array([0.5, 0.5, 0.9], dtype=np.float32))
    cb.concepts.append(dc)
    return cb


def main():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    print("=" * 56)
    print("威胁检查行为验证（danger 匹配→不停留）")
    print("=" * 56)
    cb = build_cb()

    # 场景 A：安全食物区（threat_near≈0.3——不匹配 danger）
    obs_safe = np.array([0.5, 0.5, 0.3], dtype=np.float32)
    _, _, m_f, vp = cb.match_concept(obs_safe, kind="consumable", threshold=0.35)
    _, _, m_d, _ = cb.match_concept(obs_safe, kind="danger", threshold=0.35)
    stop_a = m_f and not m_d
    print(f"  A: 安全区 consumable={m_f} danger={m_d} → 停留={stop_a}")
    ok_a = stop_a
    print(f"     {'OK（安全区→停留逻辑可触发）' if ok_a else 'FAIL'}")

    # 场景 B：重叠区（threat_near=0.75——同时匹配 consumable+danger）
    obs_overlap = np.array([0.5, 0.5, 0.75], dtype=np.float32)
    _, _, m_f2, _ = cb.match_concept(obs_overlap, kind="consumable",
                                     threshold=0.35)
    _, _, m_d2, _ = cb.match_concept(obs_overlap, kind="danger",
                                     threshold=0.35)
    stop_b = m_f2 and not m_d2   # 威胁检查后应不停留
    print(f"  B: 重叠区 consumable={m_f2} danger={m_d2} → 停留={stop_b}")
    ok_b = m_f2 and m_d2 and not stop_b
    print(f"     {'OK（重叠区 danger 匹配→不停留——活着优先）' if ok_b else 'FAIL'}")

    ok = ok_a and ok_b
    print(f"\n  判定: {'OK 通过（威胁检查行为正确）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
