"""
friend-audit 修复③验证：override_action 死变量已接上应用点。

验证：
  A. 设置 override_action=3 → 下一 tick 实际动作 == 3（覆盖生效）
  B. 用后即清（-1）→ 后续 tick 恢复自主决策（动作 != 固定值）
  C. 睡眠时 override_action 不应用（动作保持 0）
"""
import sys
from main import AGI


def main():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    agi = AGI()
    # 让 env 可 step（AGI 默认无 env 也能跑——用无 env 模式更可控）
    results = {}

    # A：override 生效（step 返回状态 dict，取 ["action"]）
    agi.override_action = 3
    a1 = agi.step()
    act1 = a1["action"] if isinstance(a1, dict) else a1
    results["A_override_eff"] = (act1 == 3)
    print(f"  A: override_action=3 → 动作={act1} {'OK' if act1 == 3 else 'FAIL'}")

    # B：用后即清 → 恢复自主（连续多 tick 动作不应恒为 3）
    a2 = agi.step()
    a3 = agi.step()
    act2 = a2["action"] if isinstance(a2, dict) else a2
    act3 = a3["action"] if isinstance(a3, dict) else a3
    results["B_auto_restore"] = (act2 != 3 or act3 != 3)
    print(f"  B: 清除后 动作={act2},{act3} "
          f"{'OK（已恢复自主）' if results['B_auto_restore'] else 'FAIL'}")

    # C：睡眠时不应用
    agi.body.is_sleeping = True
    agi.override_action = 3
    a4 = agi.step()
    agi.body.is_sleeping = False
    act4 = a4["action"] if isinstance(a4, dict) else a4
    results["C_sleep_guard"] = (act4 == 0)
    print(f"  C: 睡眠+override=3 → 动作={act4} "
          f"{'OK（睡眠守卫生效）' if act4 == 0 else 'FAIL'}")

    # D：值域钳制（security warn 修复——非法覆盖动作被钳到 [0, n_actions-1]）
    agi.override_action = 99
    a5 = agi.step()
    act5 = a5["action"] if isinstance(a5, dict) else a5
    results["D_clamp"] = (0 <= act5 < agi.n_actions)
    print(f"  D: override=99 → 动作={act5} "
          f"{'OK（值域钳制）' if results['D_clamp'] else 'FAIL'}")

    # E：睡眠陈旧意图清除（security warn 修复——醒来后不残留）
    agi.body.is_sleeping = True
    agi.override_action = 2
    agi.step()
    agi.body.is_sleeping = False
    a6 = agi.step()
    act6 = a6["action"] if isinstance(a6, dict) else a6
    results["E_no_stale"] = (act6 != 2)
    print(f"  E: 睡眠中设 override=2 → 醒来后动作={act6} "
          f"{'OK（陈旧意图已清）' if results['E_no_stale'] else 'FAIL'}")

    ok = all(results.values())
    print(f"\n判定: {'OK（override 通路修复生效）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
