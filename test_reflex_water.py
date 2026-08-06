"""reflex_vote 水方向导航单测（欲望架构阶段 A——review should-fix 2）

覆盖：
  A. 渴且不饿 → 朝水方向（obs[2:4]）
  B. 水<0.3 濒死 → 水优先于食物
  C. 又渴又饿且水>=0.3 → 朝食物方向
  D. 水方向不可见（0,0）→ 回退食物方向（不失明）
  E. 双驱动叠加：thirst+hunger bias[:4] == 0.6
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from cognition.decision.committee import DecisionCommittee
from core.drives import DriveSystem


def make_body(energy=1.0, water=1.0):
    return {"energy": energy, "water": water}


def check(name, ok):
    print(f"  {name}: {'OK' if ok else 'FAIL'}")
    return ok


def main():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    ok = True
    com = DecisionCommittee(n_actions=5)
    # A. 渴且不饿 → 朝水（水在右侧 dx=0.8）
    obs = np.array([0.0, 0.0, 0.8, 0.0])  # 食物不可见, 水在右
    v = com.reflex_vote(obs, np.zeros(6), make_body(energy=1.0, water=0.4))
    ok &= check("A 渴不饿→水方向(右=3)", int(v.argmax()) == 3 and v[3] == 1.0)
    # B. 水<0.3 濒死 → 水优先于食物（食物在左 dx=-0.8, 水在右）
    obs = np.array([-0.8, 0.0, 0.8, 0.0])
    v = com.reflex_vote(obs, np.zeros(6), make_body(energy=0.5, water=0.2))
    ok &= check("B 濒死→水优先(右=3)", int(v.argmax()) == 3)
    # C. 又渴又饿且水>=0.3 → 朝食物（食物在左）
    obs = np.array([-0.8, 0.0, 0.8, 0.0])
    v = com.reflex_vote(obs, np.zeros(6), make_body(energy=0.5, water=0.4))
    ok &= check("C 渴+饿(水>=0.3)→食物(左=2)", int(v.argmax()) == 2)
    # D. 水方向不可见 → 回退食物方向
    obs = np.array([-0.8, 0.0, 0.0, 0.0])
    v = com.reflex_vote(obs, np.zeros(6), make_body(energy=1.0, water=0.2))
    ok &= check("D 水不可见→回退食物(左=2)", int(v.argmax()) == 2)
    # E. 双驱动叠加
    dr = DriveSystem()
    dr.hunger = 0.6
    dr.thirst = 0.6
    b = dr.get_action_bias()
    ok &= check(f"E thirst+hunger bias[:4]==0.6 (got {b[0]:.1f})",
                abs(b[0] - 0.6) < 1e-6 and b[4] == 0.0)
    # F. 单口渴（不饿）
    dr2 = DriveSystem()
    dr2.hunger = 0.1
    dr2.thirst = 0.6
    b2 = dr2.get_action_bias()
    ok &= check(f"F 仅口渴 bias[:4]==0.3 (got {b2[0]:.1f})", abs(b2[0] - 0.3) < 1e-6)
    print("判定:", "ALL OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
