"""
情绪系统验证 — 情感是物理的（恐惧/好奇/平静显式可测）

验证项：
  A. 快饿死（energy→0.05）→ fear 飙升（>0.5）
  B. 高应激（stress=1.0）→ fear 上升（> 低应激）
  C. 高 surprise（1.0）→ curiosity 上升（> 低 surprise）
  D. 稳态（energy/health 满、stress 低、surprise 低）→ calm 主导（>0.5）
  E. 恐惧调制：fear 高时探索率显著高于 calm 时（行为激进）
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from core.emotion import EmotionSystem


def main():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    print("=" * 60)
    print("情绪系统验证 — 情感是物理的（恐惧/好奇/平静）")
    print("=" * 60)

    emo = EmotionSystem()

    # A. 快饿死 → fear
    a = emo.update(energy=0.05, health=1.0, stress=0.0, surprise=0.0)
    a_ok = a["fear"] > 0.5
    print(f"\n[A] 快饿死→fear: {a['fear']:.2f} (应>0.5) {'OK' if a_ok else 'FAIL'}")

    # B. 高应激 vs 低应激
    hi = emo.update(energy=0.5, health=1.0, stress=1.0, surprise=0.0)
    lo = emo.update(energy=0.5, health=1.0, stress=0.0, surprise=0.0)
    b_ok = hi["fear"] > lo["fear"]
    print(f"[B] 高应激→fear: {hi['fear']:.2f} > 低应激 {lo['fear']:.2f} "
          f"{'OK' if b_ok else 'FAIL'}")

    # C. 高 surprise vs 低 surprise
    hs = emo.update(energy=0.8, health=1.0, stress=0.0, surprise=1.0)
    ls = emo.update(energy=0.8, health=1.0, stress=0.0, surprise=0.0)
    c_ok = hs["curiosity"] > ls["curiosity"]
    print(f"[C] 高surprise→curiosity: {hs['curiosity']:.2f} > 低 {ls['curiosity']:.2f} "
          f"{'OK' if c_ok else 'FAIL'}")

    # D. 稳态 → calm
    d = emo.update(energy=1.0, health=1.0, stress=0.0, surprise=0.0)
    d_ok = d["calm"] > 0.5
    print(f"[D] 稳态→calm: {d['calm']:.2f} (应>0.5) {'OK' if d_ok else 'FAIL'}")

    # E. 恐惧调制探索率（行为激进）
    emo.update(energy=0.05, health=0.3, stress=0.9, surprise=0.1)  # 恐惧态
    fear_mod = emo.modulate_action(base_exploration=0.05)
    emo.update(energy=1.0, health=1.0, stress=0.0, surprise=0.0)   # 平静态
    calm_mod = emo.modulate_action(base_exploration=0.05)
    e_ok = fear_mod > calm_mod
    print(f"[E] 恐惧调制: 恐惧态探索率 {fear_mod:.2f} > 平静态 {calm_mod:.2f} "
          f"{'OK' if e_ok else 'FAIL'}")

    ok = a_ok and b_ok and c_ok and d_ok and e_ok
    print(f"\n判定: {'OK 通过' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
