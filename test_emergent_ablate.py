"""
E14 组合调优：模块消融（哪条线拖累全模块）

E14 全模块 1.6±1.0 vs 对照 3.4±2.7（对照反超）——消融定位拖累线。

组：
  full     = 语言+情绪+他者（E14 全模块）
  no_lang  = 无语言（情绪+他者）
  no_emo   = 无情绪（语言+他者）
  no_other = 无他者（语言+情绪）
  base     = 全关（对照）

判定：full 若显著低于某消融组 → 该线拖累；
      base 最高 → 模块组合整体负收益（装配本身未受益全模块）。
"""
import sys
import numpy as np
from main import AGI
from cognition import CognitionPipeline

VOCAB = ["food", "water", "east", "west", "north", "south", "threat"]


def make_agi(language=True, emotion=True, other=True):
    cfg = {
        "input_dim": 6, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4,
        "language": language, "language_vocab": VOCAB,
    }
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    if emotion:
        agi._emotion_enabled = True
    if other:
        agi._other_agent_enabled = True
    agi._transfer_selector_enabled = False
    return agi


def run_episode(language=True, emotion=True, other=True,
                max_ticks=10000, env_seed=1):
    from test_emergent import SharedWorld
    agi = make_agi(language, emotion, other)
    env = SharedWorld(seed=env_seed, language=language, other=other)
    agi.set_env(env)
    foods = 0
    for _ in range(max_ticks):
        before = agi.body.energy
        agi.step()
        if agi.body.energy > before + 0.01:
            foods += 1
        if not agi.alive:
            break
    return foods


def main():
    groups = [
        ("full",     True,  True,  True),
        ("no_lang",  False, True,  True),
        ("no_emo",   True,  False, True),
        ("no_other", True,  True,  False),
        ("base",     False, False, False),
    ]
    results = {}
    for name, lang, emo, other in groups:
        seeds = []
        for es in [1, 2, 3]:
            seeds.append(run_episode(lang, emo, other, env_seed=es))
        m, s = np.mean(seeds), np.std(seeds)
        results[name] = (m, s, seeds)
        print(f"  {name:8s}: 食物 {m:.1f}±{s:.1f} {seeds}")

    full = results["full"][0]
    print("\n  消融对比（full 是否显著低于某组）：")
    worst_gap = None
    for name in ["no_lang", "no_emo", "no_other"]:
        gap = results[name][0] - full
        tag = "拖累" if gap > 1.0 else ("无显著" if abs(gap) <= 1.0 else "反助")
        print(f"    {name:8s}: Δ={gap:+.1f} → {tag}")
        if gap > 1.0 and (worst_gap is None or gap > worst_gap[1]):
            worst_gap = (name, gap)
    base_gap = results["base"][0] - full
    print(f"    base     : Δ={base_gap:+.1f} "
          f"→ {'模块组合负收益' if base_gap > 1.0 else '组合不劣于对照'}")

    if worst_gap:
        print(f"\n  最拖累线: {worst_gap[0]} (Δ={worst_gap[1]:+.1f})")
    else:
        print("\n  无显著单线拖累——模块组合整体问题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
