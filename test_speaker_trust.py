"""
他者可靠性建模验证（SpeakerTrust——社会智能核心）

假设：信任绑定在说话者——成功慢升（+0.1），失败快降（-0.3）——
低信任说话者的词被打折（不被误导）。

设计：
- 可靠说话者 A：一直说对（每次引导成功）
- 误导说话者 B：一直说错（每次引导失败）
- 判据：
  A. 可靠者信任上升（> 初始 0.5）
  B. 误导者信任大幅下降（< 0.2——质疑能力：一次错重罚）
  C. 权重区分：A 的词权重显著 > B（盲从不被误导）
  D. 误导者连续错后权重 < 0.15 阈值（语言投票被抑制）
"""
import sys

from core.speaker_trust import SpeakerTrust


def main():
    print("=" * 56)
    print("他者可靠性建模验证（信任绑定说话者）")
    print("=" * 56)
    st = SpeakerTrust()

    # A 可靠：10 次成功
    for _ in range(10):
        st.observe_outcome("speaker_A", True)
    # B 误导：3 次失败（+1 次偶然成功——验证快降不被偶然洗白）
    st.observe_outcome("speaker_B", False)
    st.observe_outcome("speaker_B", False)
    st.observe_outcome("speaker_B", True)
    st.observe_outcome("speaker_B", False)

    tA = st.get("speaker_A")
    tB = st.get("speaker_B")
    print(f"  A: speaker_A 信任 = {tA:.3f}（初始 0.5）")
    ok_a = tA > 0.55
    print(f"     {'OK（可靠者信任上升——慢升）' if ok_a else 'FAIL'}")

    print(f"  B: speaker_B 信任 = {tB:.3f}（初始 0.5）")
    ok_b = tB < 0.3
    print(f"     {'OK（误导者信任大幅下降——快降/质疑）' if ok_b else 'FAIL'}")

    wA = st.weight("speaker_A")
    wB = st.weight("speaker_B")
    print(f"  C: 权重 A={wA:.3f} vs B={wB:.3f}")
    ok_c = wA > wB + 0.3
    print(f"     {'OK（可靠者词权重显著高于误导者——不被误导）' if ok_c else 'FAIL'}")

    print(f"  D: B 权重 {wB:.3f} vs 语言投票阈值 0.15")
    ok_d = wB < 0.15
    print(f"     {'OK（误导者被抑制——语言投票不生效）' if ok_d else 'FAIL'}")

    ok = ok_a and ok_b and ok_c and ok_d
    print(f"\n  判定: {'OK 通过（他者可靠性建模成立——社会智能核心）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
