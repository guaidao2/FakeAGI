"""
P8c 句法层验证 — 词序学习 + 短语组合（语法从经验来，非硬编码）

验证项：
  A. 词序学习：高频 bigram（"food east"×10）被识别为合法顺序（score>0.8）
  B. 短语生成：按经验顺序组词（输出 "food east" 而非 "east food"）
  C. 顺序反转：经验变化（"east food" 变常见）后输出顺序跟着变（可学习语法）
  D. 与 select_words 集成：多词输出按句法排序
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from cognition.language.syntax import BigramTracker, PhraseBuilder, integrate_with_select
from cognition.language.grounding import LinguisticOrgan

VOCAB = ["food", "water", "east", "west", "north", "south"]


def main():
    print("=" * 60)
    print("P8c 句法层验证 — 词序学习 + 短语组合")
    print("=" * 60)

    # A. 词序学习
    tr = BigramTracker(VOCAB)
    for _ in range(10):
        tr.observe_sequence(["food", "east"])
    for _ in range(2):
        tr.observe_sequence(["east", "food"])
    score = tr.order_score("food", "east")
    a_ok = score > 0.8
    print(f"\n[A] 词序学习: food→east 得分 {score:.2f} (应>0.8) "
          f"{'OK' if a_ok else 'FAIL'}")

    # B. 短语生成：按经验顺序
    pb = PhraseBuilder(VOCAB)
    for _ in range(10):
        pb.learn([["food", "east"]])
    for _ in range(2):
        pb.learn([["east", "food"]])
    phrase = pb.build(["east", "food"], n_words=2)
    b_ok = phrase == ["food", "east"]
    print(f"[B] 短语生成: 输入 [east, food] → 输出 {phrase} (应 [food, east]) "
          f"{'OK' if b_ok else 'FAIL'}")

    # C. 顺序反转：经验翻转
    pb2 = PhraseBuilder(VOCAB)
    for _ in range(2):
        pb2.learn([["food", "east"]])
    for _ in range(10):
        pb2.learn([["east", "food"]])
    phrase2 = pb2.build(["east", "food"], n_words=2)
    c_ok = phrase2 == ["east", "food"]
    print(f"[C] 顺序反转: 经验翻转后输出 {phrase2} (应 [east, food]) "
          f"{'OK' if c_ok else 'FAIL'}")

    # D. 与 select_words 集成（LinguisticOrgan）
    organ = LinguisticOrgan(vocab_size=len(VOCAB))
    builder = PhraseBuilder(VOCAB)
    builder.learn([["food", "east"], ["water", "west"]])
    # select_words 输入是 LNN hidden 状态（64 维，probe_input_dim）
    state_vec = np.random.RandomState(0).randn(64).astype(np.float32)
    out = integrate_with_select(organ, builder, state_vec, VOCAB, n_words=2)
    d_ok = len(out) == 2 and all(w in VOCAB for w in out)
    print(f"[D] 集成: select_words+句法 → {out} (2 词合法) "
          f"{'OK' if d_ok else 'FAIL'}")

    ok = a_ok and b_ok and c_ok and d_ok
    print(f"\n判定: {'OK 通过' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
