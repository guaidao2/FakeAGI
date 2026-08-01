"""
⑥ 接入主循环验证 — 迁移价值评估在完整 AGI 循环中

验证项：
  A. 默认关闭零影响：_transfer_selector_enabled=False 时 set_env 无副作用
  B. 环境切换检测：开启后首次 set_env 不触发（无 old_env），第二次触发选择
  C. scratch 决策执行：可靠性低时切换环境→GameNN 重置（从头学）
  D. 迁移决策执行：可靠性高时切换环境→GameNN 保留（迁移）
  E. 容错：无 cognition 时 set_env 不报错（零影响护栏）
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from main import AGI
from cognition import CognitionPipeline

VOCAB = ["food", "water", "east", "west", "north", "south"]


class SimpleEnv:
    def __init__(self):
        self.pos = [8, 8]
        self.size = 16

    def observe(self):
        return np.zeros(4, dtype=np.float32)

    def get_pos(self):
        return self.pos

    def step(self, action):
        dirs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dirs[action % 5]
        self.pos[0] = max(0, min(self.size - 1, self.pos[0] + dx))
        self.pos[1] = max(0, min(self.size - 1, self.pos[1] + dy))
        return {"energy_delta": -0.002, "water_delta": -0.0005}

    def get_energy_delta(self, a):
        return -0.002

    def get_damage(self, a):
        return 0.0

    def food_nearby(self):
        return False


def make_agi():
    cfg = {
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4,
        "language": False, "language_vocab": VOCAB,
    }
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    return agi


def main():
    np.random.seed(42)
    print("=" * 60)
    print("⑥ 接入主循环验证 — 迁移价值评估")
    print("=" * 60)

    # A. 默认关闭零影响
    agi = make_agi()
    q_before = list(agi.cognition.gamenn.q_nets[0].parameters())[0].data.clone()
    agi.set_env(SimpleEnv())
    q_after = list(agi.cognition.gamenn.q_nets[0].parameters())[0].data.clone()
    a_ok = (agi.transfer_selector is None
            and torch_equal(q_before, q_after))
    print(f"\n[A] 默认关闭: selector={agi.transfer_selector} 权重未动 "
          f"{'OK' if a_ok else 'FAIL'}")

    # B. 开启后：首次 set_env 不触发（无 old_env），第二次触发
    agi2 = make_agi()
    agi2._transfer_selector_enabled = True
    agi2.set_env(SimpleEnv())  # 首次：不算切换
    b1 = agi2.transfer_selector is None
    agi2.set_env(SimpleEnv())  # 第二次：切换触发
    b2 = agi2.transfer_selector is not None
    b_ok = b1 and b2
    print(f"[B] 切换检测: 首次后 selector={agi2.transfer_selector if not b1 else 'None(正确)'} "
          f"二次后={agi2.transfer_selector is not None} "
          f"{'OK' if b_ok else 'FAIL'}")

    # C. scratch 决策：可靠性低（<0.60）→ 切换后 GameNN 重置
    agi3 = make_agi()
    agi3._transfer_selector_enabled = True
    agi3.set_env(SimpleEnv())
    # 初始可靠性 0.5 < 0.60 → scratch
    w_before = list(agi3.cognition.gamenn.q_nets[0].parameters())[0].data.clone()
    agi3.set_env(SimpleEnv())
    w_after = list(agi3.cognition.gamenn.q_nets[0].parameters())[0].data.clone()
    c_ok = agi3.transfer_choice == "scratch" and not torch_equal(w_before, w_after)
    print(f"[C] scratch执行: choice={agi3.transfer_choice}, 权重已重置 "
          f"{'OK' if c_ok else 'FAIL'}")

    # D. 迁移决策：高可靠性（≥0.60）→ 保留权重
    agi4 = make_agi()
    agi4._transfer_selector_enabled = True
    from core.transfer_selector import TransferSelector
    sel = TransferSelector(min_reliability=0.60)
    # 注入同构经验（5 次迁移胜：32 vs 18）→ 可靠性升到 >0.60
    for _ in range(5):
        sel.observe_feedback(32.0, 18.0)
    agi4.transfer_selector = sel
    agi4.set_env(SimpleEnv())  # 首次：不算切换
    w_b4 = list(agi4.cognition.gamenn.q_nets[0].parameters())[0].data.clone()
    agi4.set_env(SimpleEnv())  # 第二次：切换触发决策
    w_a4 = list(agi4.cognition.gamenn.q_nets[0].parameters())[0].data.clone()
    d_ok = agi4.transfer_choice == "transfer" and torch_equal(w_b4, w_a4)
    print(f"[D] 迁移执行: choice={agi4.transfer_choice}, 权重保留 "
          f"{'OK' if d_ok else 'FAIL'}")

    # E. 容错：无 cognition 时开启不报错
    agi5 = AGI()
    agi5._transfer_selector_enabled = True
    try:
        agi5.set_env(SimpleEnv())
        agi5.set_env(SimpleEnv())
        e_ok = True
    except Exception as ex:
        e_ok = False
        print(f"  [异常] {ex}")
    print(f"[E] 无认知容错: {'OK' if e_ok else 'FAIL'}")

    ok = a_ok and b_ok and c_ok and d_ok and e_ok
    print(f"\n判定: {'OK 通过' if ok else 'FAIL'}")
    return 0 if ok else 1


def torch_equal(a, b):
    return bool((a == b).all().item())


if __name__ == "__main__":
    sys.exit(main())
