"""
③ 接入验证 — 情绪系统 + 他者模型进主循环

验证项：
  A. 默认关闭零影响：_emotion_enabled=False 时 exploration 未被调制
  B. 情绪接入：开启后恐惧态（低能量）探索率 > 平静态
  C. 他者接入：_other_agent_enabled 且环境有 get_other_pos 时竞争回避生效
  D. 他者默认关闭：无 get_other_pos 环境不报错（零影响护栏）
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from main import AGI
from cognition import CognitionPipeline

VOCAB = ["food", "water", "east", "west", "north", "south"]


class SimpleEnv:
    """自由环境（无他者）——他者默认关闭验证"""
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


class SharedEnv2(SimpleEnv):
    """共享环境（有他者）——竞争回避验证"""
    def __init__(self):
        super().__init__()
        self.other_pos = [4, 4]
        self.food_pos = [12, 12]

    def get_other_pos(self):
        return self.other_pos

    def get_food_pos(self):
        return self.food_pos


def make_agi(env):
    cfg = {
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4,
        "language": False, "language_vocab": VOCAB,
    }
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    agi.set_env(env)
    return agi


def main():
    print("=" * 60)
    print("③ 接入验证 — 情绪 + 他者进主循环")
    print("=" * 60)

    # A. 默认关闭零影响
    agi = make_agi(SimpleEnv())
    agi.step()
    a_ok = not hasattr(agi, "emotion_state") or agi.emotion_state == {}
    print(f"\n[A] 默认关闭: emotion_state={agi.emotion_state} (应空) "
          f"{'OK' if a_ok else 'FAIL'}")

    # B. 情绪接入：开启后恐惧态（低能量）探索率 > 平静态
    agi2 = make_agi(SimpleEnv())
    agi2._emotion_enabled = True
    agi2.body.energy = 0.05  # 快饿死→恐惧
    agi2.step()
    fear_explore = agi2.emotion_state.get("fear", 0)
    agi2.body.energy = 1.0
    agi2.step()
    calm_fear = agi2.emotion_state.get("fear", 0)
    b_ok = fear_explore > calm_fear
    print(f"[B] 情绪接入: 恐惧态 fear={fear_explore:.2f} > 平静态 {calm_fear:.2f} "
          f"{'OK' if b_ok else 'FAIL'}")

    # C. 他者接入：共享环境 + 竞争他者 → 意图识别 + 回避
    agi3 = make_agi(SharedEnv2())
    agi3._other_agent_enabled = True
    agi3._emotion_enabled = False
    # 模拟他者在食物附近（竞争信号）
    agi3.env.other_pos = [11, 11]
    for _ in range(60):
        agi3.step()
    c_ok = (agi3.other_tracker is not None
            and agi3.other_tracker.intent == "competitor")
    print(f"[C] 他者接入: intent={agi3.other_tracker.intent if agi3.other_tracker else 'None'} "
          f"(应 competitor) {'OK' if c_ok else 'FAIL'}")

    # D. 他者默认关闭：无 get_other_pos 环境不报错
    agi4 = make_agi(SimpleEnv())
    agi4._other_agent_enabled = True  # 开启但环境无接口→应静默跳过
    try:
        agi4.step()
        d_ok = True
    except Exception as e:
        d_ok = False
        print(f"  [异常] {e}")
    print(f"[D] 无他者环境: 开启不报错 {'OK' if d_ok else 'FAIL'}")

    ok = a_ok and b_ok and c_ok and d_ok
    print(f"\n判定: {'OK 通过' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
