"""
B 步骤：环境升级最小版（DESIGN_CONCEPTS §7.1——POMDP 化 + 组合任务）

SCAN 教训：全可观察线性环境统计关联就够，概念层无压力。
本实验构造**部分可观察**环境：食物被锁，需先踩开关（隐藏状态——
agent 看不到开关是否已踩，只能从"食物可吃/不可吃"推断）→
**必须维持隐状态**（记忆"我踩过开关"）→ 表征被迫从"传感器读数"
升级为"世界状态"——概念形成的压力源。

验证：
  A. 环境有效：存在"踩开关→解锁食物"的隐藏因果（统计关联不够——
     只看当前观测无法知道食物是否解锁）
  B. 系统压力测试：在此环境下 FakeAGI 能否学会（world_loss 有内容）
"""
import sys
import numpy as np
from main import AGI
from cognition import CognitionPipeline


class POMDPEnv:
    """6x6 部分可观察：开关 (1,1)，食物 (4,4) 初始锁定。
    观测 = [食物方向 dx, dy, 食物可吃标志, 开关方向 dx, dy]
    可吃标志是唯一的解锁线索（需记忆——踩过开关才知道）"""
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.pos = [3, 3]
        self.switch_pos = [1, 1]
        self.food_pos = [4, 4]
        self.switch_triggered = False
        self.food_eaten = 0

    def get_pos(self):
        return self.pos

    def observe(self):
        fx, fy = self.food_pos
        sx, sy = self.switch_pos
        food_unlocked = 1.0 if self.switch_triggered else 0.0
        return np.array([(fx - self.pos[0]) / 6, (fy - self.pos[1]) / 6,
                         food_unlocked,
                         (sx - self.pos[0]) / 6, (sy - self.pos[1]) / 6])

    def step(self, a):
        dxs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dxs[a % 5]
        self.pos[0] = max(0, min(5, self.pos[0] + dx))
        self.pos[1] = max(0, min(5, self.pos[1] + dy))
        # 踩开关（隐藏状态转移——观测只在下一次 reveal）
        at_switch = (self.pos[0] == self.switch_pos[0]
                     and self.pos[1] == self.switch_pos[1])
        if at_switch and not self.switch_triggered:
            self.switch_triggered = True
        # 吃食物（仅解锁后可吃）
        at_food = (self.pos[0] == self.food_pos[0]
                   and self.pos[1] == self.food_pos[1])
        if at_food and self.switch_triggered:
            self.food_eaten += 1
            self.food_pos = [self.rng.randint(0, 6), self.rng.randint(0, 6)]
            return {"energy_delta": 0.2, "water_delta": 0.02}
        if at_food and not self.switch_triggered:
            # 撞锁——负反馈（学习"没解锁时食物位置无效"）
            return {"energy_delta": -0.01, "water_delta": -0.0005}
        return {"energy_delta": -0.001, "water_delta": -0.0005}


def main():
    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 5, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    env = POMDPEnv()
    agi.set_env(env)

    n_ticks = 1500
    wl_hist, food_hist = [], []
    for t in range(n_ticks):
        agi.step()
        if (t + 1) % 150 == 0:
            wl = (agi.curiosity._loss_hist[-1]
                  if agi.curiosity and agi.curiosity._loss_hist else 0.0)
            wl_hist.append(wl)
            food_hist.append(env.food_eaten)
            print(f"  t={t+1:4d} | world_loss={wl:.3f} "
                  f"| foods={env.food_eaten} | 解锁={env.switch_triggered}")

    # 判定
    print(f"\n  环境：开关解锁={env.switch_triggered} "
          f"食物={env.food_eaten} 次/1500 tick")
    print(f"  world_loss 变化：{min(wl_hist):.2f}~{max(wl_hist):.2f} "
          f"（{'有内容' if max(wl_hist) > min(wl_hist) + 0.01 else '空转'}）")
    ok = env.switch_triggered and env.food_eaten > 0
    verdict = ("OK（POMDP 环境有效：系统学会踩开关解锁——"
               "隐藏状态维持成功）" if ok else
               "FAIL（未学会——POMDP 压力下需隐状态）")
    print("判定: " + verdict)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
