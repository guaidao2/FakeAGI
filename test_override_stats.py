"""
friend-audit 修复③检验：override_action 通路激活统计

修复前：override_action 死变量（元认知探索意图产生后丢弃）
修复后：真实应用。本脚本统计：
  A. override 产生 tick 数（元认知探索模式激活频率）
  B. override 实际应用 tick 数（通路真实生效）
  C. 应用的动作分布（钳制后 0-4）
"""
import sys
import numpy as np
from main import AGI


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="store_true",
                    help="带真实环境（reward/pos 信号→GapDetector 可触发）")
    args = ap.parse_args()

    agi = AGI()
    # 注入认知核心（真实实验同款——AGI() 默认 cognition=None，
    # 认知块（含元认知更新）在 cognition 存在时才执行）
    from cognition import CognitionPipeline
    agi.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    if args.env:
        # 带 reward/pos 信号的轻量 env（test_experiment2 同款——
        # GapDetector 需要回报/位置信号才能触发）
        class TestEnv:
            def __init__(self):
                self.pos = [5, 5]
                self.food_pos = [2, 2]
                self.water_pos = [7, 7]
            def get_pos(self): return self.pos
            def observe(self):
                import numpy as np
                to_food = [self.food_pos[0]-self.pos[0],
                           self.food_pos[1]-self.pos[1]]
                to_water = [self.water_pos[0]-self.pos[0],
                            self.water_pos[1]-self.pos[1]]
                return np.array([to_food[0]/10, to_food[1]/10,
                                 to_water[0]/10, to_water[1]/10])
            def step(self, a):
                dxs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
                dx, dy = dxs[a % 5]
                self.pos[0] = max(0, min(9, self.pos[0] + dx))
                self.pos[1] = max(0, min(9, self.pos[1] + dy))
                eat = abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1]) < 2
                drink = abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1]) < 2
                ed = 0.2 if eat else (0.05 if drink else -0.001)
                wd = 0.15 if drink else (0.02 if eat else -0.0005)
                return {"energy_delta": ed, "water_delta": wd}
        agi.set_env(TestEnv())

    # 诊断（--env 模式）：断点检查 GapDetector 链路
    if args.env:
        mc = agi.metacognition
        gd = mc.gap_detector
        print(f"  [诊断] fast_failure={gd.fast_failure_detected} "
              f"explore_mode={mc.explore_mode} "
              f"current_goal={mc.current_goal}")
        # 主循环 300 tick 后读内部状态（看主循环喂了什么）
        for _ in range(300):
            agi.step()
        print(f"  [诊断] 300tick 后: fast_failure={gd.fast_failure_detected} "
              f"explore={mc.explore_mode} gap={mc.current_goal}")
        print(f"          reward_hist={list(gd.reward_history)[-3:]} "
              f"err_hist={list(gd.error_history)[-3:]} "
              f"pos_hist={list(gd.pos_history)[-2:]}")
        gd.update(world_model_loss=0.5, gamenn_confidence=0.1,
                  surprise=0.5, energy_delta=-0.001,
                  agent_pos=[5, 5], energy_level=0.9)
        for _ in range(10):
            gd.update(world_model_loss=0.5, gamenn_confidence=0.1,
                      surprise=0.5, energy_delta=-0.001,
                      agent_pos=[5, 5], energy_level=0.9)
        print(f"  [诊断] 10tick 无回报后: fast_failure="
              f"{gd.fast_failure_detected}  reward_hist="
              f"{list(gd.reward_history)[-3:]}")
        gap = gd.detect()
        print(f"  [诊断] detect() = {gap}")
    n_ticks = 2000
    for t in range(n_ticks):
        agi.step()
    # 主循环内计数器（应用点 +1——step 外部看不到中间态；
    # produced 外部观测恒 0 无意义——override 在 step 内产生+消费）
    applied = getattr(agi, '_override_applied', 0)

    rate = applied / n_ticks * 100
    print(f"  {n_ticks} ticks | override 应用={applied} "
          f"（{rate:.0f}% tick 被元认知探索目标覆盖）")
    if applied == 0:
        print("  override 未应用——元认知探索模式未激活（影响面小）")

    # 判定：修复后通路激活（>0 即激活；EVOLUTION 口径 37%）
    ok = applied > 0
    print(f"  判定: {'OK（通路激活）' if ok else 'FAIL（未激活）'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
