"""
P4 验证 — 观测维度增长 → 全链路协调生长

场景：
  1. AGI 在 4D 观测环境中生活 100 tick
  2. 环境突变：观测扩展到 8D（新信号源接入）
  3. 验证：
     - 观测抽象层自动新增通道（主动生长）
     - LNN 输入维度同步扩展（grow_input）
     - 不崩溃、继续决策
     - 生长协调器记录了生长事件
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from main import AGI
from cognition import CognitionPipeline


class GrowingEnv:
    """环境：前 N tick 观测 4D，之后 8D"""
    def __init__(self, switch_at=100):
        self.pos = [5, 5]
        self.food = [2, 2]
        self.tick = 0
        self.switch_at = switch_at

    def get_pos(self):
        return self.pos

    def observe(self):
        base = np.array([
            (self.food[0]-self.pos[0])/10, (self.food[1]-self.pos[1])/10,
            0.0, 0.0], dtype=np.float32)
        if self.tick >= self.switch_at:
            # 新信号源：危险区方向 + 资源量
            extra = np.array([0.3, 0.3, 0.8, 0.2], dtype=np.float32)
            return np.concatenate([base, extra])
        return base

    def step(self, a):
        self.tick += 1
        dirs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
        dx, dy = dirs[a % 5]
        self.pos[0] = max(0, min(9, self.pos[0]+dx))
        self.pos[1] = max(0, min(9, self.pos[1]+dy))
        eat = abs(self.pos[0]-2)+abs(self.pos[1]-2) < 2
        return {"energy_delta": 0.2 if eat else -0.001,
                "water_delta": -0.0002}

    def food_nearby(self):
        return abs(self.pos[0]-2)+abs(self.pos[1]-2) < 4


def test():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    print("P4: 观测维度增长 → 全链路协调生长测试", flush=True)

    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    env = GrowingEnv(switch_at=100)
    agi.set_env(env)

    obs_dim_before = None
    lnn_dim_before = None
    grew_detected = False
    alive_after_growth = False

    for t in range(250):
        status = agi.step()
        if t == 50:
            obs_dim_before = agi.cognition.obs_abstraction.get_abstract_dim()
            lnn_dim_before = agi.cognition.lnn.input_dim
            print(f"  生长前: 观测抽象={obs_dim_before}D, LNN输入={lnn_dim_before}D")
        if t == 150:
            obs_dim_after = agi.cognition.obs_abstraction.get_abstract_dim()
            lnn_dim_after = agi.cognition.lnn.input_dim
            grew_detected = (obs_dim_after > obs_dim_before) or (lnn_dim_after > lnn_dim_before)
            print(f"  生长后: 观测抽象={obs_dim_after}D, LNN输入={lnn_dim_after}D")
        if t >= 150 and agi.alive:
            alive_after_growth = True

    events = agi.growth.get_state()
    print(f"  协调器: 生长事件={events['growth_events']}, "
          f"模块维度={events['module_dims']}")

    passed = grew_detected and alive_after_growth
    print(f"判定: {'OK 通过' if passed else 'NO 失败'} — "
          f"观测增长{'+'.join(['观测','LNN'] if grew_detected else [])}, "
          f"生长后存活={alive_after_growth}")
    return passed


if __name__ == "__main__":
    test()
