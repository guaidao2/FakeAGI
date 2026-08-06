"""
P6 器官生长验证 — 高维输入自动生成感知器官

场景：
  1. AGI 在低维环境（8D）正常生活
  2. 环境突变：观测升为 64D 像素流（视觉模态）
     - 64D = 8x8 网格，其中食物位置编码为 4x4 亮斑
  3. 验证：
     - 器官生成器自动生成候选器官（超量生成）
     - 竞争期结束后有器官成熟（使用选择）
     - 成熟器官输出维度 = 配置 output_dim（下游无感）
     - 器官在生长协调器中注册（结构生长）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from main import AGI
from cognition import CognitionPipeline


class PixelEnv:
    """64D 像素流环境：8x8 网格，食物位置为 4x4 亮斑"""
    def __init__(self, size=8):
        self.size = size
        self.pos = [0, 0]
        self.food_pos = [size - 1, size - 1]
        self.food_eaten = 0
        self.steps = 0

    def get_pos(self):
        return self.pos

    def observe(self):
        """64D：每个格子亮度。食物处 4x4 亮斑（0.8），其余随机 0-0.1 噪声"""
        grid = np.random.uniform(0, 0.1, (self.size, self.size))
        fx, fy = self.food_pos
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    grid[nx, ny] = 0.8
        return grid.flatten().astype(np.float32)

    def step(self, action):
        self.steps += 1
        dirs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dirs[action % 5]
        nx = max(0, min(self.size - 1, self.pos[0] + dx))
        ny = max(0, min(self.size - 1, self.pos[1] + dy))
        moved = (nx, ny) != tuple(self.pos)
        self.pos = [nx, ny]
        energy_delta = -0.0014 if moved else -0.0002
        water_delta = -0.0002
        if tuple(self.pos) == tuple(self.food_pos):
            energy_delta = 0.3
            self.food_eaten += 1
        return {"energy_delta": energy_delta, "water_delta": water_delta}

    def food_nearby(self):
        return tuple(self.pos) == tuple(self.food_pos)


def main():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    print("P6: 器官生长验证 — 高维输入自动生成感知器官", flush=True)
    cfg = {
        "input_dim": 8, "self_state_dim": 14,
        "hidden_dim": 32, "n_actions": 5, "n_strategies": 4,
        "superposition_world": True,
        "organ_growth": True,          # 启用器官生长
        "organ_output_dim": 8,         # 器官输出维度
        "organ_competition_ticks": 80, # 竞争期长度（加速测试）
    }
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    env = PixelEnv()
    agi.set_env(env)
    gen = agi.cognition.organ_generator

    max_ticks = 600
    generated = False
    settled = False
    ate_at = None
    for t in range(max_ticks):
        agi.step()
        # 检查器官生成
        state = gen.get_state() if gen else {}
        if not generated and (state.get("candidates") or state.get("organs")):
            generated = True
            print(f"  [t={t}] 器官生成: 候选={state.get('candidates', {})}", flush=True)
        if not settled and state.get("organs"):
            settled = True
            print(f"  [t={t}] 器官成熟: {state.get('organs')} "
                  f"(生成 {gen.generation_count} 次, 凋亡 {gen.pruned_count})", flush=True)
        if env.food_eaten > 0 and ate_at is None:
            ate_at = t
        if ate_at is not None and t > ate_at + 100:
            break

    state = gen.get_state() if gen else {}
    print(f"\n器官状态: {state}", flush=True)
    print(f"吃到食物: t={ate_at}", flush=True)
    print(f"存活: {agi.alive}", flush=True)
    print(f"器官总数: {gen.next_organ_id if gen else 0}", flush=True)

    # 判定
    checks = []
    checks.append(("器官生成", generated))
    checks.append(("器官成熟", settled))
    checks.append(("找到食物", ate_at is not None))
    checks.append(("存活", agi.alive))
    if settled and gen:
        organ = list(gen.registry.values())[0][0]
        checks.append(("输出维度正确", organ.output_dim == 8))
        checks.append(("结构非空", len(organ.patches) > 0))

    passed = all(c for _, c in checks)
    print("\n判定: " + ("OK 通过" if passed else "FAIL 未通过"), flush=True)
    for name, ok in checks:
        print(f"  {'[OK]' if ok else '[X]'} {name}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
