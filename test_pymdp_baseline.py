"""
FakeAGI vs pymdp（Active Inference）基线对比 —— E6 规则变化适应

Fable 批评："你的预测误差机制 = active inference 的重新发现——去跑 pymdp
基线，你的工作才开始有坐标系。"

本实验：同一环境（网格 + 食物/水 + 稳态需求），规则变化（食物位置移动），
对比：
  A. FakeAGI（完整认知管线——LNN+世界模型+GameNN+概念）
  B. pymdp Agent（active inference——变分自由能最小化 + 在线 A/B 学习）

判据：
  1. 规则变化前存活 tick（稳态维持能力）
  2. 规则变化后的适应速度（恢复性能所需 tick）
  3. 稳态维持（能量/水在安全区的时间比例）

环境（公平离散化）：5x5 网格（25 位置状态）+ 资源类型（食物/水/空）
  = 25*3 = 75 隐藏状态；观测 = 8 方向 + 资源类型（离散 8*3=24 或简化）；
  动作 = 4 方向 + 停留（5）。

pymdp 生成模型：
  A（似然）：状态→观测映射（位置+资源 → 方向+类型）
  B（转移）：动作→状态转移
  C（先验偏好）：偏好"高能量/高水"状态（等价 FakeAGI 的 V 价值）
  在线学习：update_A/update_B（规则变化后 A/B 自适应）
"""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pymdp import utils
from pymdp.agent import Agent


# ─── 共享环境（离散化——两系统同一接口）───
class GridEnv:
    """5x5 网格 + 食物/水 + 稳态需求（能量/水递减）+ 规则变化（食物移动）"""
    SIZE = 5
    N_ACTIONS = 5  # 上/左/右/下/停留（与 FakeAGI 动作对齐）

    def __init__(self, seed=0, change_at=None):
        self.rng = np.random.RandomState(seed)
        self.pos = [2, 2]
        self.food_pos = [0, 0]
        self.water_pos = [4, 0]
        self.energy = 1.0   # [0, 2]
        self.water = 1.0    # [0, 1]
        self.tick = 0
        self.change_at = change_at
        self.changed = False
        self.eats_before = 0
        self.eats_after = 0

    def observe_direction(self):
        """8 方向离散观测：食物相对方向（+资源类型单独模态）
        0=同格 1=上 2=左上 3=左 4=左下 5=下 6=右下 7=右 8=右上"""
        dx = self.food_pos[0] - self.pos[0]
        dy = self.food_pos[1] - self.pos[1]
        # 简化 4 方向（与动作对齐）：0=同格 1=上 2=左 3=右 4=下
        if dx == 0 and dy == 0:
            return 0
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        return 1 if dy > 0 else 4

    def observe(self):
        """FakeAGI 连续观测（main.py 期望）——食物/水方向向量"""
        dx = self.food_pos[0] - self.pos[0]
        dy = self.food_pos[1] - self.pos[1]
        wx = self.water_pos[0] - self.pos[0]
        wy = self.water_pos[1] - self.pos[1]
        return np.array([dx / self.SIZE, dy / self.SIZE,
                         wx / self.SIZE, wy / self.SIZE], dtype=np.float32)

    def observe_water_dir(self):
        """水的方向（pymdp 第 2 方向模态）"""
        dx = self.water_pos[0] - self.pos[0]
        dy = self.water_pos[1] - self.pos[1]
        if dx == 0 and dy == 0:
            return 0
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        return 1 if dy > 0 else 4

    def observe_type(self):
        """资源类型：0=无 1=食物 2=水（当前格）"""
        if self.pos == self.food_pos:
            return 1
        if self.pos == self.water_pos:
            return 2
        return 0

    def step(self, a):
        self.tick += 1
        # 规则变化：食物移到对角（第 change_at tick 生效一次）
        if (self.change_at is not None and self.tick >= self.change_at
                and not self.changed):
            self.food_pos = [4, 4]
            self.changed = True
        dxs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dxs[a % 5]
        self.pos[0] = max(0, min(self.SIZE - 1, self.pos[0] + dx))
        self.pos[1] = max(0, min(self.SIZE - 1, self.pos[1] + dy))
        # 吃/喝判定（与 test_concept_gate 一致：邻近 + 停留 a==0——
        # 两系统公平：FakeAGI 反射/概念停留、pymdp 推断后停留均可达）
        near_food = (abs(self.pos[0] - self.food_pos[0]) +
                     abs(self.pos[1] - self.food_pos[1]) < 3)
        near_water = (abs(self.pos[0] - self.water_pos[0]) +
                      abs(self.pos[1] - self.water_pos[1]) < 2)
        eat = near_food and a == 0
        drink = near_water and a == 0
        if eat:
            self.energy = min(2.0, self.energy + 0.5)
            if self.changed:
                self.eats_after += 1
            else:
                self.eats_before += 1
        if drink:
            self.water = min(1.0, self.water + 0.4)
        self.energy -= 0.004   # 代谢（两系统一致——review 修复：原 FakeAGI
        self.water -= 0.002    # 端 delta=0 靠 body base（~0.0005）=10× 不对称）
        # FakeAGI 适配：delta 含环境代谢（body base 额外 -0.0003 披露——
        # 总代谢 FakeAGI ≈0.0043 vs pymdp 0.004——7% 差异如实记录）
        ed = 0.5 if eat else -0.004
        wd = 0.4 if drink else -0.002
        return {"energy_delta": ed, "water_delta": wd,
                "energy": self.energy, "water": self.water,
                "ate": eat, "drank": drink}

    def alive(self):
        return self.energy > 0.0 and self.water > 0.0


# ─── B. pymdp Agent（active inference）───
def build_pymdp_agent(seed=0):
    """生成模型（单因子位置 25）：
    A（似然/感知模型）= 食物在 [0,0] 的先验——方向观测 + 类型观测
      （规则变化后 A 错误→在线学习 infer_parameters 适应——正是测试点）
    B（转移）= 位置随动作确定性移动（先验结构——对标 FakeAGI 本能）
    C（偏好）= 偏好"看到食物"（类型=1）——对标 FakeAGI 的 V 价值
    D（先验）= 位置均匀
    动作 = 5（上/左/右/下/停留——与 FakeAGI 一致）"""
    SIZE = 5
    num_states = [25]
    num_obs = [5, 5, 3]       # 食物方向 + 水方向 + 类型
    num_actions = [5]
    key = jax.random.PRNGKey(seed)
    food_init = [0, 0]        # 初始食物位置（规则变化后需学习）
    water_init = [4, 0]       # 水固定

    def direction(dx, dy):
        if dx == 0 and dy == 0:
            return 0
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        return 1 if dy > 0 else 4

    # A[0] 食物方向似然 (5, 25)
    A_food = np.zeros((5, 25))
    for x in range(SIZE):
        for y in range(SIZE):
            pos = x * SIZE + y
            d = direction(food_init[0] - x, food_init[1] - y)
            A_food[d, pos] = 1.0
    # A[1] 水方向似然 (5, 25)
    A_water = np.zeros((5, 25))
    for x in range(SIZE):
        for y in range(SIZE):
            pos = x * SIZE + y
            d = direction(water_init[0] - x, water_init[1] - y)
            A_water[d, pos] = 1.0
    # A[2] 类型似然 (3, 25)
    A_type = np.zeros((3, 25))
    for x in range(SIZE):
        for y in range(SIZE):
            pos = x * SIZE + y
            if [x, y] == food_init:
                A_type[1, pos] = 1.0
            elif [x, y] == water_init:
                A_type[2, pos] = 1.0
            else:
                A_type[0, pos] = 1.0
    A = [jnp.array(A_food), jnp.array(A_water), jnp.array(A_type)]

    # B 位置转移 (25, 25, 5)
    B_pos = np.zeros((25, 25, 5))
    dxs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
    for a in range(5):
        dx, dy = dxs[a]
        for x in range(SIZE):
            for y in range(SIZE):
                nx = max(0, min(SIZE - 1, x + dx))
                ny = max(0, min(SIZE - 1, y + dy))
                B_pos[nx * SIZE + ny, x * SIZE + y, a] = 1.0
    B = [jnp.array(B_pos)]
    # C：偏好食物+水（类型 1=食物 2=水——对标 FakeAGI 双驱动力）
    C = [jnp.ones(5) * 0.5, jnp.ones(5) * 0.5, jnp.array([0.1, 2.0, 1.5])]
    D = [jnp.ones(25) / 25.0]
    agent = Agent(A=A, B=B, C=C, D=D, num_controls=num_actions,
                  categorical_obs=True, policy_len=2)
    return agent


def run_pymdp(env, agent, max_ticks=600, rng_key=None,
              lr_pA=0.5, lr_pB=0.5, learn_b=True):
    """pymdp agent 主循环：观测→推断→策略→行动→学习
    返回存活 tick（食物统计在 env 计数器）
    扩展（分论文一抗辩实验）：lr_pA/lr_pB 可扫；learn_b=False=只学 A"""
    key = jax.random.PRNGKey(42) if rng_key is None else rng_key
    for t in range(max_ticks):
        # 观测（one-hot + batch：食物方向/水方向/类型）
        obs_dir = env.observe_direction()
        obs_water = env.observe_water_dir()
        obs_type = env.observe_type()
        obs = [jnp.eye(5)[obs_dir][None], jnp.eye(5)[obs_water][None],
               jnp.eye(3)[obs_type][None]]
        # 推断 + 策略 + 行动
        qs = agent.infer_states(obs, empirical_prior=agent.D)
        q_pi = agent.infer_policies(qs)
        # 手动确定性选策略（绕开 sample_action 的 vmap tuple bug——
        # pymdp 1.0.3 beta 缺陷）：q_pi[0] = (num_policies,) 后验
        pi_idx = int(np.asarray(q_pi[0]).argmax())
        # policy_arr 形状 (num_policies, policy_len, num_factors)
        a_idx = int(np.asarray(agent.policies.policy_arr)[pi_idx][0][0])
        a_idx = min(a_idx, env.N_ACTIONS - 1)
        # 环境步进
        env.step(a_idx)
        # 在线学习（规则变化后 A/B 自适应——infer_parameters）
        if t % 2 == 0:  # 隔 tick 学习（历史窗口对齐）
            try:
                agent = agent.infer_parameters(
                    beliefs_A=qs, observations=obs,
                    actions=jnp.array([[[a_idx]]], dtype=jnp.int32),
                    beliefs_B=qs if learn_b else None,
                    lr_pA=lr_pA, lr_pB=lr_pB if learn_b else 0.0)
            except Exception as e:
                print(f"  [WARN] infer_parameters: {e}", flush=True)
        if not env.alive():
            return t
    return max_ticks


# ─── A. FakeAGI（复用 test_concept_gate 的环境适配）───
def run_fakeagi(env, max_ticks=600, seed=0):
    from main import AGI
    from cognition import CognitionPipeline
    np.random.seed(1000 + seed)
    import torch
    torch.manual_seed(1000 + seed)
    agi = AGI()
    agi.set_env(env)
    agi.set_cognition(CognitionPipeline({}))
    agi.metacognition = None
    agi._info_seek_enabled = False
    for _ in range(max_ticks):
        agi.step()
        if not agi.alive:
            return agi.tick
    return max_ticks


def main():
    import os
    print("=" * 60)
    print("FakeAGI vs pymdp（Active Inference）基线对比")
    print("=" * 60)
    # 分论文一抗辩实验模式（环境变量）：
    #   PM_SCAN=1    pymdp lr 扫描（0.1/0.5/1.0/2.0 × 3seeds）
    #   PM_LONG=1    长时窗 2000 tick（FakeAGI vs pymdp × 3seeds）
    #   PM_AONLY=1   只学 A（排除 B 学习干扰）
    scan = os.environ.get("PM_SCAN", "0") == "1"
    longw = os.environ.get("PM_LONG", "0") == "1"
    aonly = os.environ.get("PM_AONLY", "0") == "1"
    N = 3 if (scan or longw or aonly) else 5
    change_at = 300 if not longw else 1000  # 长时窗变化点后移
    max_t = 2000 if longw else 600

    if scan:
        print("\n=== pymdp lr 扫描（变化后食物——适应速度）===")
        for lr in (0.1, 0.5, 1.0, 2.0):
            ea_vals = []
            for s in range(N):
                e2 = GridEnv(seed=s, change_at=change_at)
                ag = build_pymdp_agent(seed=s)
                run_pymdp(e2, ag, max_ticks=max_t, lr_pA=lr, lr_pB=lr)
                ea_vals.append(e2.eats_after)
            print(f"  lr={lr}: 变化后食物 {np.mean(ea_vals):.1f}±{np.std(ea_vals):.1f}"
                  f"（{ea_vals}）", flush=True)
        print("\n  解读: 若高 lr 显著提升变化后食物→多通道优势需修正（诚实记录）")
        return

    if aonly:
        print("\n=== pymdp 只学 A（排除 B 学习干扰）===")
        for lb in (True, False):
            ea_vals = []
            for s in range(N):
                e2 = GridEnv(seed=s, change_at=change_at)
                ag = build_pymdp_agent(seed=s)
                run_pymdp(e2, ag, max_ticks=max_t, learn_b=lb)
                ea_vals.append(e2.eats_after)
            print(f"  学B={'是' if lb else '否'}: 变化后食物 {np.mean(ea_vals):.1f}"
                  f"±{np.std(ea_vals):.1f}（{ea_vals}）", flush=True)
        return

    fa_ticks, pm_ticks = [], []
    fa_eb, fa_ea, pm_eb, pm_ea = [], [], [], []
    for s in range(N):
        # FakeAGI
        e1 = GridEnv(seed=s, change_at=change_at)
        t1 = run_fakeagi(e1, max_ticks=max_t, seed=s)
        fa_ticks.append(t1)
        fa_eb.append(e1.eats_before)
        fa_ea.append(e1.eats_after)
        # pymdp
        e2 = GridEnv(seed=s, change_at=change_at)
        ag = build_pymdp_agent(seed=s)
        t2 = run_pymdp(e2, ag, max_ticks=max_t)
        pm_ticks.append(t2)
        pm_eb.append(e2.eats_before)
        pm_ea.append(e2.eats_after)
        print(f"  seed{s}: FakeAGI={t1}t(食{int(e1.eats_before)}/{int(e1.eats_after)})"
              f" pymdp={t2}t(食{int(e2.eats_before)}/{int(e2.eats_after)})",
              flush=True)
    print(f"\n  存活 tick（×{N}seeds 均值）:")
    print(f"    FakeAGI: {np.mean(fa_ticks):.0f} ± {np.std(fa_ticks):.0f}")
    print(f"    pymdp:   {np.mean(pm_ticks):.0f} ± {np.std(pm_ticks):.0f}")
    print(f"\n  食物获取（变化前/变化后——规则变化适应速度）:")
    print(f"    FakeAGI: {np.mean(fa_eb):.1f} / {np.mean(fa_ea):.1f}")
    print(f"    pymdp:   {np.mean(pm_eb):.1f} / {np.mean(pm_ea):.1f}")
    print(f"\n  判定: 存活 {'FakeAGI 优' if np.mean(fa_ticks) > np.mean(pm_ticks) else 'pymdp 优'}"
          f"；适应 {'FakeAGI 优' if np.mean(fa_ea) > np.mean(pm_ea) else 'pymdp 优'}"
          f"——差异定位见 COMPARISON.md（非胜负，机制增量定位）")


if __name__ == "__main__":
    sys.exit(main())
