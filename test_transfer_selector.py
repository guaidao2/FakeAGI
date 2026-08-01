"""
⑥ 迁移价值评估器验证 — "何时能迁移"的元认知

⑤ 揭示真正门槛：关系映射同构性决定迁移成败。本实验让系统**学会判断**
"该不该迁移"——用迁移后的真实反馈（迁移性能 vs 从头学性能）更新
迁移可靠性，然后对未知域做选择。

域设计：
  域 A（迷宫）：墙不可走 + 目标——训练源策略
  域 B（威胁场）：威胁可走扣血 + 中央绕行岛——与 A 关系同构（避障碍+朝目标）
  域 C（逆场）：威胁=奖励区（+0.5），空地 -0.1——与 A 关系**反转**
    （避威胁策略在 C 是灾难）→ 迁移必有害

验证：
  A. 域 B 迁移后：迁移可靠性**升**（同构——迁移有用）
  B. 域 C 迁移后：迁移可靠性**降**（异构——迁移有害）
  C. 未知域选择：同构域选 transfer、异构域选 scratch（按估计）
  D. 对比：有评估器 vs 无脑迁移（无脑在异构域被拖累）
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from core.transfer_selector import TransferSelector


class RelEnv:
    """三域环境：迷宫 / 威胁场 / 逆场（共享关系观测）"""
    def __init__(self, size=6, mode="maze", seed=0):
        self.size = size
        self.mode = mode
        self.rng = np.random.RandomState(seed)
        self.pos = [0, 0]
        self.goal = [size - 1, size - 1]
        self.history = []
        self.blocks = self._gen_blocks()

    def _gen_blocks(self):
        blocks = set()
        for y in range(self.size):
            for x in range(self.size):
                if (x, y) in [(0, 0), (self.size - 1, self.size - 1)]:
                    continue
                if self.rng.random() < 0.2:
                    blocks.add((x, y))
        if self.mode in ("threat", "inverse"):
            c = self.size // 2
            for y in range(c - 1, c + 1):
                for x in range(c - 1, c + 1):
                    if (x, y) not in [(0, 0), (self.size - 1, self.size - 1)]:
                        blocks.add((x, y))
        return blocks

    def observe(self):
        x, y = self.pos
        gx, gy = self.goal
        def obs_block(dx, dy):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                return 1.0
            return 1.0 if (nx, ny) in self.blocks else 0.0
        visit = sum(1 for p in self.history[:-1] if p == self.pos)
        return np.array([
            obs_block(0, -1), obs_block(-1, 0), obs_block(1, 0),
            (gx - x) / self.size, x / self.size, y / self.size,
            (abs(gx - x) + abs(gy - y)) / (self.size * 2),
            min(1.0, visit / 5.0),
        ], dtype=np.float32)

    def step(self, a):
        dirs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dirs[a % 5]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            return 0.0, False
        if self.mode == "maze" and (nx, ny) in self.blocks:
            return 0.0, False
        self.history.append(tuple(self.pos))
        old_d = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
        self.pos = [nx, ny]
        new_d = abs(nx - self.goal[0]) + abs(ny - self.goal[1])
        r = 0.1 if new_d < old_d else (-0.1 if new_d > old_d else 0.0)
        if self.mode == "threat" and (nx, ny) in self.blocks:
            return -0.5, False
        if self.mode == "inverse":
            # 逆场：关系完全反转——目标区（右下）是惩罚区，起点区（左上）是奖励区
            # 朝目标走的策略（学的朝 (5,5)）在这里是灾难
            dist_from_start = abs(nx - 0) + abs(ny - 0)
            if dist_from_start <= 2:
                return 0.5, False    # 靠近起点=奖励（反转目标）
            return -0.2, False       # 远离起点=惩罚
        return r, False

    def done(self):
        if self.mode == "inverse":
            return False  # 逆场无终点（反转目标：靠近起点=好）
        return self.pos == self.goal


class QTable:
    def __init__(self, n_bins=4, n_actions=5, lr=0.2, gamma=0.9, eps=0.3):
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.lr, self.gamma, self.eps = lr, gamma, eps
        self.Q = {}

    def _key(self, obs):
        return tuple(min(self.n_bins - 1, max(0, int(v * self.n_bins))) for v in obs)

    def _get(self, key):
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_actions)
        return self.Q[key]

    def act(self, obs, train=True):
        key = self._key(obs)
        if train and np.random.random() < self.eps:
            return np.random.randint(1, self.n_actions)
        return int(np.argmax(self._get(key)))

    def learn(self, obs, a, r, obs_next, done):
        key, nk = self._key(obs), self._key(obs_next)
        q = self._get(key)
        qn = self._get(nk)
        q[a] += self.lr * (r + (0 if done else self.gamma * np.max(qn)) - q[a])


def train_policy(mode, episodes=500, seed_base=0):
    """训练 Q 表策略（域 A 迷宫 / 或域内从头学）"""
    agent = QTable()
    for ep in range(episodes):
        env = RelEnv(mode=mode, seed=seed_base + ep)
        obs = env.observe()
        for _ in range(60):
            a = agent.act(obs)
            r, _ = env.step(a)
            done = env.done()
            obs_next = env.observe()
            agent.learn(obs, a, (1.0 if done else r), obs_next, done)
            obs = obs_next
            if done:
                break
        agent.eps = max(0.05, agent.eps * 0.99)
    return agent


def eval_policy(agent, mode, trials=50, max_steps=60, seed=500):
    if mode == "inverse":
        # 逆场性能：平均靠近起点距离（越小=越靠近起点=越好）
        total_dist = 0
        for t in range(trials):
            env = RelEnv(mode=mode, seed=seed + t * 7)
            obs = env.observe()
            for _ in range(max_steps):
                a = agent.act(obs, train=False)
                _, _ = env.step(a)
                obs = env.observe()
            d = abs(env.pos[0] - 0) + abs(env.pos[1] - 0)
            total_dist += d
        # 转成"性能分"：距离越小分越高（0-10 分制，0 距离=10 分）
        avg_d = total_dist / trials
        return max(0.0, 10.0 - avg_d)
    solved = 0
    for t in range(trials):
        env = RelEnv(mode=mode, seed=seed + t * 7)
        obs = env.observe()
        for _ in range(max_steps):
            a = agent.act(obs, train=False)
            _, _ = env.step(a)
            if env.done():
                solved += 1
                break
            obs = env.observe()
    return solved


def few_shot(agent_q, mode, episodes=30, seed_base=100):
    """少样本微调：复制 Q 表 + 目标域少量样本"""
    a = QTable()
    a.Q = {k: v.copy() for k, v in agent_q.Q.items()}
    for ep in range(episodes):
        env = RelEnv(mode=mode, seed=seed_base + ep)
        obs = env.observe()
        for _ in range(60):
            act = a.act(obs)
            r, _ = env.step(act)
            done = env.done()
            obs_next = env.observe()
            a.learn(obs, act, (1.0 if done else r), obs_next, done)
            obs = obs_next
            if done:
                break
    return a


def main():
    np.random.seed(42)
    print("=" * 60)
    print("⑥ 迁移价值评估器验证 — '何时能迁移'的元认知")
    print("=" * 60)

    # 1. 域 A 训练源策略
    src = train_policy("maze", episodes=500)
    print(f"\n[源] 域A迷宫策略训练完成")

    # 2. 迁移评估器初始化（阈值 0.60：需要更明确证据才迁移——保守元认知）
    sel = TransferSelector(min_reliability=0.60)

    # 3. 域 B 系列（同构威胁场）反馈 ×5：多次同构经验建立信任（更接近真实）
    for i in range(5):
        b_mig = eval_policy(few_shot(src, "threat", seed_base=100 + i * 50), "threat", seed=500 + i * 50)
        b_scr = eval_policy(few_shot(QTable(), "threat", seed_base=100 + i * 50), "threat", seed=500 + i * 50)
        sel.observe_feedback(b_mig, b_scr)
    r_after_b = sel.estimator.reliability
    a_ok = r_after_b > 0.60  # 与 C1 判定阈值一致（同构经验后应≥阈值）
    print(f"[A] 同构域B×5反馈: 迁移{b_mig} vs 从头{b_scr} (末次) → 可靠性 "
          f"{r_after_b:.2f} (应>0.5) {'OK' if a_ok else 'FAIL'}")

    # 3b. D（同构新威胁场）在 B 反馈后测——应选 transfer
    d_choice = sel.choose("unknown_threat_D")
    d_ok_tmp = d_choice == "transfer"
    print(f"[C1] 同构经验后 D(新威胁场): 可靠性 {r_after_b:.2f} "
          f"→ {d_choice} (应 transfer) {'OK' if d_ok_tmp else 'FAIL'}")

    # 4. 域 C 系列（异构逆场）反馈 ×5：多次异构经验建立"勿迁移"
    for i in range(5):
        c_mig = eval_policy(few_shot(src, "inverse", seed_base=200 + i * 50), "inverse", seed=600 + i * 50)
        c_scr = eval_policy(few_shot(QTable(), "inverse", seed_base=200 + i * 50), "inverse", seed=600 + i * 50)
        sel.observe_feedback(c_mig, c_scr)
    r_after_c = sel.estimator.reliability
    b_ok = r_after_c < r_after_b
    print(f"[B] 异构域C×5反馈: 迁移{c_mig} vs 从头{c_scr} (末次) → 可靠性 "
          f"{r_after_c:.2f} (应<{r_after_b:.2f}) {'OK' if b_ok else 'FAIL'}")

    # 5. E（异构新逆场）在 C 反馈后测——应选 scratch
    e_choice = sel.choose("unknown_inverse_E")
    c_ok = d_ok_tmp and e_choice == "scratch"
    print(f"[C2] 异构经验后 E(新逆场): 可靠性 {sel.estimator.reliability:.2f} "
          f"→ {e_choice} (应 scratch) {'OK' if c_ok else 'FAIL'}")

    # 6. 对比：有评估器（按选择器实际决策）vs 无脑迁移
    #    有评估器：D 选 transfer→用迁移策略；E 选 scratch→用从头策略
    d_pol = few_shot(src, "threat") if d_choice == "transfer" \
        else few_shot(QTable(), "threat")
    e_pol = few_shot(QTable(), "inverse") if e_choice == "scratch" \
        else few_shot(src, "inverse")
    sel_perf = eval_policy(d_pol, "threat") * 0.5 \
        + eval_policy(e_pol, "inverse") * 0.5
    #    无脑迁移：都从域 A 迁移
    brainless = (eval_policy(few_shot(src, "threat"), "threat") * 0.5
                 + eval_policy(few_shot(src, "inverse"), "inverse") * 0.5)
    d_ok = sel_perf > brainless
    print(f"[D] 对比(按选择器输出): 有评估器 {sel_perf:.1f} "
          f"(D={d_choice}, E={e_choice}) vs 无脑迁移 {brainless:.1f} "
          f"(应有评估器优) {'OK' if d_ok else 'FAIL'}")

    ok = a_ok and b_ok and c_ok and d_ok
    print(f"\n判定: {'OK 通过——迁移价值评估成立' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
