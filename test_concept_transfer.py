"""
路线 B 实验 — 概念迁移（抽象概念形成）v2

v1 教训（设计缺陷，非实验失败）：
  1. 方向线索 4 布尔压成 1 标量 → 混叠不可辨识（"上更近"与"下更近"同值）
  2. Q 表是查表：外观维度变化 → 桶 key 全变 → 全未见状态 → argmax=0 停留
     ——查表无法测概念抽象（朋友"查表被排除"教训的实证）

v2 设计（人类式概念抽象）：
  训练跨两种表面（食物 0.2 + 水源 0.5）抽象出"可消耗物"概念，
  零样本测试第三种表面（果实 0.8）——见过苹果梨，识得香蕉。
  模型必须学到"外观无预测力"（两种表面下相同动作最优）→ 忽略外观维度。

关键修正：
  - 方向线索 one-hot（obs[3:7] 四向，无混叠）
  - Q 表 → 线性函数近似（LinearQ，θ 5×11）：泛化而非记忆
  - L2 正则：外观权重 |w_app|→0（抽象=丢弃表面）
  - 观测维度域无关（邻接/方向/位置/距离）与域特有（外观）显式分离

验证：
  A. 训练域可学（food+water 混合训练，平均 >2x 真随机）
  B. 零样本概念迁移（fruit 第三种表面 ≥40/50=80%，3 独立评估布局集
     取最差——固定先验阈值而非相对 2x，防基线波动）——**唯一判定**
  C. 少样本适配（fruit 30ep 迁移 vs 从头，迁移应更快）
  D. 诚实报告：外观权重 |w_app|（小=抽象成功；大=过拟合表面）
判定：B 通过即 OK（D 未过则结论降级"跨表面泛化"）
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class ConceptEnv:
    """三表面共享"可消耗物"概念环境（11D 观测）
    obs[0:3]  = 上/左/右 邻接可消耗物（域无关）
    obs[3:7]  = 上/左/右/下 是否更近（one-hot，域无关）
    obs[7:9]  = 位置（域无关）
    obs[9]    = 到可消耗物距离（域无关）
    obs[10]   = 外观特征（0.2 食物 / 0.5 水源 / 0.8 果实——域特有）"""
    def __init__(self, size=6, mode="food", seed=0):
        self.size = size
        self.mode = mode
        self.rng = np.random.RandomState(seed)
        self.pos = [0, 0]
        while True:
            self.item = [self.rng.randint(size), self.rng.randint(size)]
            if self.item != [0, 0]:
                break
        self.appearance = {"food": 0.2, "water": 0.5, "fruit": 0.8}.get(mode, 0.5)

    def observe(self):
        x, y = self.pos
        ix, iy = self.item
        dx, dy = ix - x, iy - y
        def adj(ax, ay):
            nx, ny = x + ax, y + ay
            return 1.0 if (nx, ny) == (ix, iy) else 0.0
        # one-hot：每方向是否更接近（无混叠）
        closer = []
        for ax, ay in [(0, -1), (-1, 0), (1, 0), (0, 1)]:
            ndx, ndy = dx - ax, dy - ay
            closer.append(1.0 if abs(ndx) + abs(ndy) < abs(dx) + abs(dy) else 0.0)
        dist = (abs(dx) + abs(dy)) / (self.size * 2)
        return np.array([
            adj(0, -1), adj(-1, 0), adj(1, 0),
            closer[0], closer[1], closer[2], closer[3],
            x / self.size, y / self.size,
            dist,
            self.appearance,
        ], dtype=np.float32)

    def step(self, a):
        dirs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dirs[a % 5]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            return 0.0, False
        self.pos = [nx, ny]
        if self.pos == self.item:
            return 1.0, True
        return 0.0, False

    def done(self):
        return self.pos == self.item


class LinearQ:
    """线性函数近似：Q(s,a) = θ_a · φ(s)（泛化而非查表记忆）
    L2 正则 → 无预测力维度（外观）权重趋 0（概念抽象）"""
    def __init__(self, obs_dim=11, n_actions=5, lr=0.05, gamma=0.95,
                 eps=0.3, l2=0.01):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.l2 = l2
        self.theta = np.zeros((n_actions, obs_dim))
        self.appearance_weights = None  # D 判定：外观维度权重

    def act(self, obs, train=True):
        if train and np.random.random() < self.eps:
            return np.random.randint(1, self.n_actions)
        q = self.theta @ obs
        return int(np.argmax(q))

    def learn(self, obs, a, r, obs_next, done):
        q = self.theta @ obs
        qn = self.theta @ obs_next
        target = r + (0 if done else self.gamma * np.max(qn))
        delta = target - q[a]
        # TD 更新 + L2 收缩（θ -= lr*δ*φ + l2*θ）
        self.theta[a] += self.lr * (delta * obs - self.l2 * self.theta[a])


def train_mixed(episodes_per=250, max_steps=40, seed=0, l2=0.01):
    """跨两种表面（food+water）混合训练——抽象"可消耗物"概念"""
    agent = LinearQ(l2=l2)
    for ep in range(episodes_per * 2):
        mode = "food" if ep % 2 == 0 else "water"
        env = ConceptEnv(size=6, mode=mode, seed=seed + ep)
        obs = env.observe()
        for _ in range(max_steps):
            a = agent.act(obs)
            r, done = env.step(a)
            obs_next = env.observe()
            agent.learn(obs, a, r, obs_next, done)
            obs = obs_next
            if done:
                break
        agent.eps = max(0.05, agent.eps * 0.998)
    return agent


def eval_policy(agent, mode, trials=50, max_steps=40, seed=500, train=False,
                random_baseline=False):
    reached = 0
    for t in range(trials):
        env = ConceptEnv(size=6, mode=mode, seed=seed + t * 7)
        obs = env.observe()
        for _ in range(max_steps):
            if random_baseline:
                a = np.random.randint(1, 5)
            else:
                a = agent.act(obs, train=train)
            _, done = env.step(a)
            if done:
                reached += 1
                break
            obs = env.observe()
    return reached


def main():
    print("=" * 60)
    print("路线 B — 概念迁移实验 v2（跨表面抽象'可消耗物'）")
    print("=" * 60)

    # 3 独立训练 seeds（review warn：D 单点脆弱——0.257 距 0.3 仅 0.043，
    # 换全局 seed 可能翻越；A/B/D 全改为每 seed 独立训练+独立评估布局，
    # 判定取最差——真正 3 seeds 鲁棒性）
    seeds = (42, 7, 2026)
    rows = []          # (seed, a_food, a_water, b_fruit, r_food, r_water, r_fruit, w_app)
    agents = []
    for s in seeds:
        np.random.seed(s)                      # 独立训练探索流
        agent = train_mixed(episodes_per=250, l2=0.05, seed=0)
        agents.append(agent)
        # 评估布局 seed=500+s*1000——三组布局完全分离
        # （security low：原 500+s 的 offset 35 是步长 7 整数倍，
        #   s=42 与 s=7 布局重叠 45/50，"3 独立布局"打折）
        es = 500 + s * 1000
        a_food = eval_policy(agent, "food", trials=50, seed=es)
        a_water = eval_policy(agent, "water", trials=50, seed=es)
        b_fruit = eval_policy(agent, "fruit", trials=50, seed=es)
        r_food = eval_policy(LinearQ(), "food", trials=50, seed=es,
                             random_baseline=True)
        r_water = eval_policy(LinearQ(), "water", trials=50, seed=es,
                              random_baseline=True)
        r_fruit = eval_policy(LinearQ(), "fruit", trials=50, seed=es,
                              random_baseline=True)
        w_app = float(np.abs(agent.theta[:, 10]).max())
        rows.append((s, a_food, a_water, b_fruit, r_food, r_water, r_fruit, w_app))

    # A. 训练域可学（每 seed 双域 >2x 该 seed 随机基线）
    a_oks = [r[1] > r[4] * 2 and r[2] > r[5] * 2 for r in rows]
    a_ok = all(a_oks)
    a_detail = ", ".join(f"seed{r[0]}: 食{r[1]}/{r[4]} 水{r[2]}/{r[5]}"
                         for r in rows)
    print(f"\n[A] 训练域可学(×3seeds): {a_detail} "
          f"(每seed双域>2x) {'OK' if a_ok else 'FAIL'}")

    # B. 零样本概念迁移（果实——第三种表面，训练从未见过 0.8）
    #    固定先验阈值（≥40/50=80%）；3 独立训练 seeds 取最差
    b_min = min(r[3] for r in rows)
    b_ok = b_min >= 40
    b_detail = ", ".join(f"seed{r[0]}: {r[3]}/{r[6]}" for r in rows)
    print(f"[B] 零样本概念迁移(×3seeds): {b_detail} "
          f"(最差≥40/50=80%) {'OK' if b_ok else 'FAIL'}")

    # D. 外观权重（3 seeds 取最差——鲁棒性；<0.3=抽象成功忽略表面）
    w_max = max(r[7] for r in rows)
    d_ok = w_max < 0.3
    w_detail = ", ".join(f"seed{r[0]}: {r[7]:.3f}" for r in rows)
    print(f"[D] 外观权重(×3seeds): {w_detail} (最差<0.3) "
          f"{'OK' if d_ok else 'FAIL'}")

    # C. 少样本适配（fruit 30ep 迁移 vs 从头——用末 seed agent）
    #    两者 eps 同衰减（0.3→0.05）——公平比较（review nit）
    agent = agents[-1]
    def finetune(init_theta=None, seed0=100):
        ag = LinearQ()
        if init_theta is not None:
            ag.theta = init_theta.copy()
        for ep in range(30):
            env = ConceptEnv(size=6, mode="fruit", seed=seed0 + ep)
            obs = env.observe()
            for _ in range(40):
                a = ag.act(obs)
                r, done = env.step(a)
                obs_next = env.observe()
                ag.learn(obs, a, r, obs_next, done)
                obs = obs_next
                if done:
                    break
            ag.eps = max(0.05, ag.eps * 0.998)
        return ag

    agent_few = finetune(init_theta=agent.theta)
    c_mig = eval_policy(agent_few, "fruit", trials=50)
    agent_scratch = finetune(init_theta=None)
    c_scratch = eval_policy(agent_scratch, "fruit", trials=50)
    c_ok = c_mig > c_scratch
    print(f"[C] 少样本适配(30ep): 迁移 {c_mig}/50 vs 从头 {c_scratch}/50 "
          f"(应迁移多) {'OK' if c_ok else 'FAIL'}")

    ok = b_ok  # 判定只看 B（A/C/D 如实记录）
    # 结论强度条件化：D 通过才声称"概念抽象（丢弃表面）"；
    # D 未过（外观权重未收敛）则降级"跨表面泛化（机制未明）"——review should-fix
    if ok:
        conclusion = ("概念抽象（丢弃表面）" if d_ok
                      else "跨表面泛化（机制未明——外观权重未收敛）")
    else:
        conclusion = "不成立"
    print(f"\n判定: {'OK 通过——第三种表面零样本迁移成立（' + conclusion + '）' if ok else 'FAIL'}")
    print(f"  [诚实报告] A {a_detail}"
          f"/ B {b_detail}"
          f"/ C 少样本 {c_mig} vs {c_scratch}"
          f"/ D {w_detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
