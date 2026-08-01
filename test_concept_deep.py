"""
① 概念深化 — MLP 非线性概念抽象（路线 B 的推广）

路线 B（线性）证明：L2 正则压外观权重 → "概念抽象（丢弃表面）"。
本实验问两个更深的问题：
  Q1. MLP（非线性）能否**天然**忽略外观（不靠显式权重压）？
      ——线性模型靠 L2 才收敛；MLP 的隐层特征空间理论上能把
       "外观无关"编码进去（外观维度对动作无预测力 → 隐层应
       学会不依赖它）。
  Q2. 更多训练表面（4 种）→ 第 5 种表面零样本迁移——鲁棒性？

设计（预注册，判定不因结果调整）：
  - 组 1：训练 food+water（2 表面，与路线 B 可比）→ 零样本测 fruit
  - 组 2：训练 food/water/fruit/mushroom（4 表面）→ 零样本测 berry
  - D 判定（行为学——MLP 无显式外观权重）：
      外观扰动不敏感：同状态外观 ±0.1，argmax 动作不变率 ≥ 95%
      （比权重更本质：测"表面无关性"的行为表现）
  - 判定：B（零样本 ≥40/50，3 独立训练 seeds 取最差）+ D（≥95%）
  - 3 seeds 独立训练（np.random.seed(s) + 训练布局 seed=s + 评估布局 500+s*1000
    + 扰动布局 3000+s*1000 全分离）
  - 组2 同量对照（4 表面×125ep = 500ep = 组1 2表面×250ep）——消除训练量混淆
  - 组3 内插对照（2 表面→0.65 内插，同组1 量）——消除外推/内插混淆：
    组2 vs 组3 同为内插（仅表面数不同）→ 组2 优势可归因于表面多样性
  - 局限（如实记录）：无 target network 的 DQN 自举不稳定；
    D 为单幅度行为学测试（必要不充分）
"""
import sys, os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from test_concept_transfer import ConceptEnv

# 表面外观映射（0.2/0.5/0.8 同路线 B，新增 mushroom 0.35 / berry 0.65）
APPEARANCE = {"food": 0.2, "water": 0.5, "fruit": 0.8,
              "mushroom": 0.35, "berry": 0.65}


class MLPQ:
    """小 MLP Q 网络（11→16→5，ReLU）+ 简单 DQN（replay + Adam + ε-greedy）"""
    def __init__(self, obs_dim=11, n_actions=5, lr=0.001,
                 hidden=16, buffer_size=2000, batch=32, gamma=0.95,
                 eps=0.3, train_every=4):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps
        self.batch = batch
        self.train_every = train_every
        self.steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = []          # (obs, a, r, obs_next, done)
        self.buffer_size = buffer_size

    def act(self, obs, train=True):
        if train and np.random.random() < self.eps:
            return np.random.randint(1, self.n_actions)
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=self.device)
            q = self.net(o)
            return int(q.argmax().item())

    def act_perturbed(self, obs, delta=0.1):
        """外观扰动不敏感测试：外观维度 ±delta，argmax 不变？"""
        o = np.array(obs, dtype=np.float32)
        with torch.no_grad():
            base = self.net(torch.tensor(o, dtype=torch.float32,
                                         device=self.device))
            a_base = int(base.argmax().item())
            for sign in (-1.0, 1.0):
                op = o.copy()
                op[10] = float(np.clip(op[10] + sign * delta, 0.0, 1.0))
                q = self.net(torch.tensor(op, dtype=torch.float32,
                                          device=self.device))
                if int(q.argmax().item()) != a_base:
                    return False
        return True

    def learn(self, obs, a, r, obs_next, done):
        self.buffer.append((obs, a, r, obs_next, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.steps += 1
        if self.steps % self.train_every != 0 or len(self.buffer) < self.batch:
            return
        idx = np.random.choice(len(self.buffer), size=self.batch,
                               replace=False)
        obs_b = torch.tensor(np.array([self.buffer[i][0] for i in idx]),
                             dtype=torch.float32, device=self.device)
        a_b = torch.tensor([self.buffer[i][1] for i in idx],
                           dtype=torch.long, device=self.device)
        r_b = torch.tensor([self.buffer[i][2] for i in idx],
                           dtype=torch.float32, device=self.device)
        nxt_b = torch.tensor(np.array([self.buffer[i][3] for i in idx]),
                             dtype=torch.float32, device=self.device)
        done_b = torch.tensor([self.buffer[i][4] for i in idx],
                              dtype=torch.float32, device=self.device)
        q = self.net(obs_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            qn = self.net(nxt_b).max(1).values
            target = r_b + self.gamma * qn * (1.0 - done_b)
        loss = nn.MSELoss()(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


def train_multi(modes, episodes_per=250, max_steps=40, seed=0):
    """跨多种表面训练（外观各不同——抽象"可消耗物"概念）"""
    agent = MLPQ()
    ep = 0
    total = episodes_per * len(modes)
    while ep < total:
        mode = modes[ep % len(modes)]
        env = ConceptEnv(size=6, mode=mode, seed=seed + ep)
        # ConceptEnv 用固定外观映射——注入本实验扩展外观
        env.appearance = APPEARANCE[mode]
        obs = env.observe()
        for _ in range(max_steps):
            a = agent.act(obs)
            r, done = env.step(a)
            obs_next = env.observe()
            obs_next[10] = env.appearance  # 确保 next obs 外观一致
            agent.learn(obs, a, r, obs_next, done)
            obs = obs_next
            if done:
                break
        agent.eps = max(0.05, agent.eps * 0.998)
        ep += 1
    return agent


def eval_policy(agent, mode, trials=50, max_steps=40, seed=500, train=False,
                random_baseline=False):
    reached = 0
    for t in range(trials):
        env = ConceptEnv(size=6, mode=mode, seed=seed + t * 7)
        env.appearance = APPEARANCE[mode]
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
            obs[10] = env.appearance
    return reached


def perturbation_invariance(agent, mode, trials=50, seed=3000):
    """D 判定：外观扰动 ±0.1 下 argmax 不变率（行为学表面无关性）
    局限（review nit）：单幅度 ±0.1、行为学必要不充分（权重依赖大但
    argmax 不翻转也过）——报告中注明"""
    invariant = 0
    total = 0
    for t in range(trials):
        env = ConceptEnv(size=6, mode=mode, seed=seed + t * 7)
        env.appearance = APPEARANCE[mode]
        obs = env.observe()
        for _ in range(30):
            if agent.act_perturbed(obs, delta=0.1):
                invariant += 1
            total += 1
            _, done = env.step(np.random.randint(1, 5))
            obs = env.observe()
            obs[10] = env.appearance
            if done:
                break
    return invariant / max(1, total)


def main():
    torch.manual_seed(42)
    print("=" * 60)
    print("① 概念深化 — MLP 非线性概念抽象")
    print("=" * 60)
    seeds = (42, 7, 2026)

    for group, train_modes, test_mode, ep_per, label in (
        (1, ("food", "water"), "fruit", 250,
         "组1: 2表面×250ep=500ep→0.8(外推)"),
        (2, ("food", "water", "fruit", "mushroom"), "berry", 125,
         "组2: 4表面×125ep=500ep(同量)→0.65(内插)"),
        (3, ("food", "mushroom"), "berry", 250,
         "组3: 2表面×250ep=500ep→0.65(内插对照!)"),
    ):
        print(f"\n--- {label} ---")
        rows = []
        for s in seeds:
            np.random.seed(s)
            torch.manual_seed(42 + s)
            # 训练布局随 seed 独立（review should-fix：原 seed=0 三组共享布局）
            agent = train_multi(train_modes, episodes_per=ep_per, seed=s)
            es = 500 + s * 1000
            b = eval_policy(agent, test_mode, trials=50, seed=es)
            r = eval_policy(MLPQ(), test_mode, trials=50, seed=es,
                            random_baseline=True)
            # 扰动测试布局完全分离（review should-fix：原 900 与组2训练布局重叠）
            inv = perturbation_invariance(agent, test_mode, trials=50,
                                          seed=3000 + s * 1000)
            rows.append((s, b, r, inv))
        b_min = min(x[1] for x in rows)
        b_ok = b_min >= 40
        d_min = min(x[3] for x in rows)
        d_ok = d_min >= 0.95
        detail = ", ".join(f"seed{x[0]}: {x[1]}/{x[2]} 不变率{x[3]:.2f}"
                           for x in rows)
        print(f"[B] 零样本迁移(×3seeds): {detail} "
              f"(最差≥40/50) {'OK' if b_ok else 'FAIL'}")
        print(f"[D] 扰动不敏感(×3seeds最差 {d_min:.2f}, ≥0.95) "
              f"{'OK' if d_ok else 'FAIL'}")
        ok = b_ok and d_ok
        verdict = ("通过——MLP 天然抽象（外观无关性行为成立）" if ok
                   else "未过——如实记录（MLP 依赖表面/或需正则化）")
        print(f"  判定: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
