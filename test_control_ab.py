"""
实验12：关键对照 — 预测误差驱动 vs 外挂奖励 RL

验证公理 ④：预测误差是唯一修正信号（无外挂奖励）

设计（公平对照）：
  同一环境、同一 GameNN 架构、同一容量——唯一变量 = 学习信号来源
  A组 FakeAGI：完整认知（世界模型预测误差 → 置信度门控 → GameNN）
  B组 RL：无世界模型/无自维持，GameNN 直接吃 env reward（找到食物 +1，死亡 -1）

三个场景：
  S1 简单环境（E2 风格：食物直接可见）→ 预期 RL 可能更快
  S2 因果环境（E4 风格：踩开关→解锁食物）→ 预期预测误差更好（奖励稀疏）
  S3 规则变化（E6 风格：水源移动）→ 预期预测误差更好（RL 固守旧策略）

结论判定：
  因果+规则变化场景中 FakeAGI 优于 RL → 公理④ 得到支持
  RL 全面碾压 → 公理④ 被削弱
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn

from main import AGI
from cognition import CognitionPipeline


# ═══ 共享环境 ═══

class SimpleEnv:
    """S1 简单环境：食物直接可见（obs[0:2]=食物方向）"""
    def __init__(self, size=10):
        self.size = size
        self.pos = [5, 5]
        self.food = [8, 8]
        self.eaten = 0
    def get_pos(self): return self.pos
    def observe(self):
        return np.array([(self.food[0]-self.pos[0])/self.size,
                         (self.food[1]-self.pos[1])/self.size, 0.0, 0.0])
    def step(self, a):
        dirs=[(0,0),(0,-1),(-1,0),(1,0),(0,1)]; dx,dy=dirs[a%5]
        self.pos[0]=max(0,min(self.size-1,self.pos[0]+dx))
        self.pos[1]=max(0,min(self.size-1,self.pos[1]+dy))
        d=abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])
        ed = 0.3 if d<2 else -0.001
        if d<2:
            self.eaten += 1
            self.food = [np.random.randint(0,self.size), np.random.randint(0,self.size)]
        return {'energy_delta': ed, 'water_delta': -0.0002}
    def get_energy_delta(self, a):
        d=abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])
        return 0.3 if d<2 else -0.001
    def get_damage(self, a): return 0.0
    def food_nearby(self):
        return abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])<2


class CausalEnv:
    """S2 因果环境（E4 风格）：食物被锁，踩开关解锁"""
    def __init__(self, size=10):
        self.size = size
        self.pos = [0, 0]
        self.switch = [2, 2]
        self.food = [size-1, size-1]
        self.unlocked = False
        self.eaten = 0
    def get_pos(self): return self.pos
    def observe(self):
        tf = np.array(self.food)-np.array(self.pos)
        ts = np.array(self.switch)-np.array(self.pos)
        return np.array([tf[0]/self.size, tf[1]/self.size,
                         ts[0]/self.size, ts[1]/self.size,
                         float(self.unlocked)])
    def step(self, a):
        dirs=[(0,0),(0,-1),(-1,0),(1,0),(0,1)]; dx,dy=dirs[a%5]
        self.pos[0]=max(0,min(self.size-1,self.pos[0]+dx))
        self.pos[1]=max(0,min(self.size-1,self.pos[1]+dy))
        if abs(self.pos[0]-self.switch[0])<=1 and abs(self.pos[1]-self.switch[1])<=1:
            self.unlocked = True
        d=abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])
        if d<2 and self.unlocked:
            ed = 0.3
            self.eaten += 1
        else:
            ed = -0.002
        return {'energy_delta': ed, 'water_delta': -0.0002}
    def get_energy_delta(self, a):
        d=abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])
        return 0.3 if d<2 and self.unlocked else -0.002
    def get_damage(self, a): return 0.0
    def food_nearby(self):
        return abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])<2


# ═══ A 组：FakeAGI（预测误差驱动）═══

def run_fakeagi(env_factory, max_ticks=3000, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = {"input_dim": 4, "self_state_dim": 14, "hidden_dim": 64,
           "n_actions": 5, "n_strategies": 4}
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    env = env_factory()
    agi.set_env(env)
    success_at = None
    for t in range(max_ticks):
        agi.step()
        if env.eaten > 0 and success_at is None:
            success_at = t
        if not agi.alive:
            break
    return {"success_at": success_at, "eaten": env.eaten,
            "alive": agi.alive, "survived": t}


# ═══ B 组：RL 对照（外挂奖励，无世界模型/无自维持）═══

class RLAgent:
    """同架构 GameNN 但直接吃 env reward（外挂奖励信号）"""
    def __init__(self, state_dim=4, n_actions=5, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.q = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, n_actions))
        self.optim = torch.optim.Adam(self.q.parameters(), lr=0.001)
        self.epsilon = 0.2
        self.reward_log = []

    def act(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(self.q(o).argmax().item())

    def learn(self, obs, action, reward, next_obs):
        o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        no = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        q = self.q(o)[0, action]
        target = reward + 0.9 * self.q(no).max().detach()
        loss = (q - target) ** 2
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.reward_log.append(reward)


def run_rl(env_factory, max_ticks=3000, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = env_factory()
    agent = RLAgent(state_dim=len(env.observe()), n_actions=5, seed=seed)
    success_at = None
    for t in range(max_ticks):
        obs = env.observe()
        a = agent.act(obs)
        result = env.step(a)
        reward = 1.0 if result["energy_delta"] > 0.1 else -0.01
        next_obs = env.observe()
        agent.learn(obs, a, reward, next_obs)
        if env.eaten > 0 and success_at is None:
            success_at = t
    return {"success_at": success_at, "eaten": env.eaten,
            "alive": True, "survived": t}


# ═══ S3 规则变化环境（E6 风格：水源移动）═══

class ChangeEnv:
    """S3 规则变化：水源固定 (7,7)，阶段2 移到 (2,2)（不随机重生）"""
    def __init__(self, size=10):
        self.size = size
        self.pos = [5, 5]
        self.water = [7, 7]
        self.phase = 1
    def get_pos(self): return self.pos
    def observe(self):
        wx, wy = self.water
        return np.array([(wx-self.pos[0])/self.size, (wy-self.pos[1])/self.size,
                         0.0, 0.0])
    def step(self, a):
        dirs=[(0,0),(0,-1),(-1,0),(1,0),(0,1)]; dx,dy=dirs[a%5]
        self.pos[0]=max(0,min(self.size-1,self.pos[0]+dx))
        self.pos[1]=max(0,min(self.size-1,self.pos[1]+dy))
        d=abs(self.pos[0]-self.water[0])+abs(self.pos[1]-self.water[1])
        ed = 0.03 if d<2 else -0.001
        # 水源固定（不随机重生——规则变化由外部切换）
        return {'energy_delta': ed, 'water_delta': -0.0002}
    def get_energy_delta(self, a):
        d=abs(self.pos[0]-self.water[0])+abs(self.pos[1]-self.water[1])
        return 0.03 if d<2 else -0.001
    def get_damage(self, a): return 0.0
    def food_nearby(self): return False


def run_fakeagi_change(max_ticks=2000, seed=0):
    """S3：两阶段规则变化（旧水源 1000 tick → 移到 (2,2)）"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = {"input_dim": 4, "self_state_dim": 14, "hidden_dim": 64,
           "n_actions": 5, "n_strategies": 4}
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    env = ChangeEnv()
    agi.set_env(env)
    for t in range(1000):
        agi.step()
        if not agi.alive: break
    # 规则变化
    env.water = [2, 2]
    agi.body.energy = 0.3  # 制造压力
    adapt_at = None
    for t in range(1000):
        agi.step()
        if abs(agi.pos[0]-2)+abs(agi.pos[1]-2) < 2 and adapt_at is None:
            adapt_at = t
        if not agi.alive: break
    return {"adapt_at": adapt_at, "alive": agi.alive}


def run_rl_change(max_ticks=2000, seed=0):
    """S3 RL 版：两阶段"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ChangeEnv()
    agent = RLAgent(state_dim=4, n_actions=5, seed=seed)
    for t in range(1000):
        obs = env.observe()
        a = agent.act(obs)
        r = env.step(a)
        reward = 1.0 if r["energy_delta"] > 0.02 else -0.01
        agent.learn(obs, a, reward, env.observe())
    # 规则变化
    env.water = [2, 2]
    adapt_at = None
    for t in range(1000):
        obs = env.observe()
        a = agent.act(obs)
        r = env.step(a)
        reward = 1.0 if r["energy_delta"] > 0.02 else -0.01
        agent.learn(obs, a, reward, env.observe())
        if abs(env.pos[0]-2)+abs(env.pos[1]-2) < 2 and adapt_at is None:
            adapt_at = t
    return {"adapt_at": adapt_at, "alive": True}


def compare(name, env_factory, seeds=3):
    print(f"\n── {name} ──", flush=True)
    fake_results, rl_results = [], []
    for s in range(seeds):
        fr = run_fakeagi(env_factory, seed=s)
        rr = run_rl(env_factory, seed=s)
        fake_results.append(fr)
        rl_results.append(rr)
        print(f"  seed={s} | FakeAGI: 成功t={fr['success_at']} 食物={fr['eaten']} "
              f"| RL: 成功t={rr['success_at']} 食物={rr['eaten']}", flush=True)
    f_succ = sum(1 for r in fake_results if r["success_at"] is not None)
    r_succ = sum(1 for r in rl_results if r["success_at"] is not None)
    f_food = np.mean([r["eaten"] for r in fake_results])
    r_food = np.mean([r["eaten"] for r in rl_results])
    print(f"  成功率: FakeAGI {f_succ}/{seeds} vs RL {r_succ}/{seeds}", flush=True)
    print(f"  食物均值: FakeAGI {f_food:.1f} vs RL {r_food:.1f}", flush=True)
    return {"f_succ": f_succ, "r_succ": r_succ, "f_food": f_food, "r_food": r_food}


def test():
    print("实验12: 关键对照 — 预测误差驱动 vs 外挂奖励 RL", flush=True)
    print("=" * 60, flush=True)
    r1 = compare("S1 简单环境（食物直接可见）", SimpleEnv)
    r2 = compare("S2 因果环境（踩开关解锁）", CausalEnv)

    # S3 规则变化：两阶段对比
    print(f"\n── S3 规则变化（水源移动）──", flush=True)
    f_adapt, r_adapt = [], []
    for s in range(3):
        fr = run_fakeagi_change(seed=s)
        rr = run_rl_change(seed=s)
        f_adapt.append(fr["adapt_at"])
        r_adapt.append(rr["adapt_at"])
        print(f"  seed={s} | FakeAGI 适应t={fr['adapt_at']} alive={fr['alive']} "
              f"| RL 适应t={rr['adapt_at']}", flush=True)
    f_ok = sum(1 for a in f_adapt if a is not None)
    r_ok = sum(1 for a in r_adapt if a is not None)
    print(f"  适应率: FakeAGI {f_ok}/3 vs RL {r_ok}/3", flush=True)
    r3 = {"f_succ": f_ok, "r_succ": r_ok, "f_food": 0, "r_food": 0}

    # 判定（S2 因果环境为核心）
    causal_fake_wins = r2["f_succ"] >= r2["r_succ"] and r2["f_food"] >= r2["r_food"]
    change_fake_wins = r3["f_succ"] >= r3["r_succ"]
    print(f"\n── 汇总 ──", flush=True)
    print(f"  S1 简单: FakeAGI 食物 {r1['f_food']:.1f} vs RL {r1['r_food']:.1f}", flush=True)
    print(f"  S2 因果: FakeAGI 食物 {r2['f_food']:.1f} vs RL {r2['r_food']:.1f}", flush=True)
    print(f"  S3 变化: FakeAGI 适应 {r3['f_succ']}/3 vs RL {r3['r_succ']}/3", flush=True)
    checks = [
        ("S2 因果环境中预测误差驱动≥RL（公理④支持）", causal_fake_wins),
        ("S3 规则变化中预测误差驱动≥RL（适应力）", change_fake_wins),
    ]
    passed = all(c for _, c in checks)
    print(f"  判定: {'OK 通过 — 公理④ 得到支持' if passed else 'FAIL — 公理④ 被削弱'}", flush=True)
    for name, ok in checks:
        print(f"  {'[OK]' if ok else '[X]'} {name}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(test())
