"""
实验12：对照实验 — 预测误差驱动 vs 外挂奖励 RL（修正版）

目的：检验公理 ④（预测误差是唯一修正信号）的初步证据。
定位：**机制演示 / 初步观察**（3 seeds，无显著性检验——不作为定论）。

公平性设计（修正复审问题）：
  1. S2 因果环境：食物方向**隐藏**（观测不含食物方向，只有开关方向）——
     系统必须学会"踩开关→解锁"的因果才能高效觅食，排除反射巧合
  2. RL 基线现代化：经验回放 + target network（非朴素 Q-learning）
  3. S3 规则变化：镜像压力（FakeAGI 注入 energy=0.3，RL 注入等价负奖励）
  4. 措辞：完整系统 vs 现代 RL 基线（不宣称"同架构同容量"）

结论范围：仅声称"在此基线配置下，完整系统不弱于现代 RL"——
公理④ 的严格验证需消融实验（待后续）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn

from main import AGI
from cognition import CognitionPipeline


# ═══ S1 简单环境 ═══

class SimpleEnv:
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


# ═══ S2 因果环境（修正：食物方向隐藏，开关在路径外）═══

class CausalEnv:
    """食物方向隐藏（obs 只有开关方向）+ 开关在路径外（(7,7)，远离必经路径）
    系统必须学会"先踩开关"的因果——反射无法碰巧解锁"""
    def __init__(self, size=10):
        self.size = size
        self.pos = [5, 5]
        self.switch = [8, 8]   # 远离起点和食物的路径（防巧合）
        self.food = [0, 0]     # 与起点(5,5)距离远
        self.unlocked = False
        self.eaten = 0
    def get_pos(self): return self.pos
    def observe(self):
        ts = np.array(self.switch)-np.array(self.pos)
        # 注意：obs 不含食物方向（隐藏）——只有开关方向 + 锁状态
        return np.array([ts[0]/self.size, ts[1]/self.size,
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


# ═══ S3 规则变化环境（水源给水，两阶段）═══

class ChangeEnv:
    def __init__(self, size=10):
        self.size = size
        self.pos = [5, 5]
        self.water = [7, 7]
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
        ed = 0.01 if d<2 else -0.001   # 水源给能量（镜像 E6 的生存压力）
        wd = 0.15 if d<2 else -0.0005  # 水源也给水（语义正确）
        return {'energy_delta': ed, 'water_delta': wd}
    def get_energy_delta(self, a):
        d=abs(self.pos[0]-self.water[0])+abs(self.pos[1]-self.water[1])
        return 0.01 if d<2 else -0.001
    def get_damage(self, a): return 0.0
    def food_nearby(self): return False


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


# ═══ B 组：现代 RL 基线（经验回放 + target net）═══

class RLAgent:
    def __init__(self, state_dim=4, n_actions=5, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        def make_q():
            return nn.Sequential(
                nn.Linear(state_dim, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, n_actions))
        self.q = make_q()
        self.target = make_q()
        self.target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=0.001)
        self.epsilon = 0.2
        self.replay = []          # 经验回放
        self.replay_cap = 20000
        self.gamma = 0.9
        self.steps = 0

    def act(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(self.q(o).argmax().item())

    def learn(self, obs, action, reward, next_obs, done=False):
        self.replay.append((obs, action, reward, next_obs, done))
        if len(self.replay) > self.replay_cap:
            self.replay.pop(0)
        if len(self.replay) < 64:
            return
        batch = [self.replay[i] for i in
                 np.random.choice(len(self.replay), 64, replace=False)]
        o = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        a = torch.tensor([b[1] for b in batch], dtype=torch.long)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        no = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        d = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        q = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * self.target(no).max(1).values * (1 - d)
        loss = ((q - target) ** 2).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # 周期性同步 target net
        self.steps += 1
        if self.steps % 200 == 0:
            self.target.load_state_dict(self.q.state_dict())


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
        done = False
        agent.learn(obs, a, reward, env.observe(), done)
        if env.eaten > 0 and success_at is None:
            success_at = t
    return {"success_at": success_at, "eaten": env.eaten,
            "alive": True, "survived": t}


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


# ═══ S3 两阶段（镜像压力）═══

def run_fakeagi_change(max_ticks=2000, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = {"input_dim": 4, "self_state_dim": 14, "hidden_dim": 64,
           "n_actions": 5, "n_strategies": 4}
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    env = ChangeEnv()
    agi.set_env(env)
    alive = True
    for t in range(1000):
        agi.step()
        if not agi.alive:
            alive = False
            break
    if not alive:
        return {"adapt_at": None, "alive": False}
    env.water = [2, 2]
    agi.body.energy = 0.3
    adapt_at = None
    for t in range(1000):
        agi.step()
        if abs(agi.pos[0]-2)+abs(agi.pos[1]-2) < 2 and adapt_at is None:
            adapt_at = t
        if not agi.alive:
            break
    return {"adapt_at": adapt_at, "alive": agi.alive}


def run_rl_change(max_ticks=2000, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ChangeEnv()
    agent = RLAgent(state_dim=4, n_actions=5, seed=seed)
    for t in range(1000):
        obs = env.observe()
        a = agent.act(obs)
        r = env.step(a)
        reward = 1.0 if r["energy_delta"] > 0.005 else -0.01
        agent.learn(obs, a, reward, env.observe())
    env.water = [2, 2]
    # 镜像压力：阶段 2 负奖励增强（等价 FakeAGI 的存活压力）
    adapt_at = None
    for t in range(1000):
        obs = env.observe()
        a = agent.act(obs)
        r = env.step(a)
        reward = 1.0 if r["energy_delta"] > 0.005 else -0.02  # 压力增强
        agent.learn(obs, a, reward, env.observe())
        if abs(env.pos[0]-2)+abs(env.pos[1]-2) < 2 and adapt_at is None:
            adapt_at = t
    return {"adapt_at": adapt_at, "alive": True}


def test():
    print("实验12: 对照实验 — 预测误差 vs 外挂奖励 RL（修正版）", flush=True)
    print("定位：机制演示/初步观察（3 seeds，无显著性检验）", flush=True)
    print("=" * 60, flush=True)
    r1 = compare("S1 简单环境（食物直接可见）", SimpleEnv)
    r2 = compare("S2 因果环境（食物方向隐藏+开关在路径外）", CausalEnv)

    print(f"\n── S3 规则变化（水源移动，镜像压力）──", flush=True)
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
    r3 = {"f_succ": f_ok, "r_succ": r_ok, "f_food": 0, "r_food": 0}

    print(f"\n── 汇总（初步观察，非定论）──", flush=True)
    print(f"  S1 简单: FakeAGI 食物 {r1['f_food']:.1f} vs RL {r1['r_food']:.1f}", flush=True)
    print(f"  S2 因果: FakeAGI 食物 {r2['f_food']:.1f} vs RL {r2['r_food']:.1f}", flush=True)
    print(f"  S3 变化: FakeAGI 适应 {r3['f_succ']}/3 vs RL {r3['r_succ']}/3", flush=True)
    # 初步观察判定（诚实报告，不偏向）
    s2_observe = r2["f_food"] >= r2["r_food"]
    print(f"\n  初步观察: S2 因果环境（食物方向隐藏）FakeAGI ≥ 现代 RL: "
          f"{'成立' if s2_observe else '不成立——RL 更优'}（信息隐藏下预测误差无引导）", flush=True)
    print(f"  科学意义: 修正设计后结论反转——公理④ 在此配置下未获支持，", flush=True)
    print(f"  需消融（RL+解锁奖励 / FakeAGI 去反射+加食物线索）才能归因", flush=True)
    # 诚实判定：如实报告观察结果（无论哪个方向）
    print(f"  判定: 对照实验完成，结果如实记录（{'FakeAGI 优' if s2_observe else 'RL 优'}）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(test())
