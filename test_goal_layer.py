"""
实验13：目标层消融 — 目标表征是否修复 S2（信息隐藏环境）

假设（DESIGN_GOALS.md）：S2 失败根因是目标表征缺失（落差机制未激活），
非公理④ 失效。补入目标层（落差驱动信息寻求）后应能定向搜索找到食物。

组别：
  G1 基线：当前 FakeAGI（无目标层）——预期稳定饿死（215 tick）
  G2 目标层：加 GoalState（能量目标 + 信息寻求探索调制）——预期存活+找到食物
  G3 RL+解锁奖励：RL 加中间奖励（分离 extrinsic 贡献）——参考
  G4 去反射+目标层：G2 但禁用反射锚定——排除"反射锚定"混淆

判定：
  G2 存活且食物>0 → 支持"S2 是误测，公理④ 精炼成立"
  G2 仍饿死 → 更深的机制缺陷
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn

from main import AGI
from cognition import CognitionPipeline
from core.goals import GoalState, Goal


# ═══ S2 环境（信息隐藏：obs 只有开关方向）═══

class CausalEnv:
    def __init__(self, size=10):
        self.size = size
        self.pos = [5, 5]
        self.switch = [8, 8]
        self.food = [0, 0]
        self.unlocked = False
        self.eaten = 0
    def get_pos(self): return self.pos
    def observe(self):
        ts = np.array(self.switch)-np.array(self.pos)
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
            # 吃到后食物重置（防止停留虚高计数）
            self.food = [np.random.randint(0,self.size), np.random.randint(0,self.size)]
            self.unlocked = False
        else:
            ed = -0.002
        return {'energy_delta': ed, 'water_delta': -0.0002}
    def get_energy_delta(self, a):
        d=abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])
        return 0.3 if d<2 and self.unlocked else -0.002
    def get_damage(self, a): return 0.0
    def food_nearby(self):
        return abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])<2


# ═══ G1/G2/G4：FakeAGI 系列 ═══

def run_fakeagi(use_goal_layer=False, disable_reflex=False,
                const_explore=None, max_ticks=3000, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = {"input_dim": 4, "self_state_dim": 14, "hidden_dim": 64,
           "n_actions": 5, "n_strategies": 4}
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    env = CausalEnv()
    agi.set_env(env)
    if use_goal_layer:
        # 注入目标层并启用调制
        agi.goal_state = GoalState()
        agi.goal_state.register(Goal(
            "energy_maintenance", target_value=0.8,
            current_fn=lambda: agi.body.energy, weight=2.0))
        agi.goal_state.register(Goal(
            "water_maintenance", target_value=0.7,
            current_fn=lambda: agi.body.water, weight=1.5))
        agi._goal_enabled = True
    else:
        # 禁用目标层（基线隔离）
        agi.goal_state = GoalState()
        agi._goal_off = True
    if disable_reflex:
        agi._disable_reflex = True
    if const_explore is not None:
        agi._const_explore = const_explore  # 恒定探索率对照
    success_at = None
    for t in range(max_ticks):
        agi.step()
        if env.eaten > 0 and success_at is None:
            success_at = t
        if not agi.alive:
            break
    return {"success_at": success_at, "eaten": env.eaten,
            "alive": agi.alive, "survived": t}


# ═══ G3：RL + 解锁中间奖励 ═══

class RLAgent:
    def __init__(self, state_dim=4, n_actions=5, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.q = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, n_actions))
        self.target = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, n_actions))
        self.target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=0.001)
        self.epsilon = 0.2
        self.replay = []
        self.steps = 0

    def act(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(self.q(o).argmax().item())

    def learn(self, obs, action, reward, next_obs, done=False):
        self.replay.append((obs, action, reward, next_obs, done))
        if len(self.replay) > 20000:
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
            target = r + 0.9 * self.target(no).max(1).values * (1 - d)
        loss = ((q - target) ** 2).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.steps += 1
        if self.steps % 200 == 0:
            self.target.load_state_dict(self.q.state_dict())


def run_rl(unlock_bonus=False, max_ticks=3000, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = CausalEnv()
    agent = RLAgent(state_dim=3, n_actions=5, seed=seed)
    energy = 1.0
    prev_unlocked = False
    success_at = None
    for t in range(max_ticks):
        obs = env.observe()
        a = agent.act(obs)
        result = env.step(a)
        ed = result["energy_delta"]
        if abs(ed) > 0.001:
            energy = max(0.0, min(2.0, energy + ed))
        reward = 1.0 if ed > 0.1 else -0.01
        if unlock_bonus and env.unlocked and not prev_unlocked:
            reward += 0.5   # 解锁中间奖励
        prev_unlocked = env.unlocked
        done = energy <= 0.0
        if done:
            break
        agent.learn(obs, a, reward, env.observe(), done)
        if env.eaten > 0 and success_at is None:
            success_at = t
    return {"success_at": success_at, "eaten": env.eaten,
            "alive": not done, "survived": t}


def run_group(name, fn, seeds=3):
    print(f"\n── {name} ──", flush=True)
    results = []
    for s in range(seeds):
        r = fn(seed=s)
        results.append(r)
        print(f"  seed={s}: 成功t={r['success_at']} 食物={r['eaten']} "
              f"存活={r['survived']} alive={r['alive']}", flush=True)
    succ = sum(1 for r in results if r["success_at"] is not None)
    food = np.mean([r["eaten"] for r in results])
    print(f"  成功率 {succ}/{seeds}, 食物均值 {food:.1f}", flush=True)
    return {"succ": succ, "food": food, "results": results}


def test():
    print("实验13: 目标层消融 — 目标表征是否修复 S2", flush=True)
    print("=" * 60, flush=True)
    g1 = run_group("G1 基线（无目标层）", lambda seed=0: run_fakeagi(False, seed=seed))
    g2 = run_group("G2 目标层（落差驱动）", lambda seed=0: run_fakeagi(True, seed=seed))
    g5 = run_group("G5 恒定0.8探索+无目标层", lambda seed=0: run_fakeagi(False, const_explore=0.8, seed=seed))
    g3 = run_group("G3 RL+解锁中间奖励", lambda seed=0: run_rl(True, seed=seed))
    g4 = run_group("G4 去反射+目标层", lambda seed=0: run_fakeagi(True, True, seed=seed))

    print(f"\n── 汇总 ──", flush=True)
    print(f"  G1 基线: 存活 {np.mean([r['survived'] for r in g1['results']]):.0f} "
          f"食物 {g1['food']:.1f}", flush=True)
    print(f"  G2 目标层: 存活 {np.mean([r['survived'] for r in g2['results']]):.0f} "
          f"食物 {g2['food']:.1f}", flush=True)
    print(f"  G5 恒定探索: 存活 {np.mean([r['survived'] for r in g5['results']]):.0f} "
          f"食物 {g5['food']:.1f}", flush=True)
    print(f"  G3 RL+奖励: 食物 {g3['food']:.1f}", flush=True)
    print(f"  G4 去反射+目标: 存活 {np.mean([r['survived'] for r in g4['results']]):.0f} "
          f"食物 {g4['food']:.1f}", flush=True)

    # 判定（≥2/3 成功 + 目标层优于恒定探索对照——分离目标表征 vs 高探索）
    g2_fixed = g2["succ"] >= 2 and g2["food"] > g5["food"] * 1.2
    verdict = ("是 — 目标层优于恒定探索（支持[目标表征是有效机制]）" if g2_fixed
               else "否 — 高探索率即可解释，目标层无增量")
    print(f"\n  判定: G2 目标层修复 S2（vs G5 恒定探索）: {verdict}", flush=True)
    print(f"  说明: 此为配置有效性证据，公理④ 精炼需进一步消融", flush=True)
    return 0 if g2_fixed else 1


if __name__ == "__main__":
    sys.exit(test())
