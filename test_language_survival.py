"""
实验10：语言生存压力 — 只有懂语言才能活

核心命题：当语言成为生存的必要信息通道时，系统才会真正理解语义。

环境设计（制造语言-生存耦合）：
  - 观测：只有水源方向（2D）——食物方向对系统**不可见**（感知盲区）
  - 语言：环境每 tick 说 "food <direction>"（食物方向指示，唯一线索）
  - 规则：朝词指示方向走 → 找到食物（+0.3 能量）；听不懂 → 盲走 → 饿死
  - 水源只补水不给能量（能量只能从食物获得）

对照组 vs 实验组：
  - 实验组：语言开启（能听到方向词）
  - 对照组：语言关闭（听不到词，只能盲走）
  判定：实验组存活/找到食物显著优于对照组 = 语言有生存价值 = 语义被理解
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from main import AGI
from cognition import CognitionPipeline

VOCAB = ["food", "water", "danger", "safe", "east", "west", "north", "south"]

DIR_WORDS = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}
DIR_VEC = {v: k for k, v in DIR_WORDS.items()}


class LangSurvivalEnv:
    """语言生存环境：食物方向只通过语言透露"""
    def __init__(self, size=16, use_language=True):
        self.size = size
        self.use_language = use_language
        self.pos = [8, 8]
        self.water_pos = [3, 3]
        self.food_pos = self._random_food()
        self.food_eaten = 0
        self.steps = 0

    def _random_food(self):
        """食物随机到远处（距离起点 > size/2），让乱走难以撞上"""
        while True:
            p = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            d = abs(p[0]-self.pos[0])+abs(p[1]-self.pos[1])
            if d > self.size // 2:
                return p

    def get_pos(self): return self.pos

    def observe(self):
        """只有水源方向（食物方向对系统隐藏）"""
        wx, wy = self.water_pos
        return np.array([(wx-self.pos[0])/self.size, (wy-self.pos[1])/self.size])

    def get_language(self):
        """语言：食物方向指示（唯一食物线索）"""
        if not self.use_language:
            return []
        dx = self.food_pos[0] - self.pos[0]
        dy = self.food_pos[1] - self.pos[1]
        # 主方向词
        if abs(dx) >= abs(dy):
            w = "east" if dx > 0 else ("west" if dx < 0 else ("south" if dy > 0 else "north"))
        else:
            w = "south" if dy > 0 else "north"
        return ["food", w]

    def step(self, a):
        self.steps += 1
        dirs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
        dx, dy = dirs[a % 5]
        self.pos[0] = max(0, min(self.size-1, self.pos[0]+dx))
        self.pos[1] = max(0, min(self.size-1, self.pos[1]+dy))
        # 食物（唯一能量来源）
        d_food = abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1])
        # 水源（纯补水，不给能量——能量只能从食物获得）
        d_water = abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1])
        ed = 0.3 if d_food < 2 else -0.002  # 代谢更快，逼迫找食物
        wd = 0.05 if d_food < 2 else (0.15 if d_water < 2 else -0.0005)
        # 吃到食物 → 食物重新随机
        if d_food < 2:
            self.food_eaten += 1
            self.food_pos = self._random_food()
        return {'energy_delta': ed, 'water_delta': wd}

    def get_energy_delta(self, a):
        d_food = abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1])
        return 0.3 if d_food < 2 else -0.002

    def get_damage(self, a):
        return 0.0

    def food_nearby(self):
        return abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1]) < 2


def run_episode(use_language, max_ticks=2000, seed=0):
    """跑一个个体：语言开/关"""
    np.random.seed(seed)
    import torch as T
    T.manual_seed(seed)
    cfg = {
        "input_dim": 2, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4,
        "language": use_language, "language_vocab": VOCAB,
    }
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    env = LangSurvivalEnv(use_language=use_language)
    agi.set_env(env)

    food_found = 0
    died_at = None
    for t in range(max_ticks):
        env.words = env.get_language()
        agi.cognition.language_tokens = env.words if use_language else None
        agi.step()
        food_found = env.food_eaten
        if not agi.alive:
            died_at = t
            break
    return {
        "died_at": died_at,
        "food_found": food_found,
        "alive": agi.alive,
        "energy": agi.body.energy,
    }


def comprehension_test(agi, env, lang, n=200):
    """词→食物方向理解：语言向量回归食物方向 vs 随机基线"""
    Xw, Yw = [], []
    Xr, Yr = [], []
    organ = lang.organ
    with torch.no_grad():
        for _ in range(n):
            pos = np.random.randint(0, env.size, 2)
            env.pos = pos.tolist()
            env.food_pos = env._random_food()
            words = env.get_language()
            # 食物方向（真实目标）
            target = np.array([
                (env.food_pos[0]-env.pos[0])/env.size,
                (env.food_pos[1]-env.pos[1])/env.size])
            tok_ids = lang.tokenize(words)
            if tok_ids:
                lv = organ.encode(tok_ids).detach().cpu().numpy().flatten()
                Xw.append(lv); Yw.append(target)
            Xr.append(np.random.randn(organ.output_dim)); Yr.append(target)
    def train_lin(X, Y):
        if len(X) < 10: return 1e9
        Xt = np.hstack([np.array(X), np.ones((len(X), 1))])
        w = np.linalg.lstsq(Xt, np.array(Y), rcond=None)[0]
        pred = Xt @ w
        return float(np.mean(np.linalg.norm(pred - np.array(Y), axis=1)))
    ew = train_lin(Xw, Yw)
    er = train_lin(Xr, Yr)
    return ew, er


def test():
    print("实验10: 语言生存压力 — 只有懂语言才能活", flush=True)

    # ─── 对照组（语言关闭）───
    print("\n── 对照组（语言关闭：食物方向不可见）──", flush=True)
    ctrl_results = []
    for seed in range(3):
        r = run_episode(False, seed=seed)
        ctrl_results.append(r)
        print(f"  seed={seed}: died_at={r['died_at']} food={r['food_found']} "
              f"alive={r['alive']} energy={r['energy']:.2f}", flush=True)
    ctrl_food = np.mean([r['food_found'] for r in ctrl_results])
    ctrl_survive = np.mean([(r['died_at'] or 2000) for r in ctrl_results])

    # ─── 实验组（语言开启）───
    print("\n── 实验组（语言开启：食物方向由词透露）──", flush=True)
    exp_results = []
    for seed in range(3):
        r = run_episode(True, seed=seed)
        exp_results.append(r)
        print(f"  seed={seed}: died_at={r['died_at']} food={r['food_found']} "
              f"alive={r['alive']} energy={r['energy']:.2f}", flush=True)
    exp_food = np.mean([r['food_found'] for r in exp_results])
    exp_survive = np.mean([(r['died_at'] or 2000) for r in exp_results])

    # ─── 理解测试（实验组语言器官）───
    # 重新跑一个实验组个体并测理解
    np.random.seed(0)
    import torch as T
    T.manual_seed(0)
    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 2, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4,
        "language": True, "language_vocab": VOCAB,
    }))
    env = LangSurvivalEnv(use_language=True)
    agi.set_env(env)
    lang = agi.cognition.language
    # 先积累语言-状态样本做接地训练
    word_samples, state_samples = [], []
    for t in range(800):
        env.words = env.get_language()
        agi.cognition.language_tokens = env.words
        agi.step()
        if env.words and t % 4 == 0:
            word_samples.append(env.words)
            state_samples.append(env.observe())
        if not agi.alive:
            break
    if word_samples:
        lang.train_grounding(word_samples, np.array(state_samples), epochs=20)
    ew, er = comprehension_test(agi, env, lang)
    print(f"\n  理解: 词→食物方向误差 {ew:.3f} vs 随机基线 {er:.3f}", flush=True)
    understood = ew < er * 0.9

    # ─── 判定 ───
    print(f"\n── 汇总 ──", flush=True)
    print(f"  对照组: 平均存活 {ctrl_survive:.0f} tick, 找到食物 {ctrl_food:.1f} 次", flush=True)
    print(f"  实验组: 平均存活 {exp_survive:.0f} tick, 找到食物 {exp_food:.1f} 次", flush=True)
    lang_value = exp_food > ctrl_food * 1.5 and exp_survive > ctrl_survive * 1.2
    checks = [
        ("语言有生存价值（实验组显著优）", lang_value),
        ("理解（词→食物方向）", understood),
    ]
    passed = all(c for _, c in checks)
    print(f"\n判定: {'OK 通过' if passed else 'FAIL 未通过'}", flush=True)
    for name, ok in checks:
        print(f"  {'[OK]' if ok else '[X]'} {name}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(test())
