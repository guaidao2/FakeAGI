"""
实验11：语言意图 — 系统主动说话才能活

核心命题：L4 意图 = 系统主动产生词，且该词服务于生存目标。
环境不再主动提供语言——系统必须"说"才能获得信息：
  - 系统说 "food" → 环境回应食物方向词（"east" 等）
  - 系统说 "water" → 环境回应水源方向词
  - 不说 → 无任何方向信息 → 盲走

意图的操作化标准（防硬编码）：
  1. 生存价值：会说"food"的个体存活/食物显著优于不会说的（对照组）
  2. 需求-说话因果：说"food"概率与饥饿水平正相关（energy 低→多说，
     不是固定频率/随机）——说话由内部状态驱动
  3. 说话→听→行动链：说"food"→环境回方向词→系统朝词方向走
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from main import AGI
from cognition import CognitionPipeline

VOCAB = ["food", "water", "danger", "safe", "east", "west", "north", "south"]
DIR_MAP = {"east": 3, "west": 2, "north": 1, "south": 4}


class LangIntentEnv:
    """意图环境：系统必须先说话，环境才响应方向"""
    def __init__(self, size=16, can_speak=True):
        self.size = size
        self.can_speak = can_speak  # 说话通道（对照=关闭）
        self.pos = [8, 8]
        self.water_pos = [3, 3]
        self.food_pos = self._random_food()
        self.food_eaten = 0
        self.steps = 0
        self.response_words = []   # 环境对系统说话的响应
        self.spoken_count = 0      # 系统说过话的次数
        self.last_spoken_word = None

    def _random_food(self):
        while True:
            p = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            d = abs(p[0]-self.pos[0])+abs(p[1]-self.pos[1])
            if d > self.size // 2:
                return p

    def get_pos(self): return self.pos

    def observe(self):
        """只有水源方向（食物方向隐藏）"""
        wx, wy = self.water_pos
        return np.array([(wx-self.pos[0])/self.size, (wy-self.pos[1])/self.size])

    def respond(self, word: str):
        """环境响应系统的话：说 food → 给食物方向词（持续有效直到重新请求）"""
        self.spoken_count += 1
        self.last_spoken_word = word
        if not self.can_speak:
            return []
        if word == "food":
            dx = self.food_pos[0] - self.pos[0]
            dy = self.food_pos[1] - self.pos[1]
            if abs(dx) >= abs(dy):
                d = "east" if dx > 0 else ("west" if dx < 0 else ("south" if dy > 0 else "north"))
            else:
                d = "south" if dy > 0 else "north"
            self.response_words = ["food", d]
            return self.response_words
        if word == "water":
            dx = self.water_pos[0] - self.pos[0]
            dy = self.water_pos[1] - self.pos[1]
            if abs(dx) >= abs(dy):
                d = "east" if dx > 0 else ("west" if dx < 0 else ("south" if dy > 0 else "north"))
            else:
                d = "south" if dy > 0 else "north"
            self.response_words = ["water", d]
            return self.response_words
        return []

    def step(self, a):
        self.steps += 1
        dirs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
        dx, dy = dirs[a % 5]
        self.pos[0] = max(0, min(self.size-1, self.pos[0]+dx))
        self.pos[1] = max(0, min(self.size-1, self.pos[1]+dy))
        d_food = abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1])
        d_water = abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1])
        ed = 0.3 if d_food < 2 else -0.002
        wd = 0.05 if d_food < 2 else (0.15 if d_water < 2 else -0.0005)
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


def run_episode(can_speak, max_ticks=2000, seed=0):
    """跑一个个体：会说 vs 不会说"""
    np.random.seed(seed)
    import torch as T
    T.manual_seed(seed)
    cfg = {
        "input_dim": 2, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4,
        "language": True, "language_vocab": VOCAB,
    }
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    env = LangIntentEnv(can_speak=can_speak)
    agi.set_env(env)

    # 说话-饥饿样本（意图相关性统计）
    speak_energy = []   # (是否说了 food, 当时的 energy)
    food_found = 0
    died_at = None
    for t in range(max_ticks):
        # 系统主动说话：需求驱动（饥饿→说 food，口渴→说 water）
        spoke = agi.speak() if can_speak else False
        if spoke:
            resp = env.respond(agi.last_spoken_word)
            agi.cognition.language_tokens = resp if resp else None
            # 说话结果更新信任：有响应（语言有用）→ 强化
            if resp:
                agi._update_speak_trust(True)
            else:
                agi._update_speak_trust(False)
            if agi.last_spoken_word == "food":
                speak_energy.append((1, agi.body.energy))
        else:
            if can_speak:
                speak_energy.append((0, agi.body.energy))
            # 不说话的 tick：沿用上次响应（方向持续有效）
            agi.cognition.language_tokens = env.response_words if env.response_words else None
        agi.step()
        food_found = env.food_eaten
        if not agi.alive:
            died_at = t
            break
    return {
        "died_at": died_at,
        "food_found": food_found,
        "alive": agi.alive,
        "spoken": env.spoken_count,
        "speak_energy": speak_energy,
        "agi": agi,
    }


def intent_correlation(samples):
    """需求-说话因果：energy 低时是否更常说 food"""
    if len(samples) < 50:
        return 0.0, 0
    spoke = np.array([s[0] for s in samples])
    energy = np.array([s[1] for s in samples])
    if spoke.sum() < 5 or (len(spoke) - spoke.sum()) < 5:
        return 0.0, int(spoke.sum())
    # 能量分两半：低能段 vs 高能段的说 food 比例
    med = np.median(energy)
    low = spoke[energy < med]
    high = spoke[energy >= med]
    p_low = low.mean() if len(low) else 0
    p_high = high.mean() if len(high) else 0
    return p_low - p_high, int(spoke.sum())


def test():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    print("实验11: 语言意图 — 系统主动说话才能活", flush=True)

    # ─── 对照组（不会说话）───
    print("\n── 对照组（不会说话：无任何方向信息）──", flush=True)
    ctrl = []
    for seed in range(5):
        r = run_episode(False, seed=seed)
        ctrl.append(r)
        print(f"  seed={seed}: died={r['died_at']} food={r['food_found']} "
              f"alive={r['alive']} spoken={r['spoken']}", flush=True)
    ctrl_food = np.mean([r['food_found'] for r in ctrl])
    ctrl_survive = np.mean([(r['died_at'] if r['died_at'] is not None else 2000)
                            for r in ctrl])

    # ─── 实验组（会说话）───
    print("\n── 实验组（会说话：说 food → 环境给方向）──", flush=True)
    exp = []
    for seed in range(5):
        r = run_episode(True, seed=seed)
        exp.append(r)
        print(f"  seed={seed}: died={r['died_at']} food={r['food_found']} "
              f"alive={r['alive']} spoken={r['spoken']}", flush=True)
    exp_food = np.mean([r['food_found'] for r in exp])
    exp_survive = np.mean([(r['died_at'] if r['died_at'] is not None else 2000)
                           for r in exp])

    # ─── 意图相关性：说 food 与饥饿的因果 ───
    all_samples = []
    for r in exp:
        all_samples.extend(r["speak_energy"])
    intent_gap, n_spoke = intent_correlation(all_samples)
    print(f"\n  意图: 说food概率差(低能-高能) = {intent_gap:.3f} "
          f"(总说话 {n_spoke} 次)", flush=True)

    # ─── 判定 ───
    lang_value = exp_food > ctrl_food * 1.5 and exp_survive > ctrl_survive * 1.2
    has_intent = intent_gap > 0.05 and n_spoke > 20
    checks = [
        ("语言有生存价值（会说 vs 不会说）", lang_value),
        ("意图（饥饿驱动说话）", has_intent),
    ]
    passed = all(c for _, c in checks)
    print(f"\n── 汇总 ──", flush=True)
    print(f"  对照组: 存活 {ctrl_survive:.0f} tick, 食物 {ctrl_food:.1f}", flush=True)
    print(f"  实验组: 存活 {exp_survive:.0f} tick, 食物 {exp_food:.1f}", flush=True)
    print(f"  判定: {'OK 通过' if passed else 'FAIL 未通过'}", flush=True)
    for name, ok in checks:
        print(f"  {'[OK]' if ok else '[X]'} {name}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(test())
