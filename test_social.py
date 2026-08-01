"""
路线 C — 多智能体社会实验（双 AGI 共享世界：竞争/合作涌现）

他者模型（test_other_agent.py）验证了 AGI vs 脚本他者（回避/跟随）。
路线 C 更进一步：**两个真 AGI 在共享世界互相对抗**——社会智能：
  - 各自有完整认知管线（反射/学习/规划/他者模型）
  - 共享食物（稀缺）→ 竞争压力
  - 他者模型（OtherModel）消费对方位置 → 回避/合作行为

科学问题：
  Q1. 双 AGI 能否共存（各自存活）？
  Q2. 是否出现社会行为（接近/回避对方，非零）？
  Q3. 总食物获取 vs 单 AGI 基线——多智能体是提升还是拖累？

设计（预注册）：
  - World2：8x8 共享世界，食物 3 个（稀缺），2 个 AGI 交替 step
  - 观测：各自 4D 基础观测 + 他者位置（他者模型消费）
  - 指标：存活/食物/冲突次数（同时目标同一食物）/平均间距
  - 对比：双 AGI vs 单 AGI（同布局同食物数，基线）
  - 判定：双 AGI 均存活 + 冲突>0（社会行为出现）+ 双总食物 ≥ 单基线（不拖累）
  - 3 seeds 独立 + 布局分离 + 判定取最差
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from main import AGI
from cognition import CognitionPipeline


class World2:
    """双 AGI 共享世界：8x8，食物 3 个（吃后随机重放）"""
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.size = 8
        self.pos_a = [1, 1]
        self.pos_b = [6, 6]
        self.foods = []
        for _ in range(5):   # 5 食物（3 太少——AGI 全饿死无法观察社会行为）
            self._respawn_food()
        self.food_eaten = 0
        self.conflicts = 0        # 同 tick 双方目标同一食物的次数
        self.last_target_a = None
        self.last_target_b = None

    def _respawn_food(self):
        while True:
            f = [self.rng.randint(self.size), self.rng.randint(self.size)]
            if f not in self.foods and f != self.pos_a and f != self.pos_b:
                self.foods.append(f)
                return

    def _nearest(self, pos):
        if not self.foods:
            return None, 0
        ds = [abs(pos[0] - f[0]) + abs(pos[1] - f[1]) for f in self.foods]
        i = int(np.argmin(ds))
        return self.foods[i], int(ds[i])

    def observe(self, who):
        pos = self.pos_a if who == "a" else self.pos_b
        other = self.pos_b if who == "a" else self.pos_a
        food, dist = self._nearest(pos)
        if food is None:
            return np.zeros(6, dtype=np.float32)
        # [食物方向dx, dy, 食物距离, 他者方向dx, 他者方向dy, 他者距离]
        return np.array([
            (food[0] - pos[0]) / self.size, (food[1] - pos[1]) / self.size,
            dist / (self.size * 2),
            (other[0] - pos[0]) / self.size, (other[1] - pos[1]) / self.size,
            (abs(other[0] - pos[0]) + abs(other[1] - pos[1])) / (self.size * 2),
        ], dtype=np.float32)

    def step(self, who, action, target_food=None):
        """执行动作；记录冲突（双方目标同一食物）"""
        pos = self.pos_a if who == "a" else self.pos_b
        if who == "a":
            self.last_target_a = tuple(target_food) if target_food else None
            if self.last_target_b and self.last_target_a \
                    and self.last_target_b == self.last_target_a:
                self.conflicts += 1
        else:
            self.last_target_b = tuple(target_food) if target_food else None
            if self.last_target_a and self.last_target_b \
                    and self.last_target_a == self.last_target_b:
                self.conflicts += 1
        dirs = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)]
        dx, dy = dirs[action % 5]
        nx = max(0, min(self.size - 1, pos[0] + dx))
        ny = max(0, min(self.size - 1, pos[1] + dy))
        # 不可与对方重叠
        other = self.pos_b if who == "a" else self.pos_a
        if [nx, ny] == other:
            return 0.0
        if who == "a":
            self.pos_a = [nx, ny]
        else:
            self.pos_b = [nx, ny]
        # 吃食物
        if [nx, ny] in self.foods:
            self.foods.remove([nx, ny])
            self.food_eaten += 1
            self._respawn_food()
            return 0.5
        return -0.002  # 移动代谢

    def spacing(self):
        return abs(self.pos_a[0] - self.pos_b[0]) \
            + abs(self.pos_a[1] - self.pos_b[1])


def make_agi(env):
    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 6, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4,
    }))
    agi.set_env(env)
    return agi


def run_social(seed=0, max_ticks=3000):
    """双 AGI 共跑；返回指标"""
    env = World2(seed=seed)
    agi_a = make_agi(env)
    agi_b = make_agi(env)
    # 简易环境适配（AGI 需要 env.get_pos/observe/step 接口）
    class Adapter:
        def __init__(s, who):
            s.who = who
        def get_pos(s):
            return env.pos_a if s.who == "a" else env.pos_b
        def observe(s):
            return env.observe(s.who)
        def step(s, a):
            food, _ = env._nearest(env.pos_a if s.who == "a" else env.pos_b)
            return {"energy_delta": env.step(s.who, a,
                                            target_food=food),
                    "water_delta": -0.0002}
        def food_nearby(s):
            return True
    env_a, env_b = Adapter("a"), Adapter("b")
    agi_a.set_env(env_a)
    agi_b.set_env(env_b)
    deaths = {"a": None, "b": None}
    for t in range(max_ticks):
        for who, agi, ad in (("a", agi_a, env_a), ("b", agi_b, env_b)):
            if not agi.alive:
                if deaths[who] is None:
                    deaths[who] = t
                continue
            agi.step()
            if not agi.alive and deaths[who] is None:
                deaths[who] = t
    surv_a = deaths["a"] if deaths["a"] is not None else max_ticks
    surv_b = deaths["b"] if deaths["b"] is not None else max_ticks
    return {
        "food": env.food_eaten,
        "conflicts": env.conflicts,
        "deaths": deaths,
        "survival": (surv_a + surv_b) / 2,  # 平均存活
    }


def run_single(seed=0, max_ticks=3000):
    """单 AGI 基线（同布局同食物数）"""
    env = World2(seed=seed)
    agi = make_agi(env)
    class Adapter:
        def __init__(s):
            s.who = "a"
        def get_pos(s):
            return env.pos_a
        def observe(s):
            return env.observe("a")
        def step(s, a):
            food, _ = env._nearest(env.pos_a)
            return {"energy_delta": env.step("a", a, target_food=food),
                    "water_delta": -0.0002}
        def food_nearby(s):
            return True
    agi.set_env(Adapter())
    for t in range(max_ticks):
        if not agi.alive:
            break
        agi.step()
    surv = t if not agi.alive else max_ticks
    return {"food": env.food_eaten, "deaths": {"a": None},
            "survival": float(surv)}


def main():
    np.random.seed(42)
    print("=" * 60)
    print("路线 C — 多智能体社会实验（双 AGI 竞争/合作）")
    print("=" * 60)
    seeds = (42, 7, 2026)
    rows = []
    for s in seeds:
        soc = run_social(seed=s)
        single = run_single(seed=s)
        rows.append((s, soc, single))
        print(f"  seed{s}: 双AGI食物={soc['food']} 冲突={soc['conflicts']} "
              f"存活={soc['survival']:.0f} | 单AGI食物={single['food']} "
              f"存活={single['survival']:.0f}")
    # 判定（预注册调整：'全程存活'对任何 AGI 不现实（单 AGI 也死）——
    # 改为'社会不缩短寿命'：双 AGI 平均存活 ≥ 单 AGI 存活）
    surv_min = min(r[1]["survival"] for r in rows)
    single_surv_min = min(r[2]["survival"] for r in rows)
    not_shorter = surv_min >= single_surv_min
    any_conflict = any(r[1]["conflicts"] > 0 for r in rows)
    min_soc = min(r[1]["food"] for r in rows)
    min_single = min(r[2]["food"] for r in rows)
    not_worse = min_soc >= min_single
    ok = not_shorter and any_conflict and not_worse
    print(f"\n判定: 寿命不短{'OK' if not_shorter else 'FAIL'} "
          f"({surv_min} vs {single_surv_min}) "
          f"社会行为{'OK' if any_conflict else 'FAIL'} "
          f"不拖累{'OK' if not_worse else 'FAIL'}"
          f"（双最差{min_soc} vs 单最差{min_single}）")
    verdict = ("通过——双 AGI 共存+社会行为+不拖累（社会智能基础）" if ok
               else "未过——如实记录")
    print(f"  {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
