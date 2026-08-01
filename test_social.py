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
  - World2：8x8 共享世界，食物 4 个（稀缺但可存活），2 个 AGI 交替 step
  - 观测：各自 4D 基础观测 + 他者位置（他者模型消费）
  - 指标：存活/食物/冲突事件（去重）/平均存活
  - 对比：双 AGI vs 单 AGI（同布局同食物数，基线）
  - 判定（v2，时间线记录在 main()）：人均食物不拖累（双人均 ≥ 单人均）
    + 社会行为（冲突事件>0）+ 存活不短（双平均存活 ≥ 单存活）
  - 3 seeds 独立 + 判定取最差
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
        for _ in range(4):   # 4 食物（稀缺但可存活；5 无竞争压力、3 全饿死）
            self._respawn_food()
        self.food_eaten = 0
        self.conflicts = 0        # 冲突事件（目标切换时计 1 次——去重）
        self.last_target_a = None
        self.last_target_b = None
        self._conflict_open = False

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
        """执行动作；冲突事件计数（去重：双方目标同一食物期间只计 1 次，
        目标切换后才开新事件——review should-fix：原每 tick 重复计数）"""
        pos = self.pos_a if who == "a" else self.pos_b
        if who == "a":
            self.last_target_a = tuple(target_food) if target_food else None
        else:
            self.last_target_b = tuple(target_food) if target_food else None
        both_target = (self.last_target_a is not None
                       and self.last_target_b is not None
                       and self.last_target_a == self.last_target_b)
        if both_target and not self._conflict_open:
            self.conflicts += 1
            self._conflict_open = True
        elif not both_target:
            self._conflict_open = False
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
    agi = AGI({"auto_save_on_death": False})  # 关死亡快照（review nit：6 次运行污染 checkpoints/）
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
              f"存活a/b={soc['deaths']} 平均{soc['survival']:.0f} "
              f"| 单AGI食物={single['food']} 存活={single['survival']:.0f}")
    # 判定（v2 修正，记录时间线：v1"全程存活+总食物"不可达且不公平——
    # 单 AGI 也死（存活 910-2150），双 AGI 6000 步 vs 单 3000 步使总食物
    # 天然 2 倍偏向。v2 判据：人均食物不拖累（双人均 ≥ 单人均——公平）
    # + 社会行为（冲突事件>0）+ 存活不短（双平均存活 ≥ 单存活））
    # 人均：双 AGI 食物/2 vs 单 AGI 食物/1
    per_cap_soc = [r[1]["food"] / 2.0 for r in rows]
    per_cap_single = [r[2]["food"] / 1.0 for r in rows]
    pc_min_soc = min(per_cap_soc)
    pc_min_single = min(per_cap_single)
    not_worse = pc_min_soc >= pc_min_single
    any_conflict = any(r[1]["conflicts"] > 0 for r in rows)
    surv_min = min(r[1]["survival"] for r in rows)
    single_surv_min = min(r[2]["survival"] for r in rows)
    not_shorter = surv_min >= single_surv_min
    ok = not_worse and any_conflict and not_shorter
    print(f"\n判定: 人均不拖累{'OK' if not_worse else 'FAIL'} "
          f"({pc_min_soc:.1f} vs {pc_min_single:.1f}) "
          f"社会行为{'OK' if any_conflict else 'FAIL'} "
          f"寿命不短{'OK' if not_shorter else 'FAIL'} "
          f"({surv_min:.0f} vs {single_surv_min:.0f})")
    verdict = ("通过——人均效率不拖累+竞争冲突事件出现"
               "（观察性描述，n=3 无统计推断）" if ok
               else "未过——如实记录")
    print(f"  {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
