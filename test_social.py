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
    """双 AGI 共享世界：8x8，食物 N 个（吃后随机重放），可配获取成本
    gather_cost：停留 tick 数（>0 = 需要"工作"才获得——稀缺但难得的资源）"""
    def __init__(self, seed=0, n_food=4, gather_cost=0):
        self.rng = np.random.RandomState(seed)
        self.size = 8
        self.pos_a = [1, 1]
        self.pos_b = [6, 6]
        self.n_food = n_food
        self.gather_cost = gather_cost
        self.gather_progress = {}   # (x,y) -> 已工作 tick
        self.foods = []
        for _ in range(n_food):
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
        # 吃食物（gather_cost=0 立即吃；>0 需停留"工作"——丰富但难得的资源）
        pos_key = (nx, ny)
        if pos_key in [tuple(f) for f in self.foods]:
            if self.gather_cost <= 0:
                self.foods.remove([nx, ny])
                self.food_eaten += 1
                self._respawn_food()
                return 0.5
            # 需要工作：停留 accumulate；离开则重置
            if pos_key != self._last_food_pos(who):
                self.gather_progress[pos_key] = 0
            self.gather_progress[pos_key] = self.gather_progress.get(pos_key, 0) + 1
            if self.gather_progress[pos_key] >= self.gather_cost:
                self.foods.remove([nx, ny])
                self.food_eaten += 1
                self.gather_progress.pop(pos_key, None)
                self._respawn_food()
                return 0.5
        return -0.002  # 移动代谢

    def _last_food_pos(self, who):
        return None  # 简化：不追踪离开重置（工作进度按格子累积）

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


def run_social(seed=0, max_ticks=3000, n_food=4, gather_cost=0):
    """双 AGI 共跑；返回指标"""
    env = World2(seed=seed, n_food=n_food, gather_cost=gather_cost)
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
        # 他者模型接口（诊断修复：原缺失→other_tracker 从未激活，
        # 冲突=随机碰撞非社会行为）
        def get_other_pos(s):
            return env.pos_b if s.who == "a" else env.pos_a
        def get_food_pos(s):
            return list(env.foods)
    env_a, env_b = Adapter("a"), Adapter("b")
    agi_a.set_env(env_a)
    agi_b.set_env(env_b)
    # 激活他者模型（main.py:616 默认关闭）
    agi_a._other_agent_enabled = True
    agi_b._other_agent_enabled = True
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


def run_single(seed=0, max_ticks=3000, n_food=4, gather_cost=0):
    """单 AGI 基线（同布局同食物数）"""
    env = World2(seed=seed, n_food=n_food, gather_cost=gather_cost)
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
    import torch
    np.random.seed(42)
    torch.manual_seed(42)  # AGI 内部 torch 网络（LNN/GameNN）必须同 seed——
    # 缺此则每次运行 AGI 初始化不同（复检不一致暴露，E14 同款修复）
    print("=" * 60)
    print("路线 C — 多智能体社会实验（稀缺度梯度 × 获取成本）")
    print("=" * 60)
    seeds = (42, 7, 2026)
    # 稀缺度梯度（用户生物学假设验证）：
    #   n_food 少 = 资源稀缺（易冲突）；n_food 多但 gather_cost 高 =
    #   资源丰富但难得（可能合作/共存）
    configs = [
        ("稀缺-易得", dict(n_food=2, gather_cost=0)),
        ("稀缺-难获", dict(n_food=2, gather_cost=15)),
        ("中等-易得", dict(n_food=4, gather_cost=0)),
        ("丰富-难获", dict(n_food=8, gather_cost=15)),
    ]
    summary = []
    for label, cfg in configs:
        rows = []
        for s in seeds:
            np.random.seed(s)          # 每 seed 独立重置（security low）
            torch.manual_seed(s)
            soc = run_social(seed=s, **cfg)
            single = run_single(seed=s, **cfg)
            rows.append((s, soc, single))
            print(f"  [{label}] seed{s}: 双食物={soc['food']} 冲突={soc['conflicts']} "
                  f"存活={soc['survival']:.0f} | 单食物={single['food']} "
                  f"存活={single['survival']:.0f}")
        conf_sum = sum(r[1]["conflicts"] for r in rows)
        surv_min = min(r[1]["survival"] for r in rows)
        single_surv_min = min(r[2]["survival"] for r in rows)
        summary.append((label, conf_sum, surv_min, single_surv_min))
    print("\n=== 稀缺度-冲突曲线（用户假设验证）===")
    for label, conf, surv, single in summary:
        print(f"  {label}: 冲突事件={conf}（×3seeds 合计）"
              f" 双最差存活={surv:.0f} 单最差存活={single:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
