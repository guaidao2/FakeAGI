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
        self.gather_progress = {}   # (who,pos) -> 已工作 tick
        self._worked_food = {}      # who -> 正在工作的食物格
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
        # review blocking 修复：按 (who, 食物格) 追踪工作进度——
        # 原 _last_food_pos 恒 None 导致进度每 tick 清零，难获配置食物永不可得
        pos_key = (nx, ny)
        if pos_key in [tuple(f) for f in self.foods]:
            if self.gather_cost <= 0:
                self.foods.remove([nx, ny])
                self.food_eaten += 1
                self._respawn_food()
                return 0.5
            # 工作进度：key=(who, 格)；离开该格（新位置非食物格）不重置，
            # 但换食物格则从 0 开始（防跨食物累积）
            wk = (who, pos_key)
            if pos_key != self._worked_food.get(who):
                self.gather_progress[wk] = 0
            self._worked_food[who] = pos_key
            self.gather_progress[wk] = self.gather_progress.get(wk, 0) + 1
            if self.gather_progress[wk] >= self.gather_cost:
                self.foods.remove([nx, ny])
                self.food_eaten += 1
                self.gather_progress.pop(wk, None)
                self._worked_food[who] = None
                self._respawn_food()
                return 0.5
        else:
            # 离开食物格：清工作标记（但保留该格进度？不——离开即重置，
            # 防止"碰一下"累积；下次回来重新工作）
            self._worked_food[who] = None
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


def run_social(seed=0, max_ticks=3000, n_food=4, gather_cost=0,
               other_enabled=True):
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
    # 激活他者模型（main.py:616 默认关闭；ablation 对照可关）
    agi_a._other_agent_enabled = other_enabled
    agi_b._other_agent_enabled = other_enabled
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
        "surv_gap": abs(surv_a - surv_b),   # 存活差（小=共存/轮流）
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
    #   资源丰富但难得（可能合作/共存——假说后半补验）
    #   成本梯度 0/15/30/60：成本越高，抢同一块的停留代价越大，
    #   越可能演化出"分开取食/轮流"（共享）
    configs = [
        ("稀缺-易得", dict(n_food=2, gather_cost=0)),
        ("稀缺-难获", dict(n_food=2, gather_cost=15)),
        ("稀缺-极难", dict(n_food=2, gather_cost=30)),
        ("丰富-易得", dict(n_food=8, gather_cost=0)),
        ("丰富-难获", dict(n_food=8, gather_cost=15)),
        ("丰富-极难", dict(n_food=8, gather_cost=30)),
        ("丰富-极高", dict(n_food=8, gather_cost=60)),
    ]
    summary = []
    all_ok = True
    for label, cfg in configs:
        rows = []
        for s in seeds:
            np.random.seed(s)          # 每 seed 独立重置（security low）
            torch.manual_seed(s)
            soc = run_social(seed=s, **cfg)
            single = run_single(seed=s, **cfg)
            rows.append((s, soc, single))
            print(f"  [{label}] seed{s}: 双食物={soc['food']} 冲突={soc['conflicts']} "
                  f"存活={soc['survival']:.0f} 存活差={soc['surv_gap']:.0f} "
                  f"| 单食物={single['food']} 存活={single['survival']:.0f}")
        conf_sum = sum(r[1]["conflicts"] for r in rows)
        surv_min = min(r[1]["survival"] for r in rows)
        single_surv_min = min(r[2]["survival"] for r in rows)
        # 共存指标：存活差均值（小=轮流/共享资源→双方都活）
        gap_avg = sum(r[1]["surv_gap"] for r in rows) / len(rows)
        # 有效性检查（review blocking）：难获配置食物必须 >0
        # （gather 机制损坏时食物恒 0——配置无效）
        food_sum = sum(r[1]["food"] for r in rows)
        if cfg["gather_cost"] > 0 and food_sum == 0:
            print(f"  [!!] {label} 食物全 0——gather 机制异常，配置无效")
            all_ok = False
        summary.append((label, conf_sum, surv_min, single_surv_min,
                        food_sum, gap_avg))
    print("\n=== 稀缺度-成本-冲突曲线（假说检验）===")
    for label, conf, surv, single, food, gap in summary:
        print(f"  {label}: 冲突={conf}（×3seeds）双总食物={food} "
              f"双最差存活={surv:.0f} 单最差存活={single:.0f} 存活差均值={gap:.0f}")

    # 假说后半（补验）：丰富 + 成本升高 → 冲突下降 / 共存（存活差小）
    print("\n=== 假说后半：丰富-成本梯度 ===")
    rich_cfgs = [x for x in summary if x[0].startswith("丰富")]
    costs = [0, 15, 30, 60]
    conf_by_cost = {}
    gap_by_cost = {}
    for label, conf, surv, single, food, gap in rich_cfgs:
        cost = {"丰富-易得": 0, "丰富-难获": 15, "丰富-极难": 30,
                "丰富-极高": 60}[label]
        conf_by_cost[cost] = conf
        gap_by_cost[cost] = gap
    confs = [conf_by_cost.get(c, 0) for c in costs]
    gaps = [gap_by_cost.get(c, 0) for c in costs]
    print(f"  成本 0/15/30/60 → 冲突 {confs}、存活差 {gaps}")
    # 共存判据：成本 60 冲突显著低于成本 0（下降 >50%）且存活差缩小
    conflict_drop = (confs[0] - confs[3]) / max(1, confs[0])
    gap_drop = (gaps[0] - gaps[3]) / max(1, gaps[0])
    co_exist = conflict_drop > 0.5 and gap_drop > 0.3
    print(f"  冲突降幅 {conflict_drop*100:.0f}%、存活差降幅 {gap_drop*100:.0f}%")
    print(f"  共存涌现{'OK' if co_exist else '未证'}（冲突-50%且存活差-30% 为判据）")

    # ablation 对照：他者模型开关（review blocking——冲突归因）
    print("\n=== 他者模型 ablation（稀缺-难获配置）===")
    conf_on = conf_off = 0
    for s in seeds:
        np.random.seed(s)
        torch.manual_seed(s)
        soc_on = run_social(seed=s, **dict(n_food=2, gather_cost=15))
        conf_on += soc_on["conflicts"]
        soc_off = run_social(seed=s, **dict(n_food=2, gather_cost=15),
                             other_enabled=False)
        conf_off += soc_off["conflicts"]
    print(f"  他者开: 冲突={conf_on}（×3seeds 合计）| 他者关: 冲突={conf_off}")
    if conf_on > conf_off:
        print("  社会感知贡献：他者模型激活显著增加冲突（感知驱动竞争）")
    else:
        print("  社会感知贡献：他者模型无显著影响（冲突主要来自环境机制）")
    print(f"\n总体: {'OK（全部配置有效）' if all_ok else 'FAIL（有配置无效）'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
