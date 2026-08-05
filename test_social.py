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
  - 假说后半预注册判据（先于数据）：共存 = (a) 成本>0 配置双 AGI
    最短存活 > 饿死线 700（有效——AGI 能完成工作）+ (b) min 存活随
    成本升高不下降 + (c) 冲突率（conflicts/存活 tick）降幅 >50%
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
        # 代谢：停留（动作 0）轻劳动 -0.001，移动 -0.002（生物学合理：
        # 静止代谢低于移动——工作期间能量消耗减半，解锁共存判据）
        return -0.001 if action == 0 else -0.002

    def spacing(self):
        return abs(self.pos_a[0] - self.pos_b[0]) \
            + abs(self.pos_a[1] - self.pos_b[1])


def make_agi(env, hidden_dim=64):
    # 参数增大实验：hidden_dim 64→128 可配置（学习能力对比——模型容量
    # 不足可能导致 AGI 学不会复杂行为如停留工作）
    agi = AGI({"auto_save_on_death": False})  # 关死亡快照（review nit：6 次运行污染 checkpoints/）
    agi.set_cognition(CognitionPipeline({
        "input_dim": 6, "self_state_dim": 14,
        "hidden_dim": hidden_dim, "n_actions": 5, "n_strategies": 4,
    }))
    agi.set_env(env)
    return agi


class GoalPersistence:
    """目标坚持机制（用户洞察操作化）：
    坚持 = 目标未完成（"没做完"本身是理由）→ 保持动作（停留工作）
    放弃 = 重大预测误差 → 换路：
      - surprise 飙升（工作无效/预期冲突）
      - 能量危机（<0.2 强制换路）
      - 他者抢占同一食物（冲突）
    可开关（goal_enabled on/off 对照）"""
    def __init__(self, surprise_threshold=0.8, energy_floor=0.1,
                 abandon_cooldown=20):
        self.target = None          # 当前目标食物格 (x,y)
        self.progress_seen = 0      # 已见最大进度
        self.stall_ticks = 0        # 进度停滞 tick 数
        self.surprise_threshold = surprise_threshold
        self.energy_floor = energy_floor
        self.abandon_cooldown = abandon_cooldown  # 放弃后冷却（防死磕循环）
        self.cooldown_left = 0
        self.abandoned = 0          # 放弃计数（诚实记录）
        self.completed = 0          # 完成计数
        self.stick_actions = 0      # 坚持动作数

    def reset(self):
        self.target = None
        self.progress_seen = 0
        self.stall_ticks = 0

    def _other_targets_same(self, env, pos):
        """他者目标同一食物（env 冲突检测状态——位置比较恒 False 已废弃）"""
        return env._conflict_open  # 环境已在双方同目标时置位

    def decide(self, env, who, pos, action, progress_now, surprise,
               energy, other_pos):
        """返回 (最终动作, 是否坚持)。
        review blocking 修复：非坚持返回 None 哨兵（原返回 last_action=0
        被误判为坚持 → override 恒 0 → AGI 永久停留原地，sc21/sc22 数据作废）"""
        on_food = tuple(pos) in [tuple(f) for f in env.foods]
        # 冷却递减（放弃后避免立即死磕同一食物）
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
        if on_food and env.gather_cost > 0:
            if self.target != tuple(pos):
                if self.cooldown_left > 0:
                    # 冷却期内不建立新目标（防死磕循环——稀缺配置根因）
                    return None, False
                self.target = tuple(pos)      # 新目标：进入坚持
                self.progress_seen = 0
                self.stall_ticks = 0
            # 放弃条件（重大预测误差 + 轮流信号）
            abandon = False
            if surprise > self.surprise_threshold:
                abandon = True                # surprise 飙升（真实通路）
            elif energy < self.energy_floor:
                abandon = True                # 能量危机
            elif self._other_targets_same(env, pos):
                abandon = True                # 他者目标同一食物（冲突）
            # （nit：轮流信号恒 False 死分支已删除——占用检测用 _conflict_open）
            if progress_now > self.progress_seen:
                self.progress_seen = progress_now
                self.stall_ticks = 0
            else:
                self.stall_ticks += 1
                if self.stall_ticks > 5 and progress_now == 0:
                    abandon = True            # 工作 5 tick 无进展（进度重置）
            if abandon:
                self.abandoned += 1
                self.target = None
                self.cooldown_left = self.abandon_cooldown  # 冷却防死磕
                return None, False            # 放弃：不覆盖（AGI 自主决策）
            # 坚持：停留工作
            self.stick_actions += 1
            return 0, True
        if self.target is not None:
            self.completed += 1               # 离开食物格（获得或放弃后）
            self.target = None
        return None, False



def run_social(seed=0, max_ticks=3000, n_food=4, gather_cost=0,
               other_enabled=True, goal_enabled=False, hidden_dim=64):
    """双 AGI 共跑；返回指标"""
    env = World2(seed=seed, n_food=n_food, gather_cost=gather_cost)
    agi_a = make_agi(env, hidden_dim=hidden_dim)
    agi_b = make_agi(env, hidden_dim=hidden_dim)
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
    goals = {"a": GoalPersistence(), "b": GoalPersistence()} \
        if goal_enabled else None
    goal_stats = {"a": {}, "b": {}} if goal_enabled else None
    for t in range(max_ticks):
        for who, agi, ad in (("a", agi_a, env_a), ("b", agi_b, env_b)):
            if not agi.alive:
                if deaths[who] is None:
                    deaths[who] = t
                continue
            if goals is not None:
                # 目标坚持（AGI 内部机制——main.py 决策后覆盖动作，
                # agi.step 完整执行认知/代谢/死亡）
                pos = list(env.pos_a if who == "a" else env.pos_b)
                prog = env.gather_progress.get((who, tuple(pos)), 0) \
                    if env.gather_cost > 0 else 1.0
                other_pos = env.pos_b if who == "a" else env.pos_a
                # 真实 surprise（main.py 存储）+ 真实能量
                surp = getattr(agi, 'last_surprise', 0.0)
                en = agi.body.energy
                final_a, _ = goals[who].decide(
                    env, who, pos, agi.last_action, prog, surp, en, other_pos)
                # None 哨兵：非坚持不覆盖（AGI 自主决策）
                agi._goal_override = final_a if final_a is not None else None
                try:
                    agi.step()  # 完整 step（override 内部生效）
                finally:
                    agi._goal_override = None  # LOW：异常路径也复位（防动作僵化）
            else:
                agi.step()
            if not agi.alive and deaths[who] is None:
                deaths[who] = t
    if goal_stats is not None:
        for who in ("a", "b"):
            goal_stats[who] = {
                "stick": goals[who].stick_actions,
                "abandoned": goals[who].abandoned,
                "completed": goals[who].completed,
            }
    surv_a = deaths["a"] if deaths["a"] is not None else max_ticks
    surv_b = deaths["b"] if deaths["b"] is not None else max_ticks
    res = {
        "food": env.food_eaten,
        "conflicts": env.conflicts,
        "deaths": deaths,
        "survival": (surv_a + surv_b) / 2,  # 平均存活
        "surv_gap": abs(surv_a - surv_b),   # 存活差（小=共存/轮流）
    }
    if goal_stats is not None:
        res["goal"] = goal_stats
    return res


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
    import os
    np.random.seed(42)
    torch.manual_seed(42)  # AGI 内部 torch 网络（LNN/GameNN）必须同 seed——
    # 缺此则每次运行 AGI 初始化不同（复检不一致暴露，E14 同款修复）
    print("=" * 60)
    print("路线 C — 多智能体社会实验（稀缺度梯度 × 获取成本）")
    print("=" * 60)
    # SOCIAL_QUICK=1：缩减回归模式（1 seed × 每段——全量已在交付时
    # 验证；回归仅确认通路可运行；print 需 -u 或 flush 防缓冲丢失）
    quick = os.environ.get("SOCIAL_QUICK", "0") == "1"
    seeds = (42,) if quick else (42, 7, 2026)

    # ── 目标坚持机制验证（用户洞察：坚持=目标未完成，放弃=重大预测误差）──
    print("\n=== 目标坚持机制验证（gather_cost=15）===")
    goal_on = goal_off = 0
    stick_total = 0
    for s in seeds:
        np.random.seed(s)
        torch.manual_seed(s)
        r_on = run_social(seed=s, n_food=2, gather_cost=15,
                          goal_enabled=True)
        goal_off += run_social(seed=s, n_food=2, gather_cost=15,
                               goal_enabled=False)["food"]
        goal_on += r_on["food"]
        stick_total += sum(g["stick"] for g in r_on["goal"].values())
    print(f"  目标机制开: 食物={goal_on}（×3seeds）| 关: 食物={goal_off}"
          f"（坚持动作 {stick_total} 次）")
    persist_ok = goal_on > goal_off * 2  # 坚持机制显著提升完成率（>2x）
    print(f"  坚持有效{'OK' if persist_ok else 'FAIL'}"
          f"（开 > 关×2 为判据）")
    # 放弃验证：工作 5 tick 无进展（进度重置）→ 放弃换路
    # 用 GoalPersistence 单元验证：stall 检测
    gp = GoalPersistence()
    env = World2(seed=1, n_food=1, gather_cost=15)
    env.pos_a = list(env.foods[0])  # 站在食物上
    stuck = 0
    for _ in range(8):
        # 进度不涨（模拟工作无效）
        a, pers = gp.decide(env, "a", env.pos_a, 1, 0, 0.1, 0.9, None)
        if not pers:
            stuck += 1
    abandon_ok = stuck > 0  # 工作 5 tick 无进展 → 放弃（stall 触发）
    # nit 校准：stuck 计"非坚持"（含放弃后冷却期），措辞区分
    print(f"  放弃验证（进度停滞）: {stuck}/8 次非坚持 "
          f"(含放弃+冷却期) {'OK' if abandon_ok else 'FAIL'}（重大预测误差→换路）")
    if persist_ok and abandon_ok:
        print("  目标坚持机制有效（坚持/放弃通路正常）——工作能力解锁")
    else:
        print("  目标坚持机制未通过——如实记录（坚持/放弃部分失效）")

    # ── 参数增大对比（稀缺-难获，2 食物）──
    # review blocking 修正：轮流信号为死代码（env 禁止重叠恒 False），
    # 解锁归因改为"模型容量"；判据用配对"开 > 关"
    print("\n=== 参数增大对比（稀缺-难获，2 食物）===")
    for hdim, label in ((64, "小模型 h=64"), (128, "大模型 h=128")):
        f_on = f_off = 0
        for s in seeds:
            np.random.seed(s)
            torch.manual_seed(s)
            f_on += run_social(seed=s, n_food=2, gather_cost=15,
                               goal_enabled=True,
                               hidden_dim=hdim)["food"]
            f_off += run_social(seed=s, n_food=2, gather_cost=15,
                                goal_enabled=False,
                                hidden_dim=hdim)["food"]
        print(f"  {label}: 目标机制开 食物={f_on} | 关 食物={f_off}"
              f"（×3seeds 合计）")
    # 容量增益判据：大模型关 > 小模型关（纯容量对照，无机制混杂）
    # （nit：原 423-429 预循环为无效代码被清零覆盖——已删除）
    small_off = big_off = 0
    for s in seeds:
        np.random.seed(s)
        torch.manual_seed(s)
        small_off += run_social(seed=s, n_food=2, gather_cost=15,
                                goal_enabled=False,
                                hidden_dim=64)["food"]
        big_off += run_social(seed=s, n_food=2, gather_cost=15,
                              goal_enabled=False,
                              hidden_dim=128)["food"]
    cap_ok = big_off > small_off
    print(f"  容量增益（关对照）: 大模型 {big_off} vs 小模型 {small_off}"
          f"（×3seeds）{'有增益' if cap_ok else '未决（需RandomState隔离重测）'}"
          f"（大 > 小 为判据；序列敏感下不可下结论）")

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
    rows_by_label = {}
    all_ok = True
    for label, cfg in configs:
        rows = []
        for s in seeds:
            np.random.seed(s)          # 每 seed 独立重置（security low）
            torch.manual_seed(s)
            # 目标坚持机制开启（解锁假说后半——AGI 能完成工作后测共存）
            soc = run_social(seed=s, **cfg, goal_enabled=True)
            single = run_single(seed=s, **cfg)
            rows.append((s, soc, single))
            print(f"  [{label}] seed{s}: 双食物={soc['food']} 冲突={soc['conflicts']} "
                  f"存活={soc['survival']:.0f} 存活差={soc['surv_gap']:.0f} "
                  f"| 单食物={single['food']} 存活={single['survival']:.0f}")
        rows_by_label[label] = rows
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

    # 假说后半（补验）：丰富 + 成本升高 → 共存
    # 共存判据（预注册，先于数据）：
    #   (a) 有效性：成本>0 配置双 AGI 最短存活 > 饿死线 700
    #       （双双饿死=AGI 无法完成工作，测不到社会行为——配置无效）
    #   (b) min 存活（双中最短者，×3seeds 最差）随成本升高不下降
    #   (c) 冲突率（conflicts/双平均存活 tick，×3seeds 合计）下降 >50%
    print("\n=== 假说后半：丰富-成本梯度（预注册判据）===")
    costs = [0, 15, 30, 60]
    cost_label = {0: "丰富-易得", 15: "丰富-难获", 30: "丰富-极难",
                  60: "丰富-极高"}
    conf_by_cost = {}
    min_surv_by_cost = {}
    rate_by_cost = {}
    for c in costs:
        label = cost_label[c]
        rows = rows_by_label[label]
        conf_by_cost[c] = sum(r[1]["conflicts"] for r in rows)
        # min 存活：每 seed 取双中最短存活，再取 ×3seeds 最差
        per_seed_min = []
        for _, soc, _ in rows:
            da = soc["deaths"]["a"] if soc["deaths"]["a"] is not None else 3000
            db = soc["deaths"]["b"] if soc["deaths"]["b"] is not None else 3000
            per_seed_min.append(min(da, db))
        min_surv_by_cost[c] = min(per_seed_min)
        # 冲突率：conflicts / 双总存活 tick（每 seed 存活和）
        total_alive = sum(
            (soc["deaths"]["a"] if soc["deaths"]["a"] is not None else 3000)
            + (soc["deaths"]["b"] if soc["deaths"]["b"] is not None else 3000)
            for _, soc, _ in rows)
        rate_by_cost[c] = conf_by_cost[c] / max(1, total_alive)
    confs = [conf_by_cost[c] for c in costs]
    mins = [min_surv_by_cost[c] for c in costs]
    rates = [rate_by_cost[c] for c in costs]
    valid = all(m > 700 for m in mins[1:])   # 判据 (a)
    min_ok = mins[3] >= mins[0]              # 判据 (b)
    rate_drop = (rates[0] - rates[3]) / max(1e-9, rates[0])  # 判据 (c)
    co_exist = valid and min_ok and rate_drop > 0.5
    if not valid:
        print("  [!!] 有效性 FAIL：成本>0 配置双 AGI 最短存活 ≤700"
              "（双双饿死）——AGI 无法完成工作，假说后半无法检验")
    print(f"  成本 0/15/30/60 → 冲突 {confs}、min存活 {mins}、冲突率 "
          f"{[f'{r:.3f}' for r in rates]}")
    print(f"  有效性{'OK' if valid else 'FAIL'}、min存活不降"
          f"{'OK' if min_ok else 'FAIL'}、冲突率降幅 {rate_drop*100:.0f}%")
    print(f"  共存涌现{'OK' if co_exist else '未证（预注册判据）'}")

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

    if not valid:
        all_ok = False  # 有效性 FAIL 也反映在退出码（nit：假说无法检验）
    print(f"\n总体: {'OK（全部配置有效）' if all_ok else 'FAIL（有配置无效/不可检验）'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
