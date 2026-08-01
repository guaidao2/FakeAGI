"""
② 过程选择第二阶段 — 完整主循环接入冒烟（单 seed 验证）

验证目标（PREREGISTER_PROCESS_SELECTION.md）：
  A. 零影响护栏：_process_sel_enabled=False 时与第一阶段完全一致
  B. 接入存活：开启后 ask 过程真实调用（speak→answer_query→方向→行动）
  C. V 曲线：ask 可靠性随失败下降（N2 组 30 tick 全错）
  D. 落差消解率信号：窗口化判定（问路后 N tick 内落差下降才算消解）

不修改任何旧文件——独立环境 + 独立控制器循环，隔离验证。
"""
import os, sys, json, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "brain001"))

from core.process_selector import ProcessSelector, ProcessEstimator

VOCAB = ["food", "water", "east", "west", "north", "south"]


class AskEnv:
    """问路环境：食物方向只通过主动问路 answer_query 透露（可配置噪声/失败）"""
    def __init__(self, size=16, mode="perfect", ask_available=True):
        self.size = size
        self.mode = mode            # perfect / noisy / fail / unavailable
        self.ask_available = ask_available
        self.pos = [8, 8]
        self.water_pos = [3, 3]
        self.food_pos = self._random_food()
        self.food_eaten = 0
        self.steps = 0
        self.ask_calls = 0
        self.words = []             # 被动广播（本测试不用——坑2：必须主动问）

    def _random_food(self):
        while True:
            p = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            d = abs(p[0]-self.pos[0])+abs(p[1]-self.pos[1])
            if d > self.size // 2:
                return p

    def get_pos(self): return self.pos

    def observe(self):
        wx, wy = self.water_pos
        return np.array([(wx-self.pos[0])/self.size, (wy-self.pos[1])/self.size])

    def answer_query(self, word):
        """主动问路接口（坑2：不依赖被动广播）"""
        self.ask_calls += 1
        if not self.ask_available:
            return None
        if word not in ("food", "water"):
            return None
        if self.mode == "fail":
            return None
        target = self.food_pos if word == "food" else self.water_pos
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        if self.mode == "noisy" and np.random.random() < 0.3:
            dx, dy = -dx, -dy  # 30% 答错
        if abs(dx) >= abs(dy):
            return "east" if dx > 0 else ("west" if dx < 0 else ("south" if dy > 0 else "north"))
        return "south" if dy > 0 else "north"

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

    def get_damage(self, a): return 0.0

    def food_nearby(self):
        return abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1]) < 2


class Controller:
    """
    独立控制器：模拟主循环接入点（落差门控后、行动合成前）。
    坑3对策：选择器输出记录 why 字段（selector/overridden）。
    坑4对策：落差消解率窗口化（ask 后 WINDOW tick 内能量上升=消解）。
    """
    WINDOW = 15

    def __init__(self, env, process_enabled=True, frozen=False):
        self.env = env
        self.pos = [8, 8]
        self.energy = 0.5
        self.water = 0.5
        self.alive = True
        self.tick = 0
        self.process_enabled = process_enabled
        self.selector = ProcessSelector() if process_enabled else None
        if self.selector is not None and frozen:
            for est in self.selector.estimators.values():
                est.freeze()
        self.ask_count = 0
        self.sweep_count = 0
        self.why = {"selector:ask": 0, "selector:sweep": 0,
                    "overridden:reflex": 0, "overridden:habit": 0}
        self.ask_reliability_trace = []
        self._ask_ever_updated = False
        # 落差消解率信号
        self._ask_history = []   # (tick, energy_at_ask)
        self._ask_success = 0
        self._ask_fail = 0

    def step(self):
        self.tick += 1
        # 消耗
        self.energy -= 0.002
        self.water -= 0.0005
        if self.energy <= 0:
            self.alive = False
            return

        # 落差（能量低→找食物）
        gap = 0.8 - self.energy
        need_food = self.energy < 0.6

        action = 0
        if self.process_enabled and need_food:
            # 过程选择：ask vs sweep（selector 内部：gap>0.15 + argmax + 5% 试探）
            choice = self.selector.choose(gap=gap, tick=self.tick)
            if choice == "ask":
                # 坑1：主动问路（speak→answer_query）
                word = "food" if self.energy < self.water else "water"
                direction = self.env.answer_query(word)
                self.ask_count += 1
                self.why["selector:ask"] += 1
                # 方向→动作（东西南北）
                DIR_MAP = {"east": 3, "west": 2, "north": 1, "south": 4}
                if direction in DIR_MAP:
                    action = DIR_MAP[direction]
                self._ask_history.append((self.tick, self.energy))
                self._update_ask_reliability(direction is not None)
            elif choice == "sweep":
                self.sweep_count += 1
                self.why["selector:sweep"] += 1
                # 扫掠：随机移动（简化，InfoSeeker 语义在完整循环）
                action = np.random.randint(1, 5)
            else:
                # choice == "none"（gap 低或无可用过程）→ 反射兜底
                self.why["overridden:reflex"] += 1
                action = np.random.randint(1, 5)
        else:
            # 默认：反射觅食（无过程选择）
            self.why["overridden:reflex"] += 1
            action = np.random.randint(1, 5)

        # 执行
        d = self.env.step(action)
        self.energy += d["energy_delta"]
        self.water += d["water_delta"]
        # 坑4：窗口化落差消解率（问路后 WINDOW tick 内能量上升=消解）
        self._check_ask_outcome()

        # 记录可靠性轨迹（ask 更新后记录）
        if self.selector is not None and self._ask_ever_updated:
            r = self.selector.estimators["ask"].reliability
            if not self.ask_reliability_trace or abs(r - self.ask_reliability_trace[-1]) > 1e-6:
                self.ask_reliability_trace.append(round(r, 3))

    def _update_ask_reliability(self, got_direction: bool):
        """即时信号：得到方向→ask 可靠性升；没得到→降（环境无响应）"""
        if self.selector is None:
            return
        self.selector.update_outcome("ask", success=got_direction)
        self._ask_ever_updated = True

    def _check_ask_outcome(self):
        """窗口化落差消解率（坑4）：问路后 WINDOW tick 内能量上升=消解"""
        if self.selector is None:
            return
        pending = [h for h in self._ask_history
                   if self.tick - h[0] <= self.WINDOW and h[0] != self.tick]
        if not pending:
            return
        for (at, ae) in pending:
            if self.energy > ae + 0.05:   # 消解
                self._ask_success += 1
                self._ask_history.remove((at, ae))
            elif self.tick - at > self.WINDOW:  # 超窗未消解
                self._ask_fail += 1
                self._ask_history.remove((at, ae))


def run_episode(mode="perfect", process_enabled=True, frozen=False,
                ask_available=True, max_ticks=2000, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = AskEnv(mode=mode, ask_available=ask_available)
    ctl = Controller(env, process_enabled=process_enabled, frozen=frozen)
    died_at = None
    for t in range(max_ticks):
        ctl.step()
        if not ctl.alive:
            died_at = t
            break
    return {
        "died_at": died_at,
        "food": env.food_eaten,
        "ask": ctl.ask_count,
        "sweep": ctl.sweep_count,
        "why": ctl.why,
        "ask_reliability": ctl.ask_reliability_trace,
        "ask_success": ctl._ask_success,
        "ask_fail": ctl._ask_fail,
    }


def main():
    print("=" * 60)
    print("② 过程选择 — 完整主循环接入冒烟（单 seed）")
    print("=" * 60)

    # A. 零影响护栏：process_enabled=False 完全无 ask
    r_a = run_episode(mode="perfect", process_enabled=False, seed=0)
    a_ok = r_a["ask"] == 0
    print(f"\n[A] 零影响护栏: 关闭时 ask={r_a['ask']} (应为0) "
          f"{'OK' if a_ok else 'FAIL'}")

    # B. 接入存活：开启后 ask 真实调用
    r_b = run_episode(mode="perfect", process_enabled=True, seed=0)
    b_ok = r_b["ask"] > 0
    print(f"[B] 接入存活: perfect 组 ask={r_b['ask']} (应>0) "
          f"{'OK' if b_ok else 'FAIL'}")

    # C. V 曲线：N2 组（fail）ask 可靠性因失败下降（0.5→0.2 后 argmax 翻盘）
    r_c = run_episode(mode="fail", process_enabled=True, seed=0, max_ticks=300)
    trace = r_c["ask_reliability"]
    dropped = trace and (trace[-1] < trace[0] - 0.1)
    c_ok = len(trace) >= 2 and dropped
    print(f"[C] V曲线: fail 组可靠性 {trace[0] if trace else '?'}→"
          f"{trace[-1] if trace else '?'} (降幅 "
          f"{trace[0]-trace[-1] if trace else 0:.2f}, 应>0.1) "
          f"{'OK' if c_ok else 'FAIL'}")

    # D. 落差消解率窗口化：perfect 组 ask_success > ask_fail
    r_d = run_episode(mode="perfect", process_enabled=True, seed=0)
    d_ok = r_d["ask_success"] > r_d["ask_fail"]
    print(f"[D] 消解率: perfect 组 success={r_d['ask_success']} "
          f"fail={r_d['ask_fail']} (应 success>fail) {'OK' if d_ok else 'FAIL'}")

    # E. N2 vs N4 方向：fail 组 ask 显著少于 perfect 组
    r_e2 = run_episode(mode="fail", process_enabled=True, seed=0)
    r_e4 = run_episode(mode="perfect", process_enabled=True, seed=0)
    e_ok = r_e2["ask"] < r_e4["ask"]
    print(f"[E] N2 vs N4: fail ask={r_e2['ask']} vs perfect ask={r_e4['ask']} "
          f"(应 fail 少) {'OK' if e_ok else 'FAIL'}")

    # F. 模型驱动切换：fail 组 ask 失败后转 sweep（argmax 翻盘，非硬编码）
    r_f = run_episode(mode="fail", process_enabled=True, seed=0)
    f_ok = r_f["sweep"] > r_f["ask"]
    print(f"[F] 切换: fail 组 sweep={r_f['sweep']} vs ask={r_f['ask']} "
          f"(应 sweep 多——argmax 翻盘) {'OK' if f_ok else 'FAIL'}")

    ok = a_ok and b_ok and c_ok and d_ok and e_ok and f_ok
    print(f"\n判定: {'OK 通过' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
