"""
过程选择验收（V1-V4）+ 负对照（N1-N4）

设计（DESIGN_PROCESS_SELECTION.md）：
  目标导向的行为选择：落差 → 评估各过程预期收益 → argmax
  过程：ask（问路，占 tick 机会成本 + 响应噪声）/ sweep（扫掠）

验收：
  V1 失败→可靠性↓：环境连续答错 → ask 可靠性必须下降
  V2 误差进模型：问路失败（落差未消解）→ 可靠性更新（非淹没）
  V3 可靠性可逆：恢复正确回答 → ask 可靠性回升
  V4 切换由可靠性驱动：ask 可靠性 < sweep 时选择翻向 sweep

负对照（对比阵列）：
  N1 冻结可靠性：估计锁死 → 表现差（证明在线更新必要）
  N2 问路永远失败：ask_noise=1.0 → 学会"从不用问路"
  N3 噪声：ask_noise=0.3 → 学会"先试自己找，问不到再问"
  N4 免费完美：ask_noise=0.0 → 纯问（正确行为，无诊断力——需对比）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from test_process_env import ProcessEnv
from core.process_selector import ProcessSelector, ProcessEstimator


# ═══ 控制器：闭环（选择→执行→更新）═══

DIR_MAP = {"east": 3, "west": 2, "north": 1, "south": 4}

class ProcessController:
    """过程选择闭环：落差 → 选择 ask/sweep → 执行 → 更新可靠性"""
    def __init__(self, env, selector, seeker=None):
        self.env = env
        self.selector = selector
        self.seeker = seeker  # InfoSeeker（sweep 用）
        self.pending_dir = None  # 问路返回的方向（下一 tick 移动用）
        self.ask_used_tick = 0   # 最近问路 tick
        self.energy = 1.0
        self.alive = True
        self.eaten = 0
        self.tick = 0

    def gap(self):
        return max(0.0, 0.8 - self.energy)

    def step(self):
        self.tick += 1
        gap = self.gap()
        choice = self.selector.choose(gap, self.tick)

        action = 0
        if choice == "ask":
            # 问路：返回 (方向词, 是否答对)，本 tick 按方向移动
            words, answered_correct = self.env.respond("food")
            self.ask_used_tick = self.tick
            if words and len(words) > 1:
                self.pending_dir = DIR_MAP.get(words[1])
                action = self.pending_dir if self.pending_dir is not None else 0
            else:
                self.pending_dir = None
        elif choice == "sweep":
            if self.seeker is not None:
                action = self.seeker.choose_action(self.env.pos)
            else:
                action = np.random.randint(1, 5)
        # else: none → stay

        result = self.env.step(action)
        ed = result["energy_delta"]
        if abs(ed) > 0.001:
            self.energy = max(0.0, min(2.0, self.energy + ed))
        self.eaten = self.env.eaten

        # 过程结果验证
        success = ed > 0.1
        if choice == "ask":
            # 问路效果：用环境返回的"是否答对"（ground truth）
            ask_ok = answered_correct
            self.selector.update_outcome("ask", ask_ok)
        elif choice == "sweep":
            self.selector.update_outcome("sweep", success)
        if self.energy <= 0:
            self.alive = False
        return choice


# ═══ 验收脚本 ═══

def test_v1_v3(seed=0):
    """V1 + V3：连续答错 → 可靠性↓；恢复正确 → 可靠性↑"""
    np.random.seed(seed)
    env = ProcessEnv(ask_noise=1.0, seed=seed)  # 全错环境
    sel = ProcessSelector()
    ctrl = ProcessController(env, sel)
    ctrl.energy = 0.3  # 制造落差（激活选择）
    # V1：全错环境跑 30 tick，ask 可靠性应显著下降
    for _ in range(30):
        ctrl.step()
        if not ctrl.alive:
            break
    rel_after_fail = sel.estimators["ask"].reliability
    # V3：换正确环境，可靠性应回升
    env.ask_noise = 0.0
    for _ in range(30):
        ctrl.step()
        if not ctrl.alive:
            break
    rel_after_ok = sel.estimators["ask"].reliability
    return rel_after_fail, rel_after_ok


def test_v2(seed=0):
    """V2：失败经历确实更新可靠性（非淹没）"""
    np.random.seed(seed)
    env = ProcessEnv(ask_noise=1.0, seed=seed)
    sel = ProcessSelector()
    ctrl = ProcessController(env, sel)
    ctrl.energy = 0.3  # 制造落差
    before = sel.estimators["ask"].reliability
    for _ in range(10):
        ctrl.step()
    after = sel.estimators["ask"].reliability
    return before, after


def test_v4(seed=0):
    """V4：切换由可靠性驱动——ask 可靠性 < sweep 时选择翻向 sweep
    单元级验证：直接设置可靠性，检查 choose 的选择"""
    np.random.seed(seed)
    sel = ProcessSelector()
    # 场景：ask 可靠（0.6）> sweep（0.3）→ 应选 ask
    sel.estimators["ask"].reliability = 0.6
    sel.estimators["sweep"].reliability = 0.3
    c1 = sel.choose(0.5)
    # 场景：ask 崩（0.1）< sweep（0.4）→ 应选 sweep
    sel.estimators["ask"].reliability = 0.1
    sel.estimators["sweep"].reliability = 0.4
    c2 = sel.choose(0.5)
    return c1, c2


# ═══ 负对照 ═══

def run_group(name, ask_noise, freeze=False, seeds=5, ticks=300,
              switch_noise=None):
    """N 组跑分：存活 tick + 食物
    switch_noise: 若指定，前一半 tick 用 ask_noise，后一半切换到此噪声
    （测"冻结无法适应环境变化"）"""
    from core.info_seeking import InfoSeeker
    results = []
    for s in range(seeds):
        np.random.seed(s)
        env = ProcessEnv(ask_noise=ask_noise, seed=s)
        sel = ProcessSelector()
        if freeze:
            sel.estimators["ask"].freeze()
            sel.estimators["sweep"].freeze()
        seeker = InfoSeeker(grid_size=env.size)
        ctrl = ProcessController(env, sel, seeker)
        ctrl.energy = 0.4  # 制造落差（激活过程选择）
        asks = 0
        for t in range(ticks):
            # 环境切换（测适应）：前一半好环境，后一半切换
            if switch_noise is not None and t == ticks // 2:
                env.ask_noise = switch_noise
            c = ctrl.step()
            if c == "ask":
                asks += 1
            if not ctrl.alive:
                break
        results.append({
            "survived": ctrl.tick, "food": ctrl.eaten,
            "alive": ctrl.alive, "asks": asks,
        })
    return results


def test():
    print("过程选择验收（V1-V4）+ 负对照（N1-N4）", flush=True)
    print("=" * 60, flush=True)

    # ─── V1+V3 ───
    rel_fail, rel_ok = test_v1_v3()
    v1 = rel_fail < 0.3
    v3 = rel_ok > rel_fail + 0.1
    print(f"\nV1: 全错环境 ask 可靠性 {rel_fail:.3f} (<0.3: {'OK' if v1 else 'FAIL'})", flush=True)
    print(f"V3: 恢复正确后可靠性 {rel_ok:.3f} (回升: {'OK' if v3 else 'FAIL'})", flush=True)

    # ─── V2 ───
    b, a = test_v2()
    v2 = a < b - 0.05
    print(f"V2: ask 可靠性 {b:.3f}→{a:.3f} (下降: {'OK' if v2 else 'FAIL'})", flush=True)

    # ─── V4 ───
    c1, c2 = test_v4()
    v4 = (c1 == "ask" and c2 == "sweep")
    print(f"V4: ask 优→选 ask({c1}), sweep 优→选 sweep({c2}): "
          f"{'OK' if v4 else 'FAIL'}", flush=True)

    # ─── N 组 ───
    print(f"\n── 负对照（对比阵列，n=5）──", flush=True)
    # N1/N4 用环境切换（好→坏）：冻结应无法适应，未冻结应学会 sweep
    n1 = run_group("N1 冻结可靠性", ask_noise=0.0, freeze=True, switch_noise=0.9)
    n4 = run_group("N4 免费完美(切换)", ask_noise=0.0, switch_noise=0.9)
    n2 = run_group("N2 问路永远失败", ask_noise=1.0)
    n3 = run_group("N3 噪声", ask_noise=0.3)
    for name, res in [("N1 冻结+切换", n1), ("N4 未冻结+切换", n4),
                      ("N2 全败", n2), ("N3 噪声", n3)]:
        food = np.mean([r["food"] for r in res])
        surv = np.mean([r["survived"] for r in res])
        asks = np.mean([r["asks"] for r in res])
        print(f"  {name}: 食物 {food:.1f} 存活 {surv:.0f} 问路 {asks:.0f}", flush=True)

    # ─── 对比阵列判定 ───
    # N2：问路永远失败 → 学会少问（全段问路 < N4 的 40%）
    a_n2 = np.mean([r["asks"] for r in n2])
    a_n4 = np.mean([r["asks"] for r in n4])
    # N1 vs N4：环境切换后，冻结的仍疯狂问（ask 可靠性锁 0.5），未冻结的学会 sweep
    # 用"切换后问路占比"：N4 后半段问路应大幅下降，N1 保持
    # 简化：N1 全段问路应显著高于 N4（冻结无法适应坏环境）
    f_n1 = np.mean([r["food"] for r in n1])
    f_n4 = np.mean([r["food"] for r in n4])
    n_ok = (a_n2 < a_n4 * 0.5          # N2 学会少问（含试探，宽松）
            and (np.mean([r["asks"] for r in n1]) >
                 np.mean([r["asks"] for r in n4]) * 1.3))  # N1 冻结仍问
    print(f"\n  对比阵列: N2 问路 {a_n2:.0f} vs N4 {a_n4:.0f}（N2 应显著少问）", flush=True)
    print(f"  N1 冻结问路 {np.mean([r['asks'] for r in n1]):.0f} vs N4 {a_n4:.0f} "
          f"（冻结应仍问——无法适应切换）", flush=True)

    checks = [("V1 失败→可靠性↓", v1), ("V2 误差进模型", v2),
              ("V3 可逆", v3), ("V4 可靠性驱动切换", v4),
              ("N 对比阵列", n_ok)]
    passed = all(c for _, c in checks)
    print(f"\n判定: {'OK 通过' if passed else 'FAIL 未通过'}", flush=True)
    for name, ok in checks:
        print(f"  {'[OK]' if ok else '[X]'} {name}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(test())
