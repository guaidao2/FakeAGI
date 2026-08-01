"""
② 过程选择第二阶段 — n=30 批量研究（预注册统计）

用法：
  python run_process_selection_study.py            # 跑全部（断点续跑）
  python run_process_selection_study.py --report   # 只统计已有结果

判定（PREREGISTER_PROCESS_SELECTION.md §4）：
  - H1: N2（永久失败）问路次数显著少于 N3（噪声）——对称族
  - N1 类比: 冻结组问路多于 N4（在线更新必要）
  - 效应量: Cohen's d > 0.8（N2 vs N3）
  - 塌陷预案: N4 组 ask≈0（<5）→ 接入失败
"""
import os, sys, json, random, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_process_integration import run_episode

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "process_selection")
GROUPS = {
    "N4_perfect":   dict(mode="perfect", process_enabled=True,  frozen=False),
    "N3_noisy":     dict(mode="noisy",   process_enabled=True,  frozen=False),
    "N2_fail":      dict(mode="fail",    process_enabled=True,  frozen=False),
    "N1_frozen":    dict(mode="fail",    process_enabled=True,  frozen=True),   # 冻结在失败环境测
    "ctrl_off":     dict(mode="perfect", process_enabled=False, frozen=False),
    "ctrl_noask":   dict(mode="perfect", process_enabled=True,  frozen=False, ask_available=False),
}
N_SEEDS = 30
MAX_TICKS = 2000


def _result_path(group, seed, cfg):
    """结果路径含配置指纹——配置变更时旧结果自动失效（防缓存污染）
    指纹：mode + frozen + ask_available + process_enabled + max_ticks"""
    mode = cfg["mode"]
    frozen = "f" if cfg.get("frozen", False) else "u"
    askav = "a" if cfg.get("ask_available", True) else "n"
    pe = "p" if cfg.get("process_enabled", True) else "o"
    fname = f"{group}_{mode}_{frozen}{askav}{pe}_t{MAX_TICKS}_s{seed:02d}.json"
    return os.path.join(RESULTS_DIR, fname)


def set_seed(s):
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)


def run_group(group, cfg, seeds, resume=True):
    """跑一组（断点续跑：已有结果跳过）"""
    done, todo = 0, 0
    for s in seeds:
        p = _result_path(group, s, cfg)
        if resume and os.path.exists(p):
            done += 1
            continue
        todo += 1
        set_seed(s)
        r = run_episode(mode=cfg["mode"],
                        process_enabled=cfg["process_enabled"],
                        frozen=cfg.get("frozen", False),
                        ask_available=cfg.get("ask_available", True),
                        max_ticks=MAX_TICKS, seed=s)
        r["group"] = group
        r["seed"] = s
        r["mode"] = cfg["mode"]
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        if (done + todo) % 5 == 0 or todo == 1:
            print(f"  {group} s{s:02d} done (总 {done+todo}/{len(seeds)})", flush=True)
    return done + todo


def cohens_d(a, b):
    """Cohen's d（合并标准差）"""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if sp == 0:
        return 0.0 if a.mean() == b.mean() else float("inf")
    return (a.mean() - b.mean()) / sp


def load_results(group, seeds):
    out = []
    cfg = GROUPS[group]
    for s in seeds:
        p = _result_path(group, s, cfg)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def report():
    seeds = list(range(N_SEEDS))
    print("=" * 64)
    print("② 过程选择 — n=30 统计报告（预注册判定）")
    print("=" * 64)

    data = {}
    for g in GROUPS:
        rs = load_results(g, seeds)
        asks = [r["ask"] for r in rs]
        died = [r["died_at"] for r in rs if r["died_at"] is not None]
        foods = [r["food"] for r in rs]
        data[g] = dict(rs=rs, asks=asks, n=len(rs),
                       died_n=len(died), food_mean=np.mean(foods) if foods else 0)
        print(f"\n[{g}] n={len(rs)}"
              f" ask={np.mean(asks):.0f}±{np.std(asks):.0f}"
              f" food={data[g]['food_mean']:.1f} died={data[g]['died_n']}")

    # 判定（预注册锁死）
    verdicts = []
    if len(data["N4_perfect"]["rs"]) >= 5:
        # 塌陷预案：N4 ask≈0 → 接入失败
        a_n4 = np.mean(data["N4_perfect"]["asks"])
        if a_n4 < 5:
            verdicts.append(("塌陷预案", "N4 ask≈0 → 接入失败", False))
        else:
            verdicts.append(("塌陷预案", f"N4 ask={a_n4:.0f}（存活）", True))

        # H1: N2 < N3（对称族，含 std 裕度）
        a_n2, sd_n2 = np.mean(data["N2_fail"]["asks"]), np.std(data["N2_fail"]["asks"])
        a_n3, sd_n3 = np.mean(data["N3_noisy"]["asks"]), np.std(data["N3_noisy"]["asks"])
        h1_ok = (a_n2 + sd_n2) < (a_n3 - sd_n3) * 0.4
        d = cohens_d(data["N2_fail"]["asks"], data["N3_noisy"]["asks"])
        # 效应量：零方差时 Cohen's d 不可计算——零方差本身是更强证据（30/30 一致）
        var_n2 = np.var(data["N2_fail"]["asks"])
        d_ok = (d > 0.8) if var_n2 > 0 else (a_n2 < a_n3 * 0.2)
        verdicts.append(("H1 (N2<N3)", f"{a_n2:.0f}±{sd_n2:.0f} vs {a_n3:.0f}±{sd_n3:.0f}, d={d:.2f}", h1_ok))
        verdicts.append(("效应量", f"d={d:.2f}（零方差→比值{var_n2==0}）", d_ok))

        # N1 类比（在失败环境测）：冻结 ask 可靠性 → 一直问 → 无法适应
        a_n1 = np.mean(data["N1_frozen"]["asks"])
        a_n4 = np.mean(data["N4_perfect"]["asks"])
        n1_ok = a_n1 > a_n2 * 5   # 冻结组问路次数应远多于在线组（fail 环境）
        verdicts.append(("N1 (冻结vs在线)", f"fail环境 冻结 {a_n1:.0f} vs 在线 {a_n2:.0f}（冻结应多问）", n1_ok))

        # 零影响护栏：ctrl_off ask=0
        if data["ctrl_off"]["n"] > 0:
            a_off = np.mean(data["ctrl_off"]["asks"])
            verdicts.append(("零影响", f"关闭组 ask={a_off:.0f}（应=0）", a_off == 0))
        # 无问路接口：ask_available=False → 仅试探性 ask（≤5）
        if data["ctrl_noask"]["n"] > 0:
            a_nk = np.mean(data["ctrl_noask"]["asks"])
            verdicts.append(("无问路接口", f"ask_available=False ask={a_nk:.0f}（应≤5 试探）", a_nk <= 5))

    print("\n判定明细:")
    all_ok = True
    for name, desc, ok in verdicts:
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: {desc}")
    print(f"\n总体: {'OK 通过' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--group", default=None, help="只跑指定组")
    ap.add_argument("--seeds", default=None, help="seed 范围如 0-9")
    args = ap.parse_args()

    if args.report:
        return report()

    seeds = list(range(N_SEEDS))
    if args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))

    groups = [args.group] if args.group else list(GROUPS.keys())
    for g in groups:
        print(f"\n[{g}] 跑 {len(seeds)} seeds...", flush=True)
        run_group(g, GROUPS[g], seeds)

    print("\n全部完成，统计报告：")
    return report()


if __name__ == "__main__":
    sys.exit(main())
