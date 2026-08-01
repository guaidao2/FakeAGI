"""
④ 重启 — 特征层迁移 v2（用概念迁移经验重做诚实负结果）

原 ④ 教训（朋友 stacked-deck 批评）：
  源 W_h 手工造对角（好投影器）vs 从头 randn*0.01（坏投影器）——
  7 倍优势来自"预训练好特征 > 随机坏特征"，不是"能力复用"。
  且基线被刻意削弱（从头模型没有机会学 W_h）。

v2 设计（防堆牌关键）：
  - W_h 由域 A **真实训练**获得（自监督时序预测收敛）——不是手工造
  - 从头组初始 W_h = randn*0.1（与域 A 训练前同分布）——不削弱
  - 两组**同样微调预算**（同 epochs 全模型微调，W_h 都参与学习）
  - 域 A/B 用关系特征空间（RelationalEnv 8D——⑤ 验证过的同构域对）
  - 3 独立训练 seeds + 评估布局分离 + 判定取最差
  - 判据：少样本后迁移组域 B 预测误差 < 从头组（能力复用证据）

若迁移不占优 → 诚实负结果（"特征层迁移无增益"，与 ⑤ 策略层对比——
策略层成立、特征层不成立，是更细的边界）。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_transfer_analogy import RelationalEnv


def train_domain_A(episodes=500, max_steps=40, seed=0, d_in=8, k=16,
                   lr=0.005):
    """域 A（迷宫）自监督时序预测训练——真实学出 W_h 特征提取器
    目标：预测下一时刻观测（关系特征空间的自监督动力学建模）"""
    rng = np.random.RandomState(seed)
    Wh = rng.randn(d_in, k) * 0.1
    theta = rng.randn(k, d_in) * 0.1
    for ep in range(episodes):
        env = RelationalEnv(size=6, mode="maze", seed=seed * 1000 + ep)
        obs = env.observe()
        for _ in range(max_steps):
            a = env.rng.randint(1, 5)  # 随机探索（自监督无需策略）
            _, _ = env.step(a)
            obs_next = env.observe()
            if env.done():
                break
            # 一步预测：h = tanh(o @ Wh); pred = h @ theta; 目标 = o_next
            h = np.tanh(obs @ Wh)
            pred = h @ theta
            err = pred - obs_next
            d_theta = np.outer(h, err)
            dh = err @ theta.T
            d_Wh = np.outer(obs, dh * (1.0 - h ** 2))
            theta -= lr * d_theta
            Wh -= lr * d_Wh
            obs = obs_next
    return Wh, theta


def eval_domain_B(Wh_init, theta_init, epochs=200, max_steps=40, seed=0,
                  d_in=8, k=16, lr=0.005):
    """域 B（威胁场）少样本微调 + 评估（预测误差）
    迁移组：Wh_init=域A训练；从头组：Wh_init=随机（同分布）"""
    rng = np.random.RandomState(seed + 999)
    Wh = Wh_init.copy()
    theta = rng.randn(k, d_in) * 0.1 if theta_init is None \
        else theta_init.copy()
    for ep in range(epochs):
        env = RelationalEnv(size=6, mode="threat", seed=seed * 1000 + ep + 500)
        obs = env.observe()
        for _ in range(max_steps):
            a = env.rng.randint(1, 5)
            _, _ = env.step(a)
            obs_next = env.observe()
            if env.done():
                break
            h = np.tanh(obs @ Wh)
            pred = h @ theta
            err = pred - obs_next
            d_theta = np.outer(h, err)
            dh = err @ theta.T
            d_Wh = np.outer(obs, dh * (1.0 - h ** 2))
            theta -= lr * d_theta
            Wh -= lr * d_Wh
            obs = obs_next
    # 评估：30 个新布局的预测误差
    total_err, total_n = 0.0, 0
    for t in range(30):
        env = RelationalEnv(size=6, mode="threat", seed=5000 + t * 7)
        obs = env.observe()
        for _ in range(max_steps):
            a = env.rng.randint(1, 5)
            _, _ = env.step(a)
            obs_next = env.observe()
            if env.done():
                break
            h = np.tanh(obs @ Wh)
            pred = h @ theta
            total_err += float(np.mean((pred - obs_next) ** 2))
            total_n += 1
            obs = obs_next
    return total_err / max(1, total_n)


def main():
    np.random.seed(42)
    print("=" * 60)
    print("④ 重启 — 特征层迁移 v2（真实训练 W_h，防堆牌）")
    print("=" * 60)
    seeds = (42, 7, 2026)
    rows = []
    for s in seeds:
        # 域 A 真实训练（每 seed 独立）
        Wh_src, theta_src = train_domain_A(episodes=500, seed=s)
        # 迁移组：域 A W_h + 域 B 微调
        err_mig = eval_domain_B(Wh_src, theta_src, seed=s)
        # 从头组：随机 W_h（同分布 randn*0.1）+ 同预算微调
        rng = np.random.RandomState(s)
        Wh_rand = rng.randn(8, 16) * 0.1
        err_scratch = eval_domain_B(Wh_rand, None, seed=s)
        rows.append((s, err_mig, err_scratch))
    err_mig_max = max(r[1] for r in rows)
    err_scratch_min = min(r[2] for r in rows)
    ok = err_mig_max < err_scratch_min  # 迁移最差仍优于从头最好
    detail = ", ".join(f"seed{r[0]}: 迁移{r[1]:.4f} vs 从头{r[2]:.4f}"
                       for r in rows)
    print(f"[B] 少样本预测误差(×3seeds): {detail}")
    print(f"    迁移最差 {err_mig_max:.4f} vs 从头最好 {err_scratch_min:.4f} "
          f"(应迁移<从头) {'OK' if ok else 'FAIL'}")
    verdict = ("通过——特征层迁移成立（域A真实训练能力复用）" if ok
               else "未过——特征层迁移无增益（与⑤策略层对比：策略层成立特征层不成立）")
    print(f"  判定: {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
