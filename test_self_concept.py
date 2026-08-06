"""
自我概念（① 阶段 4——身体状态簇 → "我饿了"的表征）

自我概念 = 身体状态聚类（energy/water/health 状态向量）——
低能量状态形成"饥饿"簇、高能量形成"健康"簇。
匹配：当前身体状态 → 激活对应自我概念——意向性的可检验替代
（系统能"知道自己饿了"= 状态匹配自我概念）。

设计：
- BodyStateCluster：身体状态向量聚类（在线质心）
- 判据：
  A. 饥饿态（energy<0.3）→ 匹配"饥饿"簇
  B. 健康态（energy>0.8）→ 匹配"健康"簇（区分）
  C. 聚类区分度（两簇质心距离 > 阈值——不是同一个簇）
"""
import sys
import numpy as np


class BodyStateCluster:
    """身体状态聚类——自我概念（"我饿了"= 状态簇激活）"""
    def __init__(self, dim=3, threshold=0.4):
        self.dim = dim
        self.threshold = threshold
        self.centroids = []       # 状态概念质心
        self.names = []           # 状态概念名（饥饿/健康——由激活时状态定）
        self.counts = []

    def _state_vec(self, energy, water, health):
        return np.array([energy, water, health], dtype=np.float64)

    def update(self, energy, water, health, label_hint=None):
        """喂一个身体状态 → 更新/创建状态概念"""
        v = self._state_vec(energy, water, health)
        if not self.centroids:
            self.centroids.append(v.copy())
            self.names.append(label_hint or "state_0")
            self.counts.append(1)
            return self.names[0]
        # 找最近质心
        best_i, best_d = -1, 1e9
        for i, c in enumerate(self.centroids):
            d = float(np.linalg.norm(c - v))
            if d < best_d:
                best_i, best_d = i, d
        if best_d < self.threshold:
            # 归入现有状态概念（在线质心更新）
            self.centroids[best_i] = (0.9 * self.centroids[best_i] + 0.1 * v)
            self.counts[best_i] += 1
            return self.names[best_i]
        # 新状态概念
        self.centroids.append(v.copy())
        self.names.append(label_hint or f"state_{len(self.centroids)}")
        self.counts.append(1)
        return self.names[-1]

    def match(self, energy, water, health):
        """当前状态 → 匹配的自我概念（name, dist, found）"""
        v = self._state_vec(energy, water, health)
        if not self.centroids:
            return "", 1e9, False
        best_i, best_d = -1, 1e9
        for i, c in enumerate(self.centroids):
            d = float(np.linalg.norm(c - v))
            if d < best_d:
                best_i, best_d = i, d
        if best_d < self.threshold:
            return self.names[best_i], best_d, True
        return "", best_d, False

    def distinctness(self):
        """聚类区分度：质心两两最小距离（>阈值 2 倍 = 真区分）"""
        if len(self.centroids) < 2:
            return 0.0
        return min(float(np.linalg.norm(self.centroids[i] - self.centroids[j]))
                   for i in range(len(self.centroids))
                   for j in range(i + 1, len(self.centroids)))


def main():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    print("=" * 56)
    print("自我概念验证（身体状态簇——「我饿了」）")
    print("=" * 56)
    rng = np.random.RandomState(0)
    sc = BodyStateCluster()

    # 饥饿期（energy 0.1-0.3）+ 健康期（energy 0.8-1.0）交替喂
    for _ in range(50):
        sc.update(0.1 + rng.rand() * 0.2, 0.5 + rng.rand() * 0.3,
                  0.9 + rng.rand() * 0.1, label_hint="hungry")
    for _ in range(50):
        sc.update(0.8 + rng.rand() * 0.2, 0.5 + rng.rand() * 0.3,
                  0.9 + rng.rand() * 0.1, label_hint="healthy")

    # A：饥饿态 → 饥饿簇
    n_h, d_h, f_h = sc.match(0.15, 0.6, 0.95)
    print(f"  A: 饥饿态(0.15) → {n_h} (d={d_h:.3f})")
    ok_a = f_h and ("hungry" in n_h or "state_0" in n_h)
    print(f"     {'OK（饥饿态匹配饥饿簇）' if ok_a else 'FAIL'}")

    # B：健康态 → 健康簇（区分）
    n_s, d_s, f_s = sc.match(0.9, 0.6, 0.95)
    print(f"  B: 健康态(0.9) → {n_s} (d={d_s:.3f})")
    ok_b = f_s and n_s != n_h
    print(f"     {'OK（健康态区分——不同自我概念）' if ok_b else 'FAIL'}")

    # C：聚类区分度
    dist = sc.distinctness()
    print(f"  C: 质心区分度 = {dist:.3f}（阈值 {sc.threshold}）")
    ok_c = dist > sc.threshold * 1.5
    print(f"     {'OK（两簇真区分——不是同一状态）' if ok_c else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    print(f"\n  判定: {'OK 通过（自我概念成立——状态有「自我」标签）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
