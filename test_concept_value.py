"""
概念层最小实验（DESIGN_CONCEPTS §3 阶段 1）：价值锚聚类

命题：概念 = 观测簇 × 价值绑定——"可消耗物"概念 = 使 V 上升的
观测的聚类。若聚类以"价值上升"为锚（而非外观），则未见过的
形态（外观全新但同样使 V 上升）应被识别为同类（跨形态泛化）。

之前路线 B 概念迁移失败的根源：无价值锚（MLP 学像素簇学不动，
学价值簇可行）。

验证：
  A. 聚类有效：训练形态的正样本聚成簇、负样本远离
  B. 跨形态泛化：测试形态 D（外观全新但 V 上升）正样本归入簇、
     负样本远离（判定：正样本到最近质心 < 负样本到最近质心）
"""
import sys
import numpy as np


class ValueAnchorCluster:
    """增量质心聚类——只聚类 V 上升事件（价值锚）"""
    def __init__(self, n_clusters=2, threshold=0.5):
        self.centroids = []  # list of np.array
        self.threshold = threshold

    def add(self, x, v_up: bool):
        """v_up=True 的事件更新最近质心；False 不更新（只用于评估）"""
        if not v_up:
            return
        if len(self.centroids) < 2:
            self.centroids.append(x.copy())
            return
        # 更新最近质心（在线 k-means）
        dists = [np.linalg.norm(x - c) for c in self.centroids]
        i = int(np.argmin(dists))
        self.centroids[i] = 0.9 * self.centroids[i] + 0.1 * x

    def dist(self, x):
        if not self.centroids:
            return 999.0
        return min(np.linalg.norm(x - c) for c in self.centroids)


VALUE_PATTERN = np.array([0.5, -0.5, 0.5, 0, 0, 0, 0, 0])  # 价值相关特征（共享）


def make_form(seed, n=60, dim=8):
    """生成一种形态：正样本（V 升）= 共享价值模式 + 形态专属外观；
    负样本（V 不升）= 纯随机（无价值模式）——生物学设定：
    可消耗物的"价值相关特征"共享（靠近→能量升），外观各异"""
    rng = np.random.RandomState(seed)
    appearance = rng.randn(dim) * 0.6  # 形态专属外观（各形态不同）
    appearance[3:] = appearance[3:] * 0.3  # 外观只在非价值维度
    appearance[:3] = 0.0                    # 价值维度由共享模式决定
    pos = VALUE_PATTERN + appearance + rng.randn(n, dim) * 0.15
    neg = rng.randn(n, dim) * 1.0  # 负样本：随机散点（无价值模式）
    return pos, neg


def run():
    # 3 种训练形态（外观各异）
    train_pos = []
    train_neg = []
    for s in [10, 20, 30]:
        p, n = make_form(s)
        train_pos.append(p)
        train_neg.append(n)
    cl = ValueAnchorCluster()
    for p in train_pos:
        for x in p:
            cl.add(x, True)

    # A：训练形态正样本距质心 vs 负样本距质心
    d_pos = np.mean([cl.dist(x) for p in train_pos for x in p[::6]])
    d_neg = np.mean([cl.dist(x) for n in train_neg for x in n[::6]])
    print(f"  A: 训练形态——正样本均距={d_pos:.3f} "
          f"负样本均距={d_neg:.3f}")
    ok_a = d_pos < d_neg * 0.5
    print(f"     {'OK（价值锚聚类有效）' if ok_a else 'FAIL'}")

    # B：测试形态 D（外观全新，V 上升）——跨形态泛化
    p_d, n_d = make_form(40)
    d_d_pos = np.mean([cl.dist(x) for x in p_d[::6]])
    d_d_neg = np.mean([cl.dist(x) for x in n_d[::6]])
    print(f"  B: 新形态 D——正样本均距={d_d_pos:.3f} "
          f"负样本均距={d_d_neg:.3f}")
    ok_b = d_d_pos < d_d_neg * 0.5
    print(f"     {'OK（跨形态泛化：新形态被识别为可消耗物）' if ok_b else 'FAIL'}")

    ok = ok_a and ok_b
    verdict = ("OK（概念层阶段 1 成立：价值锚聚类——未见过形态被识别"
               "为同类，概念=观测簇×价值绑定）" if ok else "FAIL")
    print("\n判定: " + verdict)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
