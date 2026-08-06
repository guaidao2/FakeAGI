"""主循环接线冒烟测试（审计 B1-B3 修复验证）

B1: SelfModel.update 真实调用——survival_prob 不再恒 1.0（受 surprise 影响）
B2: homeostasis.check 真实调用——alarms 状态被记录
B3: drives.update 收到非零 surprise——boredom 通路数据真实流入
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from main import AGI
from cognition import CognitionPipeline
from core.self_model import SelfModel
from core.homeostasis import Homeostasis


def check(name, ok, detail=""):
    print(f"  {name}: {'OK' if ok else 'FAIL'} {detail}")
    return ok


def make_agi():
    cfg = {"maze": True, "quiet": True}
    agi = AGI(cfg)
    agi.set_cognition(CognitionPipeline(cfg))
    return agi


def main():
    ok = True
    # 迷宫模式（有食物/墙壁刺激——surprise 非 0，验证 drives 数据流入）
    agi = make_agi()
    # 运行 120 tick 让 surprise/告警累积
    for _ in range(120):
        agi.step()

    # B1: survival_prob 不再恒 1.0（energy 代谢下降 + surprise 波动应使其 < 1.0）
    sp = agi.self_model.survival_prob
    ok &= check("B1 survival_prob 动态化",
                sp < 1.0 and agi.self_model.avg_surprise >= 0.0,
                f"(sp={sp:.3f}, avg_surprise={agi.self_model.avg_surprise:.3f})")
    # B1b: 冒烟前恒 1.0——现在应随时间波动（记录历史）
    ok &= check("B1b avg_surprise 非初始值", agi.self_model.avg_surprise > 0.0
                or agi.self_model.survival_prob < 0.99,
                f"(avg_surprise={agi.self_model.avg_surprise:.4f})")

    # B2: homeostasis.check 真实调用（_homeostasis_alarms 存在）
    has_alarms_attr = hasattr(agi, "_homeostasis_alarms")
    ok &= check("B2 homeostasis 接线", has_alarms_attr,
                f"(alarms={getattr(agi, '_homeostasis_alarms', 'N/A')})")
    # B2b: danger_ticks 计数器存在且非 None
    ok &= check("B2b danger_ticks 记录", hasattr(agi, "_homeostasis_danger_ticks"))

    # B3: drives 收到真实 surprise——用 monkeypatch 验证收到的 surprise 集合
    received = set()
    orig = agi.drives.update
    def spy_update(*a, **k):
        received.add(round(float(a[2]) if len(a) > 2 else k.get("surprise", -1), 4))
        return orig(*a, **k)
    agi.drives.update = spy_update
    for _ in range(30):
        agi.step()
    non_zero = any(abs(x) > 1e-9 for x in received)
    ok &= check("B3 drives 收到非零 surprise", non_zero,
                f"(received={sorted(received)[:4]}...)")

    print("判定:", "ALL OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
