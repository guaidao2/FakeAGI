"""
收敛重验（归一化修复后）：value_head 是否真收敛（非饱和伪象）

review blocking 修复：V 目标 (energy/2+water)/2∈[0,1]（原 (energy+water)/2
健康态恒 1.0——value_mse 收敛 0.013 是学恒 1 饱和伪象）。

验证：
  A. value_mse 随时间下降（真收敛——健康态 V≈0.75 非恒 1）
  B. value_head 输出有区分度（不同身体状态预测不同 V）
  C. 行为：食物数（归一化修复后不退化）
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from main import AGI
from cognition import CognitionPipeline


def main():
    agi = AGI()
    agi.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))

    class TestEnv:
        def __init__(self):
            self.pos = [5, 5]
            self.food_pos = [2, 2]
            self.water_pos = [7, 7]
            self.tick = 0
        def get_pos(self): return self.pos
        def observe(self):
            return np.array([(self.food_pos[0]-self.pos[0])/10,
                             (self.food_pos[1]-self.pos[1])/10,
                             (self.water_pos[0]-self.pos[0])/10,
                             (self.water_pos[1]-self.pos[1])/10])
        def step(self, a):
            self.tick += 1
            dxs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
            dx, dy = dxs[a % 5]
            self.pos[0] = max(0, min(9, self.pos[0] + dx))
            self.pos[1] = max(0, min(9, self.pos[1] + dy))
            # 前 400 tick 无食物（饥饿期——制造 v_true 低值分布）
            food_active = self.tick >= 400
            eat = (food_active
                   and abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1]) < 2)
            drink = abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1]) < 2
            ed = 0.2 if eat else -0.002
            wd = 0.15 if drink else -0.001
            return {"energy_delta": ed, "water_delta": wd}
    agi.set_env(TestEnv())

    wm = agi.cognition.world_model
    vmse_hist, foods = [], 0
    prev_h, prev_a = None, None
    paired = []  # should-fix B：收集 (pred, v_true) 对——状态对比用
    for t in range(800):
        before = agi.body.energy
        agi.step()
        if agi.body.energy > before + 0.01:
            foods += 1
        if prev_h is not None and hasattr(wm, 'value_head') and wm.value_head is not None:
            with torch.no_grad():
                pred = wm.predict(prev_h.detach(), prev_a)
                vp = wm.value_head(pred.detach())
                v_true = (agi.body.energy/2.0 + agi.body.water)/2.0
                vmse = float(F.mse_loss(vp.squeeze(-1),
                                        torch.tensor([v_true]).to(vp.device)))
                vmse_hist.append(vmse)
                paired.append((float(vp.item()), float(v_true)))
        prev_h = agi.cognition.hidden
        prev_a = torch.zeros(1, dtype=torch.long)

    if len(vmse_hist) < 50:
        print("  FAIL: value_head 未激活")
        return 1
    # A：value_mse 前后对比
    half = len(vmse_hist) // 2
    early_v = np.mean(vmse_hist[:half])
    late_v = np.mean(vmse_hist[half:])
    print(f"  A: value_mse early={early_v:.4f} late={late_v:.4f} "
          f"({early_v/max(late_v,1e-9):.2f}x)")
    ok_a = late_v < early_v * 0.7
    print(f"     {'OK（value_head 真收敛——非饱和伪象）' if ok_a else 'FAIL'}")
    # B：输出区分度（should-fix：随机 hidden 判据无判别力——
    # 改真实轨迹上 v_true 高低分组对比（分位数——water 恒满会拖底
    # 绝对阈值，相对分组更稳））
    if paired:
        vts = sorted(vt for _, vt in paired)
        n = len(vts)
        hi_th = vts[int(n * 0.7)]
        lo_th = vts[int(n * 0.3)]
        hi = [p for p, vt in paired if vt > hi_th]
        lo = [p for p, vt in paired if vt < lo_th]
        hi_m = np.mean(hi) if hi else 0.0
        lo_m = np.mean(lo) if lo else 0.0
        print(f"  B: 高价值态预测={hi_m:.3f} (n={len(hi)}, "
              f"v>={hi_th:.2f}) 低价值态预测={lo_m:.3f} "
              f"(n={len(lo)}, v<={lo_th:.2f})")
        ok_b = hi_m > lo_m + 0.03 and len(hi) > 0 and len(lo) > 0
        print(f"     {'OK（高价值态预测显著高于低价值态——有区分度）' if ok_b else 'FAIL'}")
    else:
        ok_b = False
        print("  B: FAIL（无配对数据）")
    print(f"  C: 食物={foods} 次/800 tick")
    ok_c = foods >= 3
    print(f"     {'OK（行为不退化）' if ok_c else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    verdict = ("OK（收敛重验通过：value_head 真收敛 + 有区分度 + 行为不退化）"
               if ok else "FAIL")
    print("\n判定: " + verdict)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
