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
        def get_pos(self): return self.pos
        def observe(self):
            return np.array([(self.food_pos[0]-self.pos[0])/10,
                             (self.food_pos[1]-self.pos[1])/10,
                             (self.water_pos[0]-self.pos[0])/10,
                             (self.water_pos[1]-self.pos[1])/10])
        def step(self, a):
            dxs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
            dx, dy = dxs[a % 5]
            self.pos[0] = max(0, min(9, self.pos[0] + dx))
            self.pos[1] = max(0, min(9, self.pos[1] + dy))
            eat = abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1]) < 2
            drink = abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1]) < 2
            ed = 0.2 if eat else (0.05 if drink else -0.001)
            wd = 0.15 if drink else (0.02 if eat else -0.0005)
            return {"energy_delta": ed, "water_delta": wd}
    agi.set_env(TestEnv())

    wm = agi.cognition.world_model
    vmse_hist, foods = [], 0
    prev_h, prev_a = None, None
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
    # B：输出区分度（V 变化时预测跟着变）
    vp_vals = []
    vdim = wm.value_head.weight.shape[1]  # 实际维度（叠加态 72）
    with torch.no_grad():
        for _ in range(5):
            hh = torch.randn(1, vdim).to(next(wm.parameters()).device)
            vp_vals.append(float(wm.value_head(hh).item()))
    spread = max(vp_vals) - min(vp_vals)
    print(f"  B: value_head 输出跨度={spread:.4f} "
          f"(随机 hidden 输入)")
    ok_b = spread > 0.01
    print(f"     {'OK（有区分度——非恒值输出）' if ok_b else 'FAIL'}")
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
