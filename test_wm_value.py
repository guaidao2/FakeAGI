"""
A 步骤验证：世界模型价值头（ΔV 预测——信号空转修复）

验证：
  A. 合成数据：ΔV 与 hidden 有规律关联 → value loss 显著下降
     （世界模型真的在学价值预测，非恒 0.5 兜底）
  B. 无回归：hidden 预测 loss 不被价值头破坏
  C. 生长后价值头重建（维度一致）
"""
import sys
import torch
import torch.nn.functional as F
from cognition.temporal.world_model import WorldModel


def run():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    torch.manual_seed(0)
    wm = WorldModel(input_dim=32, n_actions=5)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)

    # 构造有规律数据：dv = mean(h_next) 方向（价值与状态相关）
    h = torch.randn(16, 32)
    act = torch.zeros(16, dtype=torch.long)
    # 生成 target 与 dv（dv 由 target 状态能量方向决定——有内容）
    target = torch.randn(16, 32)
    dv = (target.mean(dim=-1, keepdim=True) * 0.5).clamp(-1, 1)

    v_hist, h_hist = [], []
    for step in range(300):
        # 手动训练（含价值头 loss）
        pred = wm.predict(h, act)
        loss_h = F.mse_loss(pred, target.detach())
        dv_pred = wm.value_head(pred.detach())
        loss_v = F.mse_loss(dv_pred, dv.detach())
        loss = loss_h + 0.5 * loss_v
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            v_hist.append(loss_v.item())
            h_hist.append(loss_h.item())

    print(f"  A: value loss: {v_hist[0]:.4f} → {v_hist[-1]:.4f}")
    ok_a = v_hist[-1] < v_hist[0] * 0.5
    print(f"     {'OK（价值头真实学习——ΔV 预测误差下降）' if ok_a else 'FAIL'}")
    print(f"  B: hidden loss: {h_hist[0]:.4f} → {h_hist[-1]:.4f}")
    ok_b = h_hist[-1] < h_hist[0] * 0.9
    print(f"     {'OK（hidden 预测未被破坏）' if ok_b else 'FAIL'}")

    # C：生长后价值头重建
    wm.grow(48)
    ok_c = wm.value_head.weight.shape == torch.Size([1, 48])
    print(f"  C: 生长后 value_head 形状 {wm.value_head.weight.shape}")
    print(f"     {'OK（生长重建）' if ok_c else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    verdict = ("OK（A 步骤完成：世界模型预测价值变化——信号有内容）"
               if ok else "FAIL")
    print("\n判定: " + verdict)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
