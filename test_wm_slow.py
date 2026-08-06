"""
B3 接线验证：世界模型慢副本 EMA + 稳态门控（meta-RL/Hubel，
DESIGN_CONCEPTS §7.5）

验证：
  A. EMA 追踪：多步训练后 shadow 接近当前权重（0.99 decay 滞后）
  B. 门控冻结：gate=0 时 shadow 不变（应激高/能量低保护已有表征）
  C. grow 后重注册：shadow 尺寸与模型一致（防维度不匹配）
"""
import sys
import torch
import torch.nn.functional as F
from cognition.temporal.world_model import WorldModel


def diff(a, b):
    return float(torch.abs(a - b).sum())


def run():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    torch.manual_seed(0)
    wm = WorldModel(input_dim=32, n_actions=5)

    # A：训练 50 步（固定目标让权重漂移），shadow 应滞后跟随
    h = torch.randn(4, 32)
    target = torch.randn(4, 32)
    act = torch.zeros(4, dtype=torch.long)
    for _ in range(50):
        wm.train_step(h, target, act, gate=1.0)
    d_after = diff(wm.shadow["predictor.0.weight"],
                   wm.predictor[0].weight)
    print(f"  A: 50 步后 shadow vs 当前 差={d_after:.4f} "
          f"(EMA 0.99 滞后应显著>0 但同量级)")
    ok_a = d_after > 1e-6 and d_after < diff(
        wm.shadow["predictor.0.weight"], torch.zeros_like(
            wm.shadow["predictor.0.weight"]))
    print(f"     {'OK（EMA 慢副本追踪）' if ok_a else 'FAIL'}")

    # B：gate=0 冻结——记录 shadow，再训练 10 步 gate=0，shadow 不变
    before = wm.shadow["predictor.0.weight"].clone()
    for _ in range(10):
        wm.train_step(h, target, act, gate=0.0)
    d_gate = diff(before, wm.shadow["predictor.0.weight"])
    print(f"  B: gate=0 训练 10 步 shadow 变化={d_gate:.6f}")
    ok_b = d_gate == 0.0
    print(f"     {'OK（稳态门控冻结慢学习）' if ok_b else 'FAIL'}")

    # C：grow 后 shadow 重注册
    wm.grow(48)
    shape_ok = (wm.shadow["predictor.0.weight"].shape
                == wm.predictor[0].weight.shape)
    print(f"  C: grow 后 shadow 形状 {wm.shadow['predictor.0.weight'].shape}"
          f" vs 模型 {wm.predictor[0].weight.shape}")
    ok_c = shape_ok
    print(f"     {'OK（生长后重注册）' if ok_c else 'FAIL'}")

    # D：设备移动后首步 EMA 不崩（review blocking 修复验证——
    # 原 __init__ CPU 注册 + .to(cuda) 后首步必 RuntimeError）
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        wm2 = WorldModel(input_dim=32, n_actions=5)
        wm2.to(dev)
        h2 = torch.randn(4, 32, device=dev)
        t2 = torch.randn(4, 32, device=dev)
        a2 = torch.zeros(4, dtype=torch.long, device=dev)
        wm2.train_step(h2, t2, a2, gate=1.0)  # 首步：设备不匹配→惰性重注册
        # 比较用 .type（torch.device('cuda') != 'cuda:0' 不归一化）
        ok_d = wm2.shadow["predictor.0.weight"].device.type == dev
        print(f"  D: {dev} 首步 train_step——shadow device="
              f"{wm2.shadow['predictor.0.weight'].device}")
        print(f"     {'OK（设备匹配惰性重注册）' if ok_d else 'FAIL'}")
    except Exception as e:
        print(f"  D: FAIL——{e}")
        ok_d = False

    ok = ok_a and ok_b and ok_c and ok_d
    verdict = ("OK（B3 接线完成：慢副本 EMA + 稳态门控 + 生长重注册）"
               if ok else "FAIL")
    print("\n判定: " + verdict)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
