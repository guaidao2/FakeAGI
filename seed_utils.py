"""统一 seed 基建（任务 ②——run 级可复现性）

问题背景：test_experiment1/3/4/5 等脚本无 seed → 单次跑数字波动
（E1 236→57、E3 90%→12%——论文数字无法定稿）；部分脚本只 seed
np/torch 漏 Python random → 模块内 random.choice 不可复现。

用法：
  from seed_utils import seed_run
  seed_run(args.seed)          # 三模块统一 seed（np/torch/random）
  seed_run(args.seed, base=42) # 自定义基

约定：
  SEED_BASE = 1000（与历史 test_pymdp_baseline 的 1000+seed 一致）
  同一 (base, seed) 必须产出同一运行轨迹——可复现性判据
"""
import os
import random

import numpy as np
import torch

SEED_BASE = 1000


def seed_run(seed: int, base: int = SEED_BASE) -> int:
    """统一 seed 入口：np.random / torch / Python random 三模块。
    返回实际使用的全局 seed（base+seed）——供日志/文档记录。"""
    s = base + int(seed)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    random.seed(s)
    return s


def get_seed_from_env(default: int = 0) -> int:
    """从环境变量 SEED 取 seed（实验脚本统一入口）。
    AGI_QUIET=1 风格：`SEED=3 python test_x.py`。"""
    try:
        return int(os.environ.get("SEED", str(default)))
    except ValueError:
        return default
