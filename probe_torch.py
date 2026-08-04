"""验证 torch 是否被 jaxlib 安装破坏"""
try:
    import torch
    print("TORCH_OK", torch.__version__)
except Exception as e:
    print("TORCH_FAIL:", repr(e))
    raise SystemExit(1)
