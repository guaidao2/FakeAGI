"""
LNN 核心 (Liquid Time Constant Network)

基于 Hasani 2019 的 LTC 细胞实现。
LTCell 具备自适应时间常数 τ，使其能处理不同时间尺度的时序模式。
"""

import torch
import torch.nn as nn
import numpy as np


class LTCell(nn.Module):
    """液体时间常数细胞"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.W_x = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
    def forward(self, x, h, dt=0.1):
        tau_input = torch.cat([x, h], dim=-1)
        tau = self.tau_net(tau_input) * 49.0 + 1.0  # τ ∈ [1, 50]
        dh = -h / tau + torch.tanh(self.W_x(x) + self.W_h(h))
        h_new = h + dt * dh
        return h_new, tau
    
    def expand(self, new_hidden: int):
        """生长时扩展隐藏层维度"""
        old_h = self.hidden_dim
        dev = self.W_x.weight.device
        new_in = new_hidden
        
        # 克隆旧权重到 CPU（新建的 Linear 默认在 CPU 上）
        old_Wx_weight = self.W_x.weight.data.clone().cpu()
        old_Wh_weight = self.W_h.weight.data.clone().cpu()
        
        self.W_x = nn.Linear(new_in, new_hidden, bias=False)
        self.W_h = nn.Linear(new_hidden, new_hidden, bias=False)
        self.tau_net = nn.Sequential(
            nn.Linear(new_hidden + new_hidden, new_hidden),
            nn.Sigmoid(),
        )
        
        with torch.no_grad():
            h_in = min(old_h, new_in)
            self.W_x.weight[:old_h, :h_in] = old_Wx_weight[:, :h_in]
            self.W_h.weight[:old_h, :old_h] = old_Wh_weight
        
        self.hidden_dim = new_hidden
        self.input_dim = new_in
        self.to(dev)


class LNN(nn.Module):
    """LNN 时序推理引擎"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, dt: float = 0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.ltc = LTCell(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        self.dt = dt
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
    
    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(1, self.hidden_dim, device=x.device)
        encoded = torch.tanh(self.encoder(x))
        h_new, tau = self.ltc(encoded, h, self.dt)
        out = self.output_layer(h_new)
        return out, h_new, tau
    
    def grow(self, new_hidden: int = None):
        """生长隐藏层（增量式：每次只长 GROW_STEP 个神经元，像海马体新生）"""
        old_h = self.hidden_dim
        dev = self.encoder.weight.device
        # 增量生长：每次 +8 神经元（而非批量 ×1.2），新神经元稀疏连接
        grow_step = 8
        if new_hidden is not None:
            new_h = new_hidden
        else:
            new_h = min(256, old_h + grow_step)  # 增量式上限 256
        if new_h <= old_h:
            return
        old_enc = self.encoder.weight.data.clone().cpu()
        old_out = self.output_layer.weight.data.clone().cpu()
        self.ltc.expand(new_h)
        self.encoder = nn.Linear(self.input_dim, new_h)
        self.output_layer = nn.Linear(new_h, new_h)
        with torch.no_grad():
            self.encoder.weight[:old_h, :] = old_enc
            self.output_layer.weight[:old_h, :old_h] = old_out
            # 新神经元稀疏初始化：只随机连接少量旧神经元（稀疏突触生长）
            n_new = new_h - old_h
            rng = np.random.default_rng()
            for i in range(n_new):
                row = old_h + i
                # 从旧神经元中随机选 ~20% 建立初始连接（其余为 0）
                n_conn = max(2, int(old_h * 0.2))
                conns = rng.choice(old_h, size=n_conn, replace=False)
                vals = torch.tensor(rng.uniform(-0.1, 0.1, n_conn).astype(np.float32))
                self.output_layer.weight[row, conns] = vals
                enc_vals = torch.tensor(
                    rng.uniform(-0.05, 0.05, self.input_dim).astype(np.float32))
                self.encoder.weight[row, :] = enc_vals * 0.1
        self.hidden_dim = new_h
        self.to(dev)
        print(f"  [GROW] {old_h}→{new_h} hidden (+{n_new} 增量)", flush=True)
    
    def grow_input(self, new_input_dim: int):
        """扩展感知输入维度（保留已有权重）"""
        dev = self.encoder.weight.device
        old_in = self.input_dim
        old_w = self.encoder.weight.data.clone().cpu()
        self.encoder = nn.Linear(new_input_dim, self.hidden_dim)
        self.input_dim = new_input_dim
        with torch.no_grad():
            h = min(self.hidden_dim, old_w.shape[0])
            w = min(new_input_dim, old_in)
            self.encoder.weight[:h, :w] = old_w[:h, :w]
        self.to(dev)
        print(f"  [GROW_INPUT] {old_in}→{new_input_dim} dims", flush=True)
