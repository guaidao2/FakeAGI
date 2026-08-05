# FakeAGI vs pymdp（Active Inference）基线对比

> 外部评审指出：本系统的"预测误差=唯一信号"与 active inference 同构
> （Friston 自由能原理）——需要 pymdp 基线建立坐标系。
> 本实验回应：**建立坐标系，定位差异，而非证明胜负。**

## 实验设计

**环境**（两系统同一接口）：5x5 网格 + 食物/水 + 稳态需求（能量/水递减）
+ **规则变化**（tick 300 食物 [0,0]→[4,4]）

| 参数 | 值 |
|------|-----|
| 判据 | 存活 tick / 变化前食物 / 变化后食物（适应速度）|
| n | 5 seeds |
| 吃判定 | 邻近 <3 格 + 停留 a==0（两系统公平）|
| 变化点 | tick 300 |

**系统配置（公平对标）**：
- **FakeAGI**：完整管线（LNN+世界模型+GameNN+概念）——观测=连续方向向量
- **pymdp**（inferactively-pymdp 1.0.3，jax 版）：active inference——
  A=食物在 [0,0] 先验似然（感知模型）、B=确定性转移（先验结构，
  对标 FakeAGI 本能）、C=偏好食物+水（对标 FakeAGI V 价值）、
  在线学习 infer_parameters（lr_pA=lr_pB=0.5）

## 结果（5 seeds 均值）

| 指标 | FakeAGI | pymdp |
|------|---------|-------|
| 存活 tick | 600（活满）| 600（活满）|
| 变化前食物 | ~110（91-130）| 295（站食物旁反复吃）|
| **变化后食物（适应）** | **0-60（seed1/2 部分适应）** | **0（未适应）** |

## 差异定位（坐标系——非胜负）

### 1. 剥削（exploitation）：pymdp 更优
pymdp 的精确贝叶斯推断（变分后验）在已知环境高效剥削（295 次吃 vs
FakeAGI ~110——反射/概念引导的粗放移动）。**这证实外部评审的判断：
我们的预测误差机制是 active inference 的未数学化版本——它的基础
推理更精确。**

### 2. 适应（adaptation）：FakeAGI 部分占优
规则变化后 FakeAGI 有 2/5 seed 恢复进食（30-60 次）——多渠道误差
修正（世界模型/概念库/GameNN TD 并行更新）；pymdp 0/5 适应（仅
A 矩阵 Dirichlet 更新，300 tick 内未收敛——lr=0.5 敏感性待探索）。

### 3. 未验证/待探索（诚实清单）
- pymdp lr_pA 敏感性（更高学习率能否在 300 tick 内适应）
- pymdp 只学 A（不学 B）是否更稳
- FakeAGI 适应波动的根因（seed 依赖——概念形成时机）
- 更长时间窗（600→2000 tick）两者适应曲线

## 对 FakeAGI 的启示（外部评审的正面转化）

1. **④ 公理确实 = 变分自由能**——FakeAGI 的误差修正机制应引入
   **精度加权**（precision-weighted prediction error——Friston）：
   世界模型当前无精度估计（老师第 1 点批评）
2. **主动感知缺失**（老师第 1 点）——"行动=降低不确定性"vs
   "行动=改变环境"双通路——FakeAGI 只有后者
3. **剥削效率提升路径**——概念匹配+预测驱动已接近，但离散符号
   索引（⑧）尚未接入决策精度

## 环境安装备注

- PyPI `pymdp` 是无关 MDP 包；官方库为 `inferactively-pymdp`
  （PyPI 1.0.3 = jax 版；numpy 经典版 0.0.7 不在 PyPI）
- **jax 版安装曾破坏 torch DLL**（WinError 193）——重装
  `torch-2.7.1+cu128`（本地 wheel）恢复；jaxlib 0.10.2 与
  torch 2.7.1 共存正常
- jax 版 API 缺陷（beta）：`sample_action` vmap tuple bug——
  用 q_pi argmax 手动选策略绕过；`update_A/update_B` 改为
  `infer_parameters`

## 文件

- `test_pymdp_baseline.py` — 对比实验
- `probe_pymdp_api.py` — API 探测（可删）
- `probe_torch.py` — torch 健康检查（可删）

## 日期

2026-08
