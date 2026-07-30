# FakeAGI — 自维持认知架构

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**玄幕安全团队-guaidao2** 出品

FakeAGI 是一个基于自维持哲学的通用认知架构实验性实现。它没有外部奖励函数，没有预训练阶段，不依赖大语言模型——它只是在活着，并在这个过程中展现出智能。

---

## 核心理念

智能不是解决问题的能力。智能是系统维持自身存在的能力。

当前所有主流 AI 路线都问同一个问题："我要如何完成这个任务？"FakeAGI 问的是另一个问题："我要如何继续存在？"

完整的哲学推导见 [`PAPER.md`](PAPER.md)。

## 架构概览

```
五层认知（反射→学习→规划→想象→元认知）
        ↓
三层循环（自我层→认知层→物理层）
        ↓
十条哲学公理（物质-信息 → 价值注入）
```

## 快速开始

```bash
git clone https://github.com/guaidao2/FakeAGI.git
cd FakeAGI
pip install torch numpy pygame
python main.py --maze 12
```

### 控制键
| 键 | 功能 |
|----|------|
| 空格 | 暂停/继续 |
| R | 重置 |
| +/- | 加速/减速 |
| K | 神经网络可视化 |

### 运行实验

```bash
# 实验 1：新奇环境适应
python test_experiment1.py

# 实验 2：多驱动力冲突
python test_experiment2.py

# 实验 3：空间记忆利用
python test_experiment3.py

# 实验 4：因果推理（核心实验）
python test_experiment4.py

# 实验 5：反事实选择
python test_experiment5.py
```

## 实验结果

| 实验 | 测试能力 | 结果 |
|------|---------|------|
| E1 新奇适应 | 环境变更后的适应速度 | ⚠️ 边缘通过 |
| E2 多驱动力冲突 | 能量/水分同时短缺 | ✅ 通过 |
| E3 空间记忆利用 | 已知路径优化 | ✅ 40% 缩减 |
| **E4 因果推理** | **隐藏规则发现** | **✅ 正式通过** |
| E5 反事实选择 | 代价权衡 | ⚠️ 边缘通过 |

## 项目结构

```
agi/
├── core/               # 自我层 — 自模型、稳态、驱动力
├── cognition/          # 认知层 — LNN、世界模型、GameNN
│   ├── temporal/       #   时序推理（LNN + τ）
│   ├── decision/       #   多策略决策（GameNN）
│   ├── metacognition/  #   元认知（缺口检测+重定向）
│   └── ...             #   规划、想象、睡眠、空间记忆
├── growth/             # 生长引擎
├── test_experiment*.py # 六组验证实验
├── main.py             # 主入口
├── PAPER.md            # 论文
└── ARCHITECTURE.md     # 架构说明
```

## 技术栈

- **时序推理**：液体时间常数网络（LTC）
- **决策**：GameNN 多策略博弈矩阵（[GameNN-WorldModel](https://github.com/guaidao2/GameNN-WorldModel)）
- **世界模型**：条件预测（hidden+action→next_hidden）
- **生长**：自适应隐藏维度扩展（[Growing-LLM](https://github.com/guaidao2/Growing-LLM)）
- **元认知**：知识缺口检测 + 目标生成 + 反射压制

## 设计原则

1. **无外部奖励** — 唯一的训练信号是预测误差
2. **自维持优先** — 所有模块服务于存在概率最大化
3. **持续在线学习** — 无训练/部署割裂
4. **生长即学习** — 容量随复杂度自适应
5. **五层耦合** — 反射→学习→规划→想象→元认知

## License

MIT
