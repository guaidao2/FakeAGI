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
人脑式决策委员会（5决策者并行投票 + 加权仲裁）
        ↓
三层循环（自我层→认知层→物理层）
        ↓
增量神经元生长（每次+8，稀疏连接，权重迁移）
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

### 接入世界温室（world-sim）

FakeAGI 完整认知架构可作为决策核心接入 [world-sim](https://github.com/guaidao2/world-sim) 生态沙盘（村庄/资源/昼夜/野兽/敌对部落）：

```bash
cd world-sim
python main.py   # AGIBeing 自动加载 FakeAGI 核心
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
| E2 多驱动力冲突 | 能量/水分同时短缺 | ✅ 三场景全部存活 |
| E3 空间记忆利用 | 已知路径优化 | ✅ 40% 缩减 |
| **E4 因果推理** | **隐藏规则发现** | **✅ 正式通过** |
| E5 反事实选择 | 代价权衡 | ⚠️ 边缘通过（食物优先，生存逻辑） |

## 项目结构

```
agi/
├── core/                    # 自我层
│   ├── body.py              #   身体模型（6维稳态+昼夜节律）
│   ├── drives.py            #   驱动力系统（6驱动力竞争）
│   ├── self_model.py        #   自模型（存在概率）
│   ├── homeostasis.py       #   稳态监控
│   ├── value_system.py      #   可进化价值系统（核心不变+次级可调）
│   └── physics_intuition.py #   物理直觉（重力/碰撞/瞬移先验）
├── cognition/               # 认知层
│   ├── __init__.py          #   认知管线（LNN+WM+GameNN+生长检测）
│   ├── temporal/            #   时序推理
│   │   ├── lnn.py           #     LNN + τ（增量生长+稀疏连接）
│   │   ├── world_model.py   #     条件世界模型（hidden+action→next）
│   │   └── tau_adapt.py     #     τ自适应
│   ├── decision/
│   │   ├── gamenn.py        #   多策略博弈决策（GameNN）
│   │   └── committee.py     #   人脑式决策委员会（5决策者+加权仲裁）
│   ├── metacognition/
│   │   ├── core.py          #   元认知层（缺口检测+目标生成）
│   │   ├── __init__.py      #   GapDetector（快速失败检测）
│   │   ├── goal_gen.py      #   探索目标生成
│   │   ├── scheduler.py     #   好奇心调度
│   │   ├── assess.py        #   自我评估
│   │   └── strategy_manager.py # 元-元认知（学习策略切换）
│   ├── planner.py           #   规划（BFS前瞻模拟）
│   ├── imagination_channel.py # 反事实想象通道
│   ├── concept_bank.py      #   概念库（组合式反事实）
│   ├── latent_state.py      #   隐变量模型（不可观测因素推断）
│   ├── attention.py         #   注意力门控（选择性感知）
│   ├── spatial_memory.py    #   空间记忆
│   ├── sleep.py             #   睡眠巩固
│   ├── hemin.py             #   他者模型（影子自我）
│   └── danger.py            #   危险感知
├── growth/                  # 生长引擎（容量监控）
├── test_experiment*.py      # 六组验证实验
├── main.py                  # 主入口
├── PAPER.md                 # 论文
└── ARCHITECTURE.md          # 架构说明
```

## 技术栈

- **时序推理**：液体时间常数网络（LTC），τ ∈ [1, 50] 自适应
- **决策**：人脑式决策委员会 —— 反射/边缘/习惯/规划/元认知 5 路并行投票，加权仲裁，冲突→深思模式，危急→恐慌模式
- **世界模型**：条件预测（hidden+action→next_hidden），误差通路选择（感知/行动）
- **生长**：增量式神经元生长（每次 +8，稀疏突触连接，tau/encoder/bias 权重迁移），输入维度自动扩展
- **元认知**：知识缺口检测 + 探索目标生成 + 反射压制 + 学习策略切换
- **价值进化**：核心价值（存续优先）不可变，次级价值（好/坏判断）由经验调整
- **物理直觉**：物理先验置信度（贝叶斯式经验修正），瞬移检测→应激

## 设计原则

1. **无外部奖励** — 唯一的训练信号是预测误差
2. **自维持优先** — 所有模块服务于存在概率最大化
3. **持续在线学习** — 无训练/部署割裂，AGI 是"活"出来的不是训练出来的
4. **增量生长** — 神经元每次少量新生，旧知识权重迁移保留
5. **并行决策** — 多决策系统竞争+加权仲裁，而非单一模块顺序覆盖
6. **五层耦合** — 反射→学习→规划→想象→元认知

## License

MIT
