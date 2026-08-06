# AGI 项目差距审计报告

> **状态：历史记录（2026-07-31）** — 以下 10 个差距**已全部修复**，
> 现状以 `ARCHITECTURE.md` / `README.md` / `PAPER.md` 为准。
> 保留此文档作为开发历程记录。

## 总览：10 个差距，分为三级

## 🔴 阻断级（系统无法运转）

| # | 差距 | 文件 | 影响 |
|---|------|------|------|
| 1 | **cognition.process() 不存在** | cognition/__init__.py 为空 | LNN、GameNN、世界模型全链无法启动 |
| 2 | **世界模型未实现** | cognition/temporal/ 下无 world_model.py | 无预测 → 无惊奇 → 自维持循环中断 |
| 3 | **紧急响应为 pass** | main.py:90 _emergency_response | 存在概率归零时系统无应急行为 → 必死 |

## 🟡 严重级（效果大打折扣）

| # | 差距 | 文件 | 影响 |
|---|------|------|------|
| 4 | **自模型没联通认知层** | main.py:52 未传自模型状态给决策 | LNN 看不到"我快死了" → 决策无视生存信号 |
| 5 | **LNN 未实例化** | 无处调用 cognition/temporal/lnn.py | 整个时序推理模块存在但闲置 |
| 6 | **GameNN 学习函数为 pass** | cognition/decision/gamenn.py:42-44 | 策略不学习 → 博弈矩阵永远不会更新 |
| 7 | **生长引擎无数据接入** | growth/monitor.py 从未被调用 | should_grow() 永远 False |

## 🟢 优化级（本可以更好）

| # | 差距 | 文件 | 影响 |
|---|------|------|------|
| 8 | **反事实通道为空** | cognition/imagination/ 只有 __init__.py | 无想象能力 → 无因果推理 |
| 9 | **Homeostasis 未影响行为** | 告警从未传入决策层 | 稳态检查了但不改变行为 |
| 10 | **好奇心和探索/利用没联动** | 两条独立的 curiosity 路径未合并 | 探索预算被重复分配 |

## 修复优先级

```
     本周（优先）
     ├── 1. 实现 CognitionPipeline（串联全链）
     ├── 2. 实现世界模型（预测 → 惊奇）
     ├── 3. 自模型状态注入 LNN
     │
     下周
     ├── 4. GameNN 接入真实学习信号
     ├── 5. 紧急响应策略
     ├── 6. 生长引擎接入训练
     │
     下月
     ├── 7. 反事实通道（想象）
     ├── 8. Homeostasis→决策反馈
     └── 9. 好奇心规划一
```

---

# 接线审计（2026-08-06——底层代码彻底排查："零件装配"系统性死通路排查）

**背景**：30+ 认知模块单独通过单元测试，但主循环接线反复出问题（历史：B1 info
NameError、B3 gate 恒 1.0、override 死变量、reflex 缺水方向）。本次全链路审计
（静态追踪 + 300 tick 冒烟）一次性抓出：

## BLOCKING（数据真实性——已修）
- **B1** SelfModel.update() 零调用 → survival_prob 恒 1.0 → curiosity/boredom
  全失真 → 已接线（energy_delta 用 body 真实变化）+ 系数校准（2.0→0.5——
  surprise 真实流入后原系数把生存概率压死）
- **B2** Homeostasis 零调用 → alarms 恒空 → 已接入紧急检测段
- **B3** drives.update 收 surprise 恒 0（process 前执行）→ boredom 恒升（伪）→
  已修（传上一 tick last_surprise）+ boredom 重构（行为重复 repeat_ticks 驱动——
  撞墙/原地转=无聊，非 surprise 低=无聊）

## WARN（已处理）
- W1 行动通路随机探索被委员会覆盖 → 移至决策后生效
- W2 GameNN.learn 归因错位（建议动作 vs 执行动作）→ 加 action 参数
- W3 causal_error 恒 0 → error_path=="action" 时=surprise
- W4 物理直觉 prior_loss 未接 → 经验记录门控接入（设计路径仍待接）
- W5 hemin 分歧无消费点 → 门控消费接入
- W6/W10 有意设计标注（跨代价值系统/睡眠巩固压缩）
- W7/W8/W9 有意设计/半实现标注（叠加态门控忽略/attention 恒等/strategy 半实现）

## 科学影响（诚实记录）
- 短窗适应 600t：3/5 → 1/5——**原适应部分依赖伪 boredom**（surprise 恒 0 时
  boredom 恒升强制探索）——分论文一适应增量如实降级为"长时窗性质"（2000t 3/3）
- 长时窗第二增量（动态稳态）：3/3 保持（76/41/74）——**探索真实化不破坏**——
  且机制更真（生理欲望→目标导航 + §3.4 水动力学）
- 经验：**接线验证必须用"数据流入断言"（monkeypatch 实测），不能只看调用点存在**
