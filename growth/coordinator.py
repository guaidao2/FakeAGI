"""
P4: 统一生长协调器 — 全链路同步生长

生命体是全局增长的：感官→神经通路→决策→身体，各层协调生长。
当前架构各模块独立 grow（LNN/世界模型各自为政），观测变化是
"被动补丁"而非"主动协调"。

本模块：
  1. 各模块注册自己的生长接口（grow_fn, current_dim_fn, prune_fn）
  2. 观测维度变化 → 触发一次"全链路协调生长事件"
  3. 生长由信息增益驱动，有上限，可剪枝回收
  4. 每层按自己的节奏增长，但保持协调（不成断层）

设计原则（来自哲学第⑨条）：
  - 增长由信息增益驱动（新信号源证明有用才长）
  - 增长有上限（资源约束）
  - 增长可回收（长期无用的通道剪枝）
  - 增长保持协调（各层一起长）
"""

import os
import numpy as np


class GrowthCoordinator:
    def __init__(self, max_hidden: int = 256, log: bool = True):
        self.max_hidden = max_hidden
        self.log = log
        self.modules = {}      # name -> {grow_fn, dim_fn, prune_fn, growth_count}
        self.growth_events = 0
        self.pruned = 0
        self.history = []      # [(tick, event, detail)]
        self.last_growth_tick = {}

    def register(self, name: str, grow_fn, dim_fn=None, prune_fn=None):
        """注册一个可生长模块"""
        self.modules[name] = {
            "grow_fn": grow_fn,
            "dim_fn": dim_fn or (lambda: 0),
            "prune_fn": prune_fn,
            "growth_count": 0,
        }
        if self.log:
            print(f"[GROWTH] 模块注册: {name}")

    def unregister(self, name: str):
        self.modules.pop(name, None)

    def sync_to(self, target_dim: int, source: str = "coordinator"):
        """
        全链路协调生长：把指定模块生长到 target_dim，
        并让其他模块同步扩展（保持维度协调）。
        返回实际发生生长的模块列表。
        """
        grown = []
        for name, m in self.modules.items():
            cur = m["dim_fn"]() if m["dim_fn"] else 0
            if cur < target_dim:
                try:
                    m["grow_fn"](target_dim)
                    m["growth_count"] += 1
                    self.last_growth_tick[name] = self.growth_events
                    grown.append(name)
                except Exception as e:
                    print(f"[GROWTH] {name} 生长失败: {e}")
        if grown:
            self.growth_events += 1
            self.history.append((self.growth_events, source, grown))
            if self.log:
                print(f"[GROWTH] 协调生长 #{self.growth_events} "
                      f"({source}): {grown} → {target_dim}", flush=True)
        return grown

    def prune(self, min_dim: int = 4):
        """
        剪枝：各模块回收长期低信息通道。
        返回被剪枝的模块名。
        """
        pruned = []
        for name, m in self.modules.items():
            if m["prune_fn"]:
                try:
                    if m["prune_fn"](min_dim):
                        self.pruned += 1
                        pruned.append(name)
                except Exception:
                    pass
        if pruned and self.log:
            print(f"[GROWTH] 剪枝: {pruned}")
        return pruned

    def get_state(self) -> dict:
        return {
            "growth_events": self.growth_events,
            "pruned": self.pruned,
            "module_dims": {name: m["dim_fn"]() for name, m in self.modules.items()},
            "growth_counts": {name: m["growth_count"] for name, m in self.modules.items()},
            "recent_events": self.history[-5:],
        }

    def get_state_dict(self) -> dict:
        return {
            "growth_events": self.growth_events,
            "pruned": self.pruned,
            "last_growth_tick": self.last_growth_tick,
        }

    def load_state_dict(self, sd: dict):
        self.growth_events = sd.get("growth_events", 0)
        self.pruned = sd.get("pruned", 0)
        self.last_growth_tick = sd.get("last_growth_tick", {})
