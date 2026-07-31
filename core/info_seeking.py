"""
信息寻求（Information Seeking）— 落差的定向搜索机制

核心（对应 DESIGN_GOALS.md 的 epistemic value）：
  信息寻求 ≠ 提高随机率。真正的信息寻求是：
    1. 知道自己不知道什么（"食物位置是未知量"——目标落差驱动）
    2. 系统性扫掠未探索区域（空间记忆标记已搜）
    3. 避免重复（不走回头路）

与恒定高探索（ε=0.8）的区别：
  - 恒定探索：均匀随机，10×10 上撞上角落目标的概率低（无记忆）
  - 定向扫掠：覆盖式搜索，每格最多访问一次直到找到目标（有记忆）

这使"目标表征 vs 高探索率"可被真正裁决（G6 vs G5）。
"""

import numpy as np


class InfoSeeker:
    """信息寻求器：落差驱动 → 定向扫掠（非随机探索）"""

    def __init__(self, grid_size: int = 16):
        self.grid_size = grid_size
        self.visited = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.search_origin = None     # 本次搜索起点
        self.sweep_direction = 0      # 当前扫掠方向（0-3: 上下左右）
        self.sweep_steps = 0          # 当前方向已走步数
        self.active = False           # 是否处于搜索模式
        self.search_goal = None       # 搜索目标（"资源"）

    def start_search(self, pos, goal_name: str = "resource"):
        """进入搜索模式（落差激活时调用）"""
        self.active = True
        self.search_goal = goal_name
        self.search_origin = tuple(pos)
        self.sweep_direction = 0
        self.sweep_steps = 0
        # 标记起点已访问
        if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
            self.visited[pos[0], pos[1]] += 1

    def stop_search(self):
        """搜索完成（找到目标）"""
        self.active = False
        self.search_goal = None

    def _mark_visited(self, pos):
        if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
            self.visited[pos[0], pos[1]] += 1

    def _is_visited(self, pos) -> bool:
        if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
            return self.visited[pos[0], pos[1]] > 0
        return True  # 越界视为已访问（不能走）

    def _best_unvisited_direction(self, pos) -> int:
        """四个方向中未访问次数最少的方向（优先未探索）"""
        dirs = [(0, -1), (-1, 0), (1, 0), (0, 1)]  # 上左右下
        best_dir, best_score = None, 1e9
        for d, (dx, dy) in enumerate(dirs):
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                score = self.visited[nx, ny]
                if score < best_score:
                    best_score = score
                    best_dir = d
        return best_dir if best_dir is not None else 0

    def choose_action(self, pos) -> int:
        """搜索模式下选动作（0=stay 1=up 2=left 3=right 4=down）"""
        if not self.active:
            return 0
        self._mark_visited(pos)
        # 1. 优先未访问方向（定向扫掠核心）
        d = self._best_unvisited_direction(pos)
        # 动作映射：up→1, left→2, right→3, down→4
        action_map = {0: 1, 1: 2, 2: 3, 3: 4}
        action = action_map.get(d, 4)
        self.sweep_steps += 1
        return action

    def coverage(self) -> float:
        """已探索覆盖率（用于报告）"""
        total = self.grid_size * self.grid_size
        return float(np.sum(self.visited > 0)) / total
