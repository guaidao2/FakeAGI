"""
空间记忆系统 — 增量构建环境地图

真正的动物会记住"哪里有食物、哪里有危险、哪里有水"。
它构建的不是精确像素地图，而是一个稀疏的、附着价值的
空间记忆网络。

结构：
  - 节点：位置(x, y) + 在该位置的经验（能量变化、惊奇、价值）
  - 边：两个节点之间的距离（用于路径规划）
  - 每个节点附带语义标签（"有食物"、"有危险"、"有水"）
"""

import numpy as np
import heapq


class SpatialMemory:
    def __init__(self, max_nodes=200):
        self.nodes = {}  # (x, y) → Node
        self.max_nodes = max_nodes
        self.current_pos = None
        self.direction_deltas = [(0, -1), (-1, 0), (1, 0), (0, 1)]
    
    def update_position(self, pos, energy_delta=0.0, surprise=0.0,
                        danger=False, food_nearby=False, water_nearby=False):
        """更新当前位置的记忆节点"""
        self.current_pos = pos
        
        if pos in self.nodes:
            node = self.nodes[pos]
            node.visit_count += 1
            node.last_seen = 0
            node.avg_energy_change = 0.9 * node.avg_energy_change + 0.1 * energy_delta
            node.avg_surprise = 0.9 * node.avg_surprise + 0.1 * surprise
            if danger:
                node.danger_count += 1
            if food_nearby:
                node.food_hint = True
            if water_nearby:
                node.water_hint = True
        else:
            # 创建新节点
            if len(self.nodes) >= self.max_nodes:
                # 淘汰最久未访问的节点
                oldest = min(self.nodes, key=lambda p: self.nodes[p].last_seen)
                del self.nodes[oldest]
            
            self.nodes[pos] = SpatialNode(
                pos=pos,
                energy_change=energy_delta,
                surprise=surprise,
                danger=danger,
                food=food_nearby,
                water=water_nearby,
            )
    
    def find_path(self, from_pos, to_pos, max_steps=50):
        """BFS 寻路"""
        if from_pos not in self.nodes or to_pos not in self.nodes:
            return None
        
        start, goal = from_pos, to_pos
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier and len(came_from) < max_steps:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            
            x, y = current
            for dx, dy in self.direction_deltas:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if neighbor not in self.nodes:
                    continue
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + abs(goal[0]-nx) + abs(goal[1]-ny)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
        
        if goal not in came_from:
            return None
        
        # 重建路径
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = came_from[cur]
            if cur is None:
                return None
        path.reverse()
        return path
    
    def get_exploration_target(self, agent_pos=None, env_size=10, rng=None):
        """信息增益导向的探索目标（friend-audit 深挖修复：
        原 goal_gen.py 检查此接口但从未实现——空间记忆引导从未生效，
        恒回退随机方向。实现后 GoalGenerator 自动启用）。

        优先级：①从未访问位置（随机采样排除已访问——最大信息增益）
                ②已访问中低熟悉度 + 高惊奇（评分：不熟悉×0.5 +
                  惊奇×0.3 + 距离适中偏好×0.2）
                ③无候选 → None（调用方随机回退）
        """
        rng = rng or np.random
        # ① 从未访问：边界内随机采样，排除已访问节点
        if len(self.nodes) < env_size * env_size:
            for _ in range(30):
                tx = int(rng.randint(0, env_size))
                ty = int(rng.randint(0, env_size))
                if (tx, ty) not in self.nodes:
                    return [tx, ty]
        # ② 已访问中选信息增益最高
        best, best_score = None, -1.0
        for pos, node in self.nodes.items():
            if agent_pos is not None and pos == tuple(agent_pos):
                continue
            unfamiliar = 1.0 - min(node.visit_count / 5.0, 1.0)
            dist = (abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1])
                    if agent_pos else 0)
            dist_pref = (1.0 if env_size * 0.15 <= dist <= env_size * 0.5
                         else 0.3)
            score = unfamiliar * 0.5 + node.avg_surprise * 0.3 + dist_pref * 0.2
            if score > best_score:
                best = list(pos)
                best_score = score
        return best  # None → 调用方随机回退

    def get_nearest_with_tag(self, from_pos, tag, max_dist=10):
        """找到最近的带有某个标签的位置"""
        if from_pos not in self.nodes:
            return None
        best = None
        best_dist = max_dist
        for pos, node in self.nodes.items():
            if getattr(node, tag, False):
                dist = abs(pos[0] - from_pos[0]) + abs(pos[1] - from_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best = pos
        return best
    
    def get_familiarity(self, pos) -> float:
        """对某个位置的熟悉度 [0, 1]"""
        if pos in self.nodes:
            return min(1.0, self.nodes[pos].visit_count / 5.0)
        return 0.0
    
    def tick_aging(self, decay=0.001):
        """每 tick 老化所有节点，附带衰减"""
        remove_list = []
        for pos, node in self.nodes.items():
            node.last_seen += 1
            # 长时间未访问的节点强度衰减
            if node.last_seen > 50:
                node.visit_count = max(0, node.visit_count - decay * node.last_seen)
            # 完全遗忘：访问计数归零且超过 200 tick 未访问
            if node.visit_count <= 0.01 and node.last_seen > 200:
                remove_list.append(pos)
        for pos in remove_list:
            del self.nodes[pos]
    
    def reset(self):
        self.__init__(self.max_nodes)


class SpatialNode:
    def __init__(self, pos, energy_change=0.0, surprise=0.0,
                 danger=False, food=False, water=False):
        self.pos = pos
        self.visit_count = 1
        self.last_seen = 0
        self.avg_energy_change = energy_change
        self.avg_surprise = surprise
        self.danger_count = 1 if danger else 0
        self.food_hint = food
        self.water_hint = water
