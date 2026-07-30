"""
AGI 生存状态可视化 — 实时观察生物状态 + 迷宫导航 + 驱动力 + LNN

用法:
  python bio_viewer.py                    # 自由环境
  python bio_viewer.py --maze 10          # 10x10 迷宫
  python bio_viewer.py --ticks 100000     # 持续运行
"""

import sys, os, time, argparse
import numpy as np
import pygame as pg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import AGI
from cognition import CognitionPipeline

# ─── 颜色 ───
C_BG     = (15, 15, 25)
C_WALL   = (40, 45, 60)
C_FLOOR  = (180, 185, 195)
C_AGENT  = (50, 200, 255)
C_GOAL   = (50, 255, 100)
C_VISITED= (30, 60, 100)
C_TEXT   = (220, 220, 230)
C_DIM    = (120, 130, 140)
C_RED    = (255, 80, 80)
C_GREEN  = (80, 220, 80)
C_ORANGE = (255, 180, 50)
C_BAR_BG = (40, 40, 55)
C_HP     = (80, 255, 120)
C_ENERGY = (255, 200, 50)
C_WATER  = (50, 150, 255)
C_FATIGUE= (180, 100, 255)
C_STRESS = (255, 100, 100)
C_CURIOS = (100, 255, 200)
C_HUNGER = (255, 180, 100)

DRIVE_COLORS = {
    "hunger": C_HUNGER, "thirst": C_WATER, "fatigue": C_FATIGUE,
    "curiosity": C_CURIOS, "boredom": C_DIM, "fear": C_RED,
}

DIR_NAMES = {0: "·", 1: "↑", 2: "←", 3: "→", 4: "Zzz"}


class BioViewer:
    def _get_font(self, size):
        """直接加载中文字体文件（绕过 pygame SysFont 的 Windows 枚举 bug）"""
        font_paths = [
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
        ]
        for path in font_paths:
            if os.path.isfile(path):
                try:
                    return pg.font.Font(path, size)
                except Exception:
                    continue
        # 全部失败才用默认字体（可能不显示中文）
        return pg.font.Font(None, size)
    
    def __init__(self, maze_size=0, max_ticks=50000):
        # AGI 初始化
        self.agi = AGI()
        cfg = {"input_dim": 4, "self_state_dim": 14, "hidden_dim": 64,
               "n_actions": 5, "n_strategies": 4}
        self.agi.set_cognition(CognitionPipeline(cfg))
        
        # 环境
        self.maze_mode = maze_size > 0
        if self.maze_mode:
            self._init_maze(maze_size)
        else:
            self._init_biome()
        
        self.max_ticks = max_ticks
        
        # 界面
        pg.init()
        self.w = 1400
        self.h = 800
        self.screen = pg.display.set_mode((self.w, self.h), pg.RESIZABLE)
        pg.display.set_caption("AGI 生物模拟器")
        self.clock = pg.time.Clock()
        
        # 字体（防止 Windows 字体枚举 bug）
        self.font = self._get_font(16)
        self.font_big = self._get_font(20)
        self.font_small = self._get_font(13)
        
        self.running = True
        self.paused = False
        self.speed = 1
        self.tick_count = 0
    
    def _init_maze(self, size):
        sys.path.insert(0, r"D:\编程\game\brain001")
        from maze_env import Maze as MazeGen
        self.env = type("MazeAdapter", (), {})()
        self.env.maze = MazeGen(size=size)
        self.env.size = size
        cx, cy = size // 2, size // 2
        for y in range(max(1, cy-2), min(size, cy+3)):
            for x in range(max(1, cx-2), min(size, cx+3)):
                if self.env.maze.grid[y][x] == 0:
                    self.env.pos = [x, y]
                    break
            else: continue
            break
        self.env.goal = list(self.env.maze.goal)
        self.env.visited_before = {tuple(self.env.pos)}
        
        def get_pos(): return self.env.pos
        def observe():
            x, y = self.env.pos
            def w(dx, dy):
                nx, ny = x+dx, y+dy
                if 0<=nx<size and 0<=ny<size: return float(self.env.maze.grid[ny][nx])
                return 1.0
            return np.array([w(0,-1), w(-1,0), w(1,0), w(0,1)])
        def step(action):
            if action == 4: return {"energy_delta": -0.0005, "water_delta": -0.0001}
            dirs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
            dx, dy = dirs[action%5]
            nx, ny = self.env.pos[0]+dx, self.env.pos[1]+dy
            if (nx, ny) != tuple(self.env.pos):
                if 0<=nx<size and 0<=ny<size and self.env.maze.grid[ny][nx]==0:
                    self.env.pos = [nx, ny]
                else:
                    return {"energy_delta": -0.001, "water_delta": -0.0003}
            else:
                return {"energy_delta": -0.0005, "water_delta": -0.0001}
            self.env.maze.agent_pos = tuple(self.env.pos)
            at_goal = tuple(self.env.pos) == self.env.maze.goal
            exploring = tuple(self.env.pos) not in self.env.visited_before
            self.env.visited_before.add(tuple(self.env.pos))
            energy = 0.2 if at_goal else (0.02 if exploring else -0.001)
            water = 0.05 if at_goal else -0.0005
            return {"energy_delta": energy, "water_delta": water}
        self.env.get_pos = get_pos
        self.env.observe = observe
        self.env.step = step
        self.agi.set_env(self.env)
    
    def _init_biome(self):
        class BioEnv:
            def __init__(self):
                self.pos = [5, 5]
                self.food = [[2,2],[7,8],[3,6]]
                self.water = [[8,1],[1,8]]
            def get_pos(self): return self.pos
            def observe(self):
                dxs=[f[0]-self.pos[0] for f in self.food]
                dys=[f[1]-self.pos[1] for f in self.food]
                nf=min(range(len(self.food)),key=lambda i:abs(dxs[i])+abs(dys[i]))
                return np.array([dxs[nf]/10, dys[nf]/10, 0.0, 0.0])
            def step(self, a):
                if a==4: return {"energy_delta":0.0}
                dxs=[(0,0),(0,-1),(-1,0),(1,0),(0,0)]
                dx,dy=dxs[a%5]
                self.pos[0]=max(0,min(9,self.pos[0]+dx))
                self.pos[1]=max(0,min(9,self.pos[1]+dy))
                eat=any(abs(self.pos[0]-f[0])+abs(self.pos[1]-f[1])<2 for f in self.food)
                drink=any(abs(self.pos[0]-w[0])+abs(self.pos[1]-w[1])<2 for w in self.water)
                return {"energy_delta":0.15 if eat else(0.05 if drink else-0.001),
                        "water_delta":0.1 if drink else-0.0003}
            def food_nearby(self):
                return any(abs(self.pos[0]-f[0])+abs(self.pos[1]-f[1])<4 for f in self.food)
        self.env = BioEnv()
        self.agi.set_env(self.env)
    
    def handle_events(self):
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.running = False
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_SPACE:
                    self.paused = not self.paused
                elif e.key in (pg.K_UP, pg.K_RIGHT):
                    self.speed = min(10, self.speed + 1)
                elif e.key in (pg.K_DOWN, pg.K_LEFT):
                    self.speed = max(1, self.speed - 1)
                elif e.key == pg.K_r:
                    self.agi = AGI()
                    self.agi.set_cognition(CognitionPipeline(
                        {"input_dim":4,"self_state_dim":14,"hidden_dim":64,
                         "n_actions":5,"n_strategies":4}))
                    if self.maze_mode:
                        self.agi.set_env(self.env)
                    self.tick_count = 0
    
    def draw(self):
        self.screen.fill(C_BG)
        
        # ─── 迷宫地图 (左) ───
        if self.maze_mode:
            self._draw_maze(20, 20, 560, 560)
        else:
            self._draw_free_map(20, 20, 560, 560)
        
        # ─── 身体状态条 (右上) ───
        self._draw_body_bars(620, 20, 360)
        
        # ─── 驱动力 (右中) ───
        self._draw_drives(620, 280, 360)
        
        # ─── 空间记忆 (右下) ───
        self._draw_space_memory(20, 600, 560, 180)
        
        # ─── 认知状态 (最右) ───
        self._draw_cognition(1020, 20, 360)
        
        # ─── 操作提示 (最底部) ───
        self._draw_controls()
        
        pg.display.flip()
    
    def _draw_maze(self, x0, y0, w, h):
        size = self.env.size
        cell = min(w, h) // size
        m = self.env.maze
        gx0, gy0 = m.goal
        
        # 网格
        for y in range(size):
            for x in range(size):
                rx = x0 + x * cell
                ry = y0 + y * cell
                if m.grid[y][x] == 1:
                    pg.draw.rect(self.screen, C_WALL, (rx, ry, cell-1, cell-1))
                else:
                    pg.draw.rect(self.screen, C_FLOOR, (rx, ry, cell-1, cell-1))
        
        # 空间记忆标记
        for pos, node in self.agi.spatial_memory.nodes.items():
            if 0 <= pos[0] < size and 0 <= pos[1] < size:
                rx = x0 + pos[0] * cell
                ry = y0 + pos[1] * cell
                col = C_VISITED
                if getattr(node, 'food_hint', False):
                    col = C_GREEN
                pg.draw.rect(self.screen, col, (rx+2, ry+2, cell-5, cell-5))
                # 访问计数
                if node.visit_count > 1:
                    t = self.font_small.render(str(node.visit_count), True, C_DIM)
                    self.screen.blit(t, (rx+2, ry+2))
        
        # 目标
        rx = x0 + gx0 * cell
        ry = y0 + gy0 * cell
        pg.draw.circle(self.screen, C_GOAL, (rx+cell//2, ry+cell//2), cell//3)
        
        # 智能体
        px, py = self.env.pos
        rx = x0 + px * cell + cell//2
        ry = y0 + py * cell + cell//2
        pg.draw.circle(self.screen, C_AGENT, (rx, ry), cell//3)
        pg.draw.circle(self.screen, (255,255,255), (rx, ry), cell//3, 2)
        
        # 标签
        t = self.font.render(f"迷宫 {size}x{size} | Tick {self.tick_count}", True, C_TEXT)
        self.screen.blit(t, (x0, y0 + h + 5))
    
    def _draw_free_map(self, x0, y0, w, h):
        """自由环境地图"""
        pg.draw.rect(self.screen, (30, 35, 45), (x0, y0, w, h))
        env = self.env
        cell = min(w, h) // 10
        
        # 食物点
        for fx, fy in env.food:
            pg.draw.circle(self.screen, C_GREEN,
                          (x0+fx*cell+cell//2, y0+fy*cell+cell//2), cell//3)
        # 水源
        for wx, wy in env.water:
            pg.draw.circle(self.screen, C_WATER,
                          (x0+wx*cell+cell//2, y0+wy*cell+cell//2), cell//3)
        # 智能体
        px, py = env.pos
        pg.draw.circle(self.screen, C_AGENT,
                      (x0+px*cell+cell//2, y0+py*cell+cell//2), cell//3)
        
        t = self.font.render(f"自由环境 | Tick {self.tick_count}", True, C_TEXT)
        self.screen.blit(t, (x0, y0 + h + 5))
    
    def _draw_body_bars(self, x0, y0, w):
        body = self.agi.body
        bars = [
            ("健康", body.health, C_HP),
            ("能量", body.energy / 2.0, C_ENERGY),
            ("水分", body.water, C_WATER),
            ("结构", body.integrity, C_GREEN),
            ("疲劳", body.fatigue, C_FATIGUE),
            ("应激", body.stress, C_STRESS),
        ]
        
        t = self.font_big.render(" 身体状态", True, C_TEXT)
        self.screen.blit(t, (x0, y0))
        
        bw = w - 80
        for i, (name, val, col) in enumerate(bars):
            by = y0 + 30 + i * 28
            # 标签
            lt = self.font.render(name, True, C_TEXT)
            self.screen.blit(lt, (x0, by))
            # 背景条
            pg.draw.rect(self.screen, C_BAR_BG, (x0+50, by+2, bw, 16), border_radius=3)
            # 值条
            fw = max(2, int(bw * min(1.0, max(0.0, val))))
            pg.draw.rect(self.screen, col, (x0+50, by+2, fw, 16), border_radius=3)
            # 数值
            vt = self.font_small.render(f"{val:.2f}", True, C_TEXT)
            self.screen.blit(vt, (x0+50+fw+4, by+1))
            
            # 昼夜节律（在应激下方）
            if name == "应激":
                circ = body.circadian
                night = abs(np.sin(circ)) < 0.3
                ct = self.font_small.render("  夜" if night else "  昼", True, C_TEXT)
                self.screen.blit(ct, (x0+50+fw+60, by+1))
        
        # 总存活 Tick
        st = self.font.render(f"存活: {self.tick_count} ticks", True, C_TEXT)
        self.screen.blit(st, (x0, y0 + 30 + len(bars)*28 + 5))
    
    def _draw_drives(self, x0, y0, w):
        drives = self.agi.drives
        items = [
            ("饥饿", drives.hunger, C_HUNGER),
            ("口渴", drives.thirst, C_WATER),
            ("疲劳", drives.fatigue_drive, C_FATIGUE),
            ("安全", drives.safety, C_GREEN),
            ("好奇", drives.curiosity, C_CURIOS),
            ("无聊", drives.boredom, C_DIM),
        ]
        
        self.screen.blit(self.font_big.render(" 驱动力", True, C_TEXT), (x0, y0))
        
        bw = w - 80
        for i, (name, val, col) in enumerate(items):
            by = y0 + 30 + i * 24
            lt = self.font.render(name, True, C_TEXT)
            self.screen.blit(lt, (x0, by))
            pg.draw.rect(self.screen, C_BAR_BG, (x0+50, by+2, bw, 14), border_radius=3)
            fw = max(2, int(bw * min(1.0, max(0.0, val))))
            pg.draw.rect(self.screen, col, (x0+50, by+2, fw, 14), border_radius=3)
        
        # 主导驱动力
        dom = drives.get_dominance()
        dcol = DRIVE_COLORS.get(dom, C_TEXT)
        dt = self.font.render(f"主导: ", True, C_TEXT)
        dv = self.font_big.render(dom.upper(), True, dcol)
        self.screen.blit(dt, (x0, y0 + 30 + len(items)*24 + 5))
        self.screen.blit(dv, (x0 + 55, y0 + 30 + len(items)*24 + 3))
        
        # 睡眠状态
        sleep_col = C_GREEN if self.agi.body.is_sleeping else C_DIM
        st = self.font.render(f"睡眠: {'Zzz' if self.agi.body.is_sleeping else '清醒'}", True, sleep_col)
        self.screen.blit(st, (x0 + 140, y0 + 30 + len(items)*24 + 5))
    
    def _draw_space_memory(self, x0, y0, w, h):
        mem = self.agi.spatial_memory
        self.screen.blit(self.font_big.render(" 空间记忆", True, C_TEXT), (x0, y0))
        self.screen.blit(self.font.render(f"{len(mem.nodes)} 节点", True, C_TEXT), (x0+150, y0+2))
        
        if mem.nodes and self.maze_mode:
            size = self.env.size
            cell = min(w-40, h-40) // size
            ox, oy = x0+20, y0+30
            for y in range(size):
                for x in range(size):
                    rx = ox + x * cell
                    ry = oy + y * cell
                    if self.env.maze.grid[y][x] == 1:
                        pg.draw.rect(self.screen, C_WALL, (rx, ry, cell-1, cell-1))
                    elif (x, y) in mem.nodes:
                        node = mem.nodes[(x, y)]
                        # 热力色：访问越多越亮
                        intense = min(1.0, node.visit_count / 5.0)
                        col = (int(30+intense*200), int(60+intense*150), int(80+intense*120))
                        pg.draw.rect(self.screen, col, (rx, ry, cell-1, cell-1))
                    else:
                        pg.draw.rect(self.screen, C_FLOOR, (rx, ry, cell-1, cell-1))
            
            # 当前位置
            px, py = self.env.pos
            rx = ox + px * cell + cell//2
            ry = oy + py * cell + cell//2
            pg.draw.circle(self.screen, C_AGENT, (rx, ry), max(4, cell//4))
        else:
            self.screen.blit(self.font.render("(等待积累)", True, C_DIM), (x0+20, y0+30))
    
    def _draw_cognition(self, x0, y0, w):
        cp = self.agi.cognition
        self.screen.blit(self.font_big.render(" 认知核心", True, C_TEXT), (x0, y0))
        
        if cp:
            lnnd = cp.lnn.hidden_dim
            growthn = cp.growth_count
            gamen_strats = cp.gamenn.n_strategies
            strategy_weights = cp.gamenn.strategy_weights
            
            lines = [
                f"LNN 隐藏层: {lnnd}d",
                f"生长次数: {growthn}/{cp.max_growths}",
                f"冷却: {cp.growth_cooldown}t",
                f"GameNN 策略: {gamen_strats}",
            ]
            
            for i, line in enumerate(lines):
                lt = self.font.render(line, True, C_TEXT)
                self.screen.blit(lt, (x0, y0 + 30 + i * 22))
            
            # 策略权重条形图
            yb = y0 + 30 + len(lines) * 22 + 5
            self.screen.blit(self.font.render("策略权重:", True, C_TEXT), (x0, yb))
            bw = w - 20
            for si in range(len(strategy_weights)):
                sy = yb + 20 + si * 18
                wt = self.font_small.render(f"S{si}", True, C_TEXT)
                self.screen.blit(wt, (x0, sy))
                fw = max(2, int(bw * strategy_weights[si]))
                pg.draw.rect(self.screen, C_BAR_BG, (x0+25, sy+1, bw, 12), border_radius=2)
                col = [C_CURIOS, C_GREEN, C_ORANGE, C_RED, C_WATER][si % 5]
                pg.draw.rect(self.screen, col, (x0+25, sy+1, fw, 12), border_radius=2)
        else:
            self.screen.blit(self.font.render("(未初始化)", True, C_DIM), (x0, y0+30))
    
    def _draw_controls(self):
        controls = [
            f"速度: {self.speed}x | {'|| PAUSED' if self.paused else '|| RUNNING'}",
            "SPACE=暂停 | ↑↓=速度 | R=重置 | ESC=退出",
        ]
        for i, c in enumerate(controls):
            ct = self.font_small.render(c, True, C_DIM)
            self.screen.blit(ct, (10, self.h - 30 + i * 16))
    
    def run(self):
        print("[AGI] 可视化启动", flush=True)
        start = time.time()
        
        while self.running:
            self.handle_events()
            
            if not self.paused:
                for _ in range(self.speed):
                    if self.tick_count < self.max_ticks:
                        self.agi.step()
                        self.tick_count += 1
                    else:
                        self.paused = True
                        break
            
            self.draw()
            self.clock.tick(30)
        
        pg.quit()
        elapsed = time.time() - start
        print(f"[AGI] 结束. {elapsed:.0f}s, {self.tick_count} ticks", flush=True)


def main():
    parser = argparse.ArgumentParser(description="AGI 生物模拟可视化")
    parser.add_argument("--maze", type=int, default=0, help="迷宫尺寸")
    parser.add_argument("--ticks", type=int, default=50000)
    args = parser.parse_args()
    
    viewer = BioViewer(maze_size=args.maze, max_ticks=args.ticks)
    viewer.run()


if __name__ == "__main__":
    main()
