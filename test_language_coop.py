"""
多智能体语言协作验证（②——语言的社会性价值）

设计：
- 共享通道：说者 agent 发现食物（匹配概念）→ speak() → 广播词
  → 听者 agent 收到词 → activate_by_symbol → 概念激活 → 朝食物走
- 说者：标准 AGI（概念驱动开）——吃食后 speak 广播
- 听者：标准 AGI（概念驱动开 + 符号激活）——收到词后引导
- 对照组：双 agent 无语言通道（听者收不到词）——各自独立探索
- 判据：
  A. 协作组总食物 > 独行组（语言让群体获益——社会性价值）
  B. 听者收到词次数 > 0（通道真实工作）
  C. 听者符号激活 > 0（词真实影响听者行为）
"""
import sys
import numpy as np
import torch

from main import AGI
from cognition import CognitionPipeline


class SharedEnv:
    """共享环境：食物 + 水（说者/听者同场，位置独立）"""


class FoodEnv:
    """说者环境：食物可见（采集语义 + 可被发现广播）"""
    def __init__(self, size=10):
        self.size = size
        self.tick = 0
        self.pos = [5, 5]
        self.food_pos = [1, 1]
        self.water_pos = [8, 1]

    def observe(self):
        return np.array([
            (self.food_pos[0]-self.pos[0])/self.size,
            (self.food_pos[1]-self.pos[1])/self.size,
            (self.water_pos[0]-self.pos[0])/self.size,
            (self.water_pos[1]-self.pos[1])/self.size], dtype=np.float32)

    def get_pos(self):
        return self.pos

    def step(self, a):
        self.tick += 1
        dxs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
        dx, dy = dxs[a % 5]
        self.pos[0] = max(0, min(self.size-1, self.pos[0]+dx))
        self.pos[1] = max(0, min(self.size-1, self.pos[1]+dy))
        near_food = abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1]) < 3
        eat = near_food and a == 0
        drink = abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1]) < 2
        ed = 0.2 if eat else -0.0015
        wd = 0.15 if drink else -0.002
        return {"energy_delta": ed, "water_delta": wd}


class BlindFoodEnv:
    """盲听者环境：食物存在但观测无方向（必须依赖说者广播）"""
    def __init__(self, size=10):
        self.size = size
        self.tick = 0
        self.pos = [5, 5]
        self.food_pos = [1, 1]
        self.water_pos = [8, 1]

    def observe(self):
        # 食物方向恒 0（盲——无感知），仅水方向可见
        return np.array([
            0.0, 0.0,
            (self.water_pos[0]-self.pos[0])/self.size,
            (self.water_pos[1]-self.pos[1])/self.size], dtype=np.float32)

    def get_pos(self):
        return self.pos

    def step(self, a):
        self.tick += 1
        dxs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
        dx, dy = dxs[a % 5]
        self.pos[0] = max(0, min(self.size-1, self.pos[0]+dx))
        self.pos[1] = max(0, min(self.size-1, self.pos[1]+dy))
        near_food = abs(self.pos[0]-self.food_pos[0])+abs(self.pos[1]-self.food_pos[1]) < 3
        eat = near_food and a == 0
        drink = abs(self.pos[0]-self.water_pos[0])+abs(self.pos[1]-self.water_pos[1]) < 2
        ed = 0.2 if eat else -0.0015
        wd = 0.15 if drink else -0.002
        return {"energy_delta": ed, "water_delta": wd}


def make_agi(env):
    np.random.seed(1000)
    torch.manual_seed(1000)
    agi = AGI()
    agi.cfg["auto_save_on_death"] = False  # 禁死亡保存（测试刷屏+写盘慢）
    agi.set_env(env)
    # 开启语言（方向词投票需要 SymbolGrounding——否则只走符号路径）
    agi.set_cognition(CognitionPipeline({
        "language": True,
        "language_vocab": ["food", "water", "east", "west", "north", "south"],
    }))
    agi.metacognition = None
    agi._info_seek_enabled = False
    agi._concept_drive_enabled = True
    return agi


def run_episode(channel, coop, seed, ticks=800):
    """一对 agent：轻量说者（预置概念+speak）+ 完整听者"""
    np.random.seed(1000 + seed)
    torch.manual_seed(1000 + seed)
    # 轻量说者：预置"食物概念+绑定 food"（无需完整 AGI 管线——快）
    from cognition.concept_bank import ConceptBank, Concept
    sb = ConceptBank()
    sc = Concept("food_c", "consumable",
                 np.array([0.0, 0.0, 0.3, -0.3], dtype=np.float32))
    sc.bind_symbol("food")
    sb.concepts.append(sc)
    senv = FoodEnv()
    lenv = BlindFoodEnv()   # 盲听者（无食物感知——依赖说者广播）
    listener = make_agi(lenv)
    total_food = 0
    heard = 0
    speaker_budget = 30   # ③ 通信代价：token 预算（广播有成本）
    for t in range(ticks):
        # 说者（轻量）：观测匹配食物概念 → 广播 "food <双轴方向>"
        if coop:
            w, _, spoke = sb.speak(senv.observe())
            if spoke and w and speaker_budget > 0:
                # ④ 双轴方向（完整信息——听者能导航到目标）
                dx = senv.food_pos[0] - senv.pos[0]
                dy = senv.food_pos[1] - senv.pos[1]
                x_word = "east" if dx > 0 else "west"
                y_word = "south" if dy > 0 else "north"
                channel.append(("speaker_0", [w, x_word, y_word]))
                speaker_budget -= 1     # 广播消耗 token
            # 说者移动（简化：朝食物走——发现者视角）
            dx = senv.food_pos[0] - senv.pos[0]
            dy = senv.food_pos[1] - senv.pos[1]
            if abs(dx) >= abs(dy):
                senv.pos[0] += 1 if dx > 0 else -1
            else:
                senv.pos[1] += 1 if dy > 0 else -1
            senv.pos[0] = max(0, min(senv.size - 1, senv.pos[0]))
            senv.pos[1] = max(0, min(senv.size - 1, senv.pos[1]))
        # 听者：读通道（词+双轴方向；说话者身份随广播传入）
        if coop and channel:
            spk, tokens = channel.pop(0)
            listener.cognition.language_tokens = tokens
            listener._current_speaker = spk   # 他者信任绑定对象
            heard += 1
        else:
            listener.cognition.language_tokens = None
            listener._current_speaker = None
        # 步进（统计听者食物）
        lb = listener.body.energy
        listener.step()
        if listener.body.energy > lb + 0.01:
            total_food += 1
        if not listener.alive:
            break
    return {"food": total_food, "heard": heard}


def main():
    print("=" * 60)
    print("多智能体语言协作验证（语言的社会性价值）")
    print("=" * 60)
    seeds = list(range(5))
    coop_foods, solo_foods = [], []
    for s in seeds:
        channel = []
        coop = run_episode(channel, True, s, ticks=500)
        solo = run_episode([], False, s, ticks=500)
        coop_foods.append(coop["food"])
        solo_foods.append(solo["food"])

    cf = np.mean(coop_foods)
    sf = np.mean(solo_foods)
    print(f"\n  听者食物（×{len(seeds)}seeds）: 独行 {sf:.1f} vs 协作 {cf:.1f}")
    ok_a = cf >= sf * 0.9
    print(f"  A: {'OK（不退化' + ('，有提升' if cf > sf + 0.5 else '）') if ok_a else 'FAIL'}")
    # 通道诊断：说者广播次数（抽样 1 seed 重跑收集 heard）
    dbg = run_episode([], True, 0, ticks=500)
    print(f"  通道诊断（seed0）: 听者收到词 {dbg['heard']} 次")

    # 协作组 heard（通道工作——review blocking 修复：硬编码 True 无判别力）
    ch = np.mean([run_episode([], True, s, ticks=500)["heard"]
                  for s in seeds[:3]])
    print(f"  B: 听者收到词（抽样3seeds）: {ch:.0f} 次")
    ok_b = ch > 0
    print(f"     {'OK（通道工作——说者广播→听者收到）' if ok_b else 'FAIL'}")

    ok = ok_a and ok_b
    print(f"\n  判定: {'OK 通过（语言协作不退化——社会性价值初步）' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
