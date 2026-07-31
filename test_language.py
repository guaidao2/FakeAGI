"""
实验9：语言能力 — 符号接地（理解 + 说话）

场景：
  环境事件伴随语言标记（词）：
    - 靠近食物时环境说 "food"
    - 靠近水源时环境说 "water"
    - 危险区（右下角）时环境说 "danger"
  阶段1（接地训练）：收集 (词, 状态方向) 样本 → 语言器官学习"词→状态"
  阶段2（理解）：只给词 → 语言向量预测状态方向（误差 vs 随机基线）
  阶段3（说话）：只给内部状态（LNN hidden）→ word_probe 选词（状态参与）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from main import AGI
from cognition import CognitionPipeline

VOCAB = ["food", "water", "danger", "safe", "near", "far", "yes", "no"]

class LangEnv:
    """语言标记环境：状态 + 词"""
    def __init__(self):
        self.pos = [5, 5]
        self.food = [7, 7]
        self.water = [2, 2]
        self.danger = [8, 8]
    def get_pos(self): return self.pos
    def observe(self):
        fx, fy = self.food; wx, wy = self.water
        return np.array([(fx-self.pos[0])/10, (fy-self.pos[1])/10,
                         (wx-self.pos[0])/10, (wy-self.pos[1])/10])
    def get_language(self):
        """根据状态生成词（模拟环境"说话"）"""
        d_food = abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])
        d_water = abs(self.pos[0]-self.water[0])+abs(self.pos[1]-self.water[1])
        d_danger = abs(self.pos[0]-self.danger[0])+abs(self.pos[1]-self.danger[1])
        words = []
        if d_food < 5: words.append("food")
        if d_water < 5: words.append("water")
        if d_danger < 5: words.append("danger")
        return words if words else ["safe"]
    def step(self, a):
        dirs=[(0,0),(0,-1),(-1,0),(1,0),(0,1)]; dx,dy=dirs[a%5]
        self.pos[0]=max(0,min(9,self.pos[0]+dx)); self.pos[1]=max(0,min(9,self.pos[1]+dy))
        d_food = abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])
        d_water = abs(self.pos[0]-self.water[0])+abs(self.pos[1]-self.water[1])
        ed = 0.3 if d_food < 2 else (0.01 if d_water < 2 else -0.001)
        wd = 0.05 if d_food < 2 else (0.15 if d_water < 2 else -0.0005)
        return {'energy_delta': ed, 'water_delta': wd}
    def get_energy_delta(self, a):
        d_food = abs(self.pos[0]-self.food[0])+abs(self.pos[1]-self.food[1])
        d_water = abs(self.pos[0]-self.water[0])+abs(self.pos[1]-self.water[1])
        return 0.3 if d_food < 2 else (0.01 if d_water < 2 else -0.001)
    def get_damage(self, a):
        return 0.0
    def food_nearby(self): return False


def test():
    print("实验9: 语言能力 — 符号接地", flush=True)
    cfg = {
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4,
        "language": True, "language_vocab": VOCAB,
    }
    agi = AGI()
    agi.set_cognition(CognitionPipeline(cfg))
    env = LangEnv()
    agi.set_env(env)
    lang = agi.cognition.language

    # ─── 阶段1：接地训练（收集 词+状态方向 样本）───
    word_samples = []
    state_samples = []
    hidden_samples = []
    last_words = None
    for t in range(800):
        env.words = env.get_language()
        if env.words != last_words:
            agi.cognition.language_tokens = env.words
            last_words = env.words
        agi.step()
        # 采集样本：词 + 当前状态方向（食物/水源方向）+ LNN hidden
        if env.words and t % 4 == 0:
            word_samples.append(env.words)
            state_samples.append(env.observe())
            if agi.cognition.last_lnn_out is not None:
                hidden_samples.append(
                    agi.cognition.last_lnn_out.detach().cpu().numpy().flatten())
        if not agi.alive:
            break
    # 接地训练：词 → 状态方向（真实梯度更新语言器官）
    loss = lang.train_grounding(word_samples, np.array(state_samples),
                                epochs=20, lr=0.01)
    print(f"  阶段1 接地训练完成（{len(word_samples)} 样本, loss={loss:.4f}, "
          f"alive={agi.alive}）", flush=True)

    # ─── 阶段2：理解 — 词→状态预测（用接地训练后的器官）───
    organ = lang.organ
    Xw, Yw = [], []
    Xr, Yr = [], []
    for _ in range(300):
        pos = np.random.randint(0, 10, 2)
        env.pos = pos.tolist()
        words = env.get_language()
        obs = env.observe()
        target = obs[:4]  # 食物+水源方向
        tok_ids = lang.tokenize(words)
        if tok_ids:
            lv = organ.encode(tok_ids).detach().cpu().numpy().flatten()
            Xw.append(lv); Yw.append(target)
        Xr.append(np.random.randn(organ.output_dim)); Yr.append(target)
    def train_lin(X, Y):
        if len(X) < 10: return 1e9
        Xt = np.hstack([np.array(X), np.ones((len(X), 1))])
        w = np.linalg.lstsq(Xt, np.array(Y), rcond=None)[0]
        pred = Xt @ w
        return float(np.mean(np.linalg.norm(pred - np.array(Y), axis=1)))
    err_word = train_lin(Xw, Yw)
    err_rand = train_lin(Xr, Yr)
    print(f"  阶段2 理解: 词→状态误差 {err_word:.3f} vs 随机基线 {err_rand:.3f}", flush=True)
    understood = err_word < err_rand * 0.9

    # ─── 阶段3：说话 — 内部状态→词选择（状态参与，非词循环）───
    # 重新采集**无词** hidden 样本（训练与测试分布一致：hidden 不含词）
    hidden_samples = []
    no_word_words = []
    agi.cognition.language_tokens = None
    for _ in range(200):
        env.pos = np.random.randint(0, 10, 2).tolist()
        agi.step()
        if agi.cognition.last_lnn_out is not None:
            hidden_samples.append(
                agi.cognition.last_lnn_out.detach().cpu().numpy().flatten())
            no_word_words.append(env.get_language())
    # 统一 hidden 维度（LNN 生长会导致维度变化，截断到最小）
    if hidden_samples:
        min_dim = min(len(h) for h in hidden_samples)
        hidden_arr = np.array([h[:min_dim] for h in hidden_samples])
    else:
        hidden_arr = np.zeros((1, 64))
    word_arr = []
    for ws in no_word_words[:len(hidden_arr)]:
        tok = lang.tokenize(ws)
        word_arr.append(tok[0] if tok else 0)
    # word_probe 输入维度对齐 hidden 截断维度
    probe_in = hidden_arr.shape[1] if hidden_arr.ndim == 2 else 64
    if (len(hidden_arr) >= 20 and len(word_arr) >= 20
            and organ.word_probe.in_features != probe_in):
        # 重建 word_probe（维度匹配）
        old_w = organ.word_probe.weight.data.clone()
        old_b = organ.word_probe.bias.data.clone()
        organ.word_probe = torch.nn.Linear(probe_in, organ.vocab_size)
        with torch.no_grad():
            n = min(old_w.shape[0], organ.vocab_size)
            m = min(old_w.shape[1], probe_in)
            organ.word_probe.weight[:n, :m] = old_w[:n, :m]
            organ.word_probe.bias[:n] = old_b[:n]
    if len(hidden_arr) >= 20 and len(word_arr) >= 20:
        Ht = torch.tensor(hidden_arr, dtype=torch.float32)[:, :probe_in]
        Wt = torch.tensor(word_arr, dtype=torch.long)
        optimizer = torch.optim.Adam(organ.word_probe.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(300):
            optimizer.zero_grad()
            logits = organ.word_probe(Ht)
            loss_p = loss_fn(logits, Wt)
            loss_p.backward()
            optimizer.step()
        # 测试：新状态 → hidden（**不含词**：language_tokens=None，只喂观测状态）
        correct = 0; total = 0
        for _ in range(100):
            pos = np.random.randint(0, 10, 2)
            env.pos = pos.tolist()
            words = env.get_language()
            obs = env.observe()
            # 关键：清空语言输入，让 hidden 只反映观测状态（真正测试状态→词）
            agi.cognition.language_tokens = None
            agi.step()
            if agi.cognition.last_lnn_out is not None:
                h = agi.cognition.last_lnn_out.detach().cpu().numpy().flatten()
                if len(h) >= probe_in:
                    with torch.no_grad():
                        logits = organ.word_probe(
                            torch.tensor(h[:probe_in], dtype=torch.float32).unsqueeze(0))
                        pred = VOCAB[logits.argmax().item()]
                    tok_ids = lang.tokenize(words)
                    if tok_ids:
                        correct += (pred == words[0])
                        total += 1
        acc = correct / max(1, total)
    else:
        acc = 0.0
    print(f"  阶段3 说话: 状态→词准确率 {acc:.2f} ({correct}/{total})", flush=True)
    speaks = acc > 0.3

    # 判定
    checks = [
        ("理解（词→状态预测）", understood),
        ("说话（状态→词选择）", speaks),
        ("存活", agi.alive),
    ]
    passed = all(c for _, c in checks)
    print(f"\n判定: {'OK 通过' if passed else 'FAIL 未通过'}", flush=True)
    for name, ok in checks:
        print(f"  {'[OK]' if ok else '[X]'} {name}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(test())
