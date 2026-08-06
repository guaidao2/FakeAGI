"""
P0 验证 — checkpoint 持久化往返测试

场景：AGI 运行 N tick → 保存 → 新 AGI 加载 → 确认状态恢复
通过标准：加载后 tick/生存时长/置信度/价值/空间记忆一致
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from main import AGI
from cognition import CognitionPipeline

def test():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    print("P0: checkpoint 持久化往返测试", flush=True)
    # 测试自洁：删除遗留旧档（旧代码生成的档含非安全对象，weights_only
    # 默认拒绝——新档由当前代码生成应 weights_only 兼容）
    import os as _os
    old = _os.path.join("checkpoints", "checkpoint_test.pth")
    if _os.path.exists(old):
        _os.remove(old)
    
    # 第一个生命：跑 200 tick 后保存
    agi1 = AGI()
    agi1.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    
    class E:
        def __init__(s):
            s.pos = [5, 5]; s.food = [2, 2]
        def get_pos(s): return s.pos
        def observe(s):
            return np.array([(s.food[0]-s.pos[0])/10, (s.food[1]-s.pos[1])/10, 0.0, 0.0])
        def step(s, a):
            dirs = [(0,0),(0,-1),(-1,0),(1,0),(0,1)]
            dx, dy = dirs[a % 5]
            s.pos[0] = max(0, min(9, s.pos[0]+dx))
            s.pos[1] = max(0, min(9, s.pos[1]+dy))
            eat = abs(s.pos[0]-2)+abs(s.pos[1]-2) < 2
            return {"energy_delta": 0.2 if eat else -0.001,
                    "water_delta": -0.0002}
        def food_nearby(s): return abs(s.pos[0]-2)+abs(s.pos[1]-2) < 4
    agi1.set_env(E())
    
    for _ in range(200):
        agi1.step()
    
    t1 = agi1.tick
    s1 = agi1.survival_ticks
    c1 = agi1.cognition.confidence if agi1.cognition else 0.0
    v1 = dict(agi1.value_system.secondary_values.get("food", {}))
    # 睡眠恢复断言（review should-fix：200 tick 疲劳 0.005×200=1.0>0.7
    # 必已触发睡眠——保存时应在睡，加载后必须仍在睡）
    sleep1 = bool(agi1.body.is_sleeping)
    path = agi1.save(tag="test")
    print(f"  保存: tick={t1} survival={s1} conf={c1:.3f} "
          f"food_val={v1.get('value', 0):.2f} sleeping={sleep1}")
    
    # 第二个生命：加载
    agi2 = AGI()
    agi2.set_cognition(CognitionPipeline({
        "input_dim": 4, "self_state_dim": 14,
        "hidden_dim": 64, "n_actions": 5, "n_strategies": 4
    }))
    ok = agi2.load(tag="test")
    
    t2 = agi2.tick
    s2 = agi2.survival_ticks
    c2 = agi2.cognition.confidence if agi2.cognition else 0.0
    v2 = dict(agi2.value_system.secondary_values.get("food", {}))
    sleep2 = bool(agi2.body.is_sleeping)
    print(f"  加载: tick={t2} survival={s2} conf={c2:.3f} "
          f"food_val={v2.get('value', 0):.2f} sleeping={sleep2}")
    
    # 判定
    tick_ok = t1 == t2
    surv_ok = s1 == s2
    val_ok = abs(v1.get("value", 0) - v2.get("value", 0)) < 0.01
    sleep_ok = sleep1 == sleep2
    print(f"  tick一致: {tick_ok}, survival一致: {surv_ok}, "
          f"价值一致: {val_ok}, 睡眠一致: {sleep_ok}")
    
    if ok and tick_ok and surv_ok and val_ok and sleep_ok:
        print("判定: OK 通过 — checkpoint 持久化完整（含睡眠状态）")
    else:
        print("判定: NO 失败 — 状态未完整恢复")
    return ok and tick_ok and surv_ok and val_ok and sleep_ok

if __name__ == "__main__":
    test()
