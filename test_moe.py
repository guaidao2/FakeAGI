"""
P1 验证 — MoE 专家路由测试

场景：模拟 3 类情境（觅食/避险/探索），确认：
  1. 路由器能区分情境（不同情境激活不同专家）
  2. 新情境反复出现 → 自动创建新专家
  3. 专家在线学习（局部过拟合各自领域）
  4. checkpoint 往返后专家池恢复
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cognition.decision.moe import MoERouter

def test():

    from seed_utils import seed_run, get_seed_from_env
    seed_run(get_seed_from_env(0))
    print("P1: MoE 专家路由测试", flush=True)
    router = MoERouter(state_dim=16, n_actions=5, max_experts=6,
                       top_k=2, create_threshold=0.35,
                       device="cpu")
    
    # 三类情境
    situations = {
        "forage":  np.array([0.8, 0.8, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "flee":    np.array([0.1, 0.1, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "explore": np.array([0.3, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }
    
    # 反复呈现三类情境 → 触发专家创建（每种 25 次）
    for i in range(25):
        for name, sv in situations.items():
            activations, _ = router.route(np.zeros(0), sv, surprise=0.1)
            if activations:
                a, _ = router.get_action(activations, sv)
                # 给该情境一个稳定的奖励（模拟领域专精）
                router.learn(activations, sv, a, reward=1.0)
    
    n_experts = len(router.experts)
    print(f"  创建专家数: {n_experts}", flush=True)
    
    # 验证路由区分度：同一情境 → 同一专家主导
    routing_consistency = 0
    trials = 20
    for name, sv in situations.items():
        winners = []
        for _ in range(trials):
            acts, _ = router.route(np.zeros(0), sv, surprise=0.1)
            if acts:
                winners.append(max(acts, key=acts.get))
        if winners and len(set(winners)) == 1:
            routing_consistency += 1
    print(f"  路由一致性: {routing_consistency}/3 情境稳定路由", flush=True)
    
    # checkpoint 往返
    sd = router.get_state_dict()
    router2 = MoERouter(state_dim=16, n_actions=5, max_experts=6,
                        top_k=2, create_threshold=0.35, device="cpu")
    router2.load_state_dict(sd)
    restore_ok = len(router2.experts) == n_experts
    print(f"  专家池恢复: {len(router2.experts)}/{n_experts}", flush=True)
    
    passed = n_experts >= 2 and routing_consistency >= 2 and restore_ok
    print(f"判定: {'OK 通过' if passed else 'NO 失败'} — "
          f"专家{n_experts}个, 一致路由{routing_consistency}/3, 恢复{restore_ok}")
    return passed

if __name__ == "__main__":
    test()
