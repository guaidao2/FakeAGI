"""pymdp 1.0.3 API 探测（完整流程：infer_states→infer_policies→sample_action）"""
import sys
import jax
import jax.numpy as jnp

try:
    from pymdp import utils
    from pymdp.agent import Agent
    key = jax.random.PRNGKey(42)
    num_states = [4, 4]
    num_obs = [4, 4]
    num_actions = [4, 2]
    A = utils.random_A_array(key, num_obs, num_states)
    B = utils.random_B_array(key, num_states, num_actions)
    C = utils.random_factorized_categorical(key, num_obs)
    D = utils.random_factorized_categorical(key, num_states)
    agent = Agent(A=A, B=B, C=C, D=D, num_controls=num_actions,
                  categorical_obs=True)
    print("Agent construct OK")
    obs = [jnp.eye(4)[0][None], jnp.eye(4)[0][None]]  # one-hot + batch
    qs = agent.infer_states(obs, empirical_prior=agent.D)
    print("infer_states OK:", [q.shape for q in qs])
    q_pi = agent.infer_policies(qs)
    print("infer_policies OK")
    keys = jax.random.split(key, 2)
    action = agent.sample_action(q_pi, rng_key=keys[1:])
    print("sample_action OK:", action)
    agent.update_A(obs, qs)
    print("update_A OK")
    print("PYMDP_API_OK")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("PYMDP_API_FAIL:", repr(e))
    sys.exit(1)
