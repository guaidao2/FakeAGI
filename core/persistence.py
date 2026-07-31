"""
P0: Checkpoint 持久化 — 跨 session 身份连续性

让 FakeAGI 在 session 结束时保存自己的完整状态，
新 session 启动时加载恢复 → "昨天的我"延续到今天。

保存内容：
  1. 认知核心：LNN、世界模型、GameNN（权重）
  2. 价值系统：次级价值表（核心价值不可变，不入盘）
  3. 空间记忆：地图节点
  4. 概念库：组合式反事实的概念
  5. 元认知状态：策略得分、缺口统计
  6. 身体状态：稳态变量（能量/水分/疲劳等）
  7. 自模型：存在概率历史
  8. 元数据：tick 数、存活时长、峰值健康（作为 fitness 评估）
"""

import os
import json
import numpy as np
import torch


CHECKPOINT_DIR = "checkpoints"


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(agi, path: str = None, tag: str = "latest") -> str:
    """保存 AGI 完整状态到 checkpoint 文件"""
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{tag}.pth")
    _ensure_dir(os.path.dirname(path))

    data = {}

    # 1. 认知核心权重（如果有 cognition）
    if agi.cognition is not None:
        try:
            g = agi.cognition.gamenn
            data["cognition"] = {
                "lnn": agi.cognition.lnn.state_dict(),
                "world_model": agi.cognition.world_model.state_dict(),
                "expert_world": agi.cognition.expert_world.get_state_dict(),
                "obs_abstraction": agi.cognition.obs_abstraction.get_state_dict(),
                # GameNN 是普通类（非 nn.Module），手动序列化
                "gamenn": {
                    "q_nets": [q.state_dict() for q in g.q_nets],
                    "strategy_weights": g.strategy_weights,
                    "strategy_scores": g.strategy_scores,
                    "strategy_counts": g.strategy_counts,
                    "game_matrix": g.game_matrix,
                    "strategy_update_counts": g.strategy_update_counts,
                    "confidence": g.confidence,
                    "epsilon": g.epsilon,
                },
                "confidence": agi.cognition.confidence,
                "hidden_dim": agi.cognition.lnn.hidden_dim,
                "input_dim": agi.cognition.lnn.input_dim,
                "growth_count": agi.cognition.growth_count,
            }
        except Exception as e:
            print(f"[PERSIST] cognition 保存失败: {e}")

    # 1b. MoE 专家路由（P1）— 确保已创建（延迟创建可能还没触发）
    if getattr(agi, 'moe', None) is not None:
        try:
            data["moe"] = agi.moe.get_state_dict()
        except Exception as e:
            print(f"[PERSIST] moe 保存失败: {e}")

    # 2. 价值系统（次级价值表）
    try:
        data["value_system"] = {
            name: entry
            for name, entry in agi.value_system.secondary_values.items()
        }
    except Exception as e:
        print(f"[PERSIST] value_system 保存失败: {e}")

    # 3. 空间记忆
    try:
        data["spatial_memory"] = {
            "nodes": [
                {"pos": list(k), "v": float(v)}
                for k, v in agi.spatial_memory.map_nodes.items()
            ] if hasattr(agi.spatial_memory, 'map_nodes') else []
        }
    except Exception as e:
        print(f"[PERSIST] spatial_memory 保存失败: {e}")

    # 4. 概念库
    try:
        data["concept_bank"] = {
            "concepts": [
                {"name": c.name, "kind": c.kind,
                 "vector": c.vector.tolist(), "freq": c.freq}
                for c in agi.concept_bank.concepts
            ]
        }
    except Exception as e:
        print(f"[PERSIST] concept_bank 保存失败: {e}")

    # 5. 元认知策略得分
    try:
        data["strategy_mgr"] = {
            "current": agi.strategy_mgr.current,
            "scores": agi.strategy_mgr.strategy_scores,
            "switch_count": agi.strategy_mgr.switch_count,
        }
    except Exception:
        pass

    # 6. 身体状态
    try:
        data["body"] = {
            "energy": float(agi.body.energy),
            "water": float(agi.body.water),
            "integrity": float(agi.body.integrity),
            "fatigue": float(agi.body.fatigue),
            "stress": float(agi.body.stress),
        }
    except Exception:
        pass

    # 7. 元数据（fitness 评估用）
    data["meta"] = {
        "tick": agi.tick,
        "survival_ticks": agi.survival_ticks,
        "peak_health": float(agi.peak_health),
        "alive": agi.alive,
    }

    torch.save(data, path)
    print(f"[PERSIST] checkpoint 已保存: {path} "
          f"(tick={agi.tick}, survival={agi.survival_ticks})")
    return path


def load_checkpoint(agi, path: str = None, tag: str = "latest") -> bool:
    """从 checkpoint 恢复 AGI 状态。返回是否成功。"""
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{tag}.pth")
    if not os.path.exists(path):
        print(f"[PERSIST] 无 checkpoint: {path}（新生命）")
        return False

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[PERSIST] checkpoint 加载失败: {e}")
        return False

    # 1. 认知核心权重
    if agi.cognition is not None and "cognition" in data:
        c = data["cognition"]
        try:
            # 先恢复到保存时的 hidden/input 维度（生长后的维度）
            saved_hidden = c.get("hidden_dim", 64)
            saved_input = c.get("input_dim", None)
            if agi.cognition.lnn.hidden_dim < saved_hidden:
                agi.cognition.lnn.grow(saved_hidden)
                agi.cognition.world_model.grow(saved_hidden)
            if saved_input is not None and agi.cognition.lnn.input_dim < saved_input:
                agi.cognition.lnn.grow_input(saved_input)
                agi.cognition.obs_dim = agi.cognition.lnn.input_dim - agi.cognition.self_state_dim
            agi.cognition.lnn.load_state_dict(c["lnn"])
            agi.cognition.world_model.load_state_dict(c["world_model"])
            if "expert_world" in c:
                try:
                    # 先 grow 到保存时的 hidden 维度（形状匹配）
                    if (agi.cognition.expert_world.input_dim < saved_hidden
                            and saved_hidden > 0):
                        agi.cognition.expert_world.grow(saved_hidden)
                    # 确保 heads 数量匹配（数唯一 head 索引，而非子 key 数）
                    ew_sd = c["expert_world"]
                    head_ids = set()
                    for k in ew_sd:
                        if k.startswith("heads."):
                            try:
                                head_ids.add(int(k.split(".")[1]))
                            except (ValueError, IndexError):
                                pass
                    n_saved = max(head_ids) + 1 if head_ids else 1
                    from cognition.temporal.world_experts import ExpertWorldHead
                    while len(agi.cognition.expert_world.heads) < n_saved:
                        agi.cognition.expert_world.heads.append(
                            ExpertWorldHead(agi.cognition.expert_world.input_dim))
                    # strict=False：容忍跨版本 extra/missing keys，失败时显式告警
                    try:
                        agi.cognition.expert_world.load_state_dict(ew_sd, strict=False)
                    except Exception as e:
                        print(f"[PERSIST] expert_world 加载告警: {e}")
                except Exception as e:
                    print(f"[PERSIST] expert_world 恢复失败: {e}")
            # 观测抽象层恢复
            if "obs_abstraction" in c:
                try:
                    agi.cognition.obs_abstraction.load_state_dict(c["obs_abstraction"])
                except Exception as e:
                    print(f"[PERSIST] obs_abstraction 恢复失败: {e}")
            # GameNN 手动恢复
            g = agi.cognition.gamenn
            gm = c["gamenn"]
            # 确保 state_dim 匹配保存时的维度
            if g.state_dim < saved_hidden:
                g.grow_state_dim(saved_hidden)
            # 校验 q_nets 数量（策略数不一致时显式告警，不静默截断）
            if len(g.q_nets) != len(gm["q_nets"]):
                print(f"[PERSIST] GameNN 策略数不匹配: "
                      f"保存 {len(gm['q_nets'])} vs 当前 {len(g.q_nets)}", flush=True)
            for q, sd in zip(g.q_nets, gm["q_nets"]):
                q.load_state_dict(sd)
            g.strategy_weights = gm["strategy_weights"]
            g.strategy_scores = gm["strategy_scores"]
            g.strategy_counts = gm["strategy_counts"]
            g.game_matrix = gm["game_matrix"]
            g.strategy_update_counts = gm["strategy_update_counts"]
            g.confidence = gm["confidence"]
            g.epsilon = gm["epsilon"]
            agi.cognition.confidence = c.get("confidence", 0.0)
            print(f"[PERSIST] 认知核心已恢复 (hidden={c.get('hidden_dim')})")
        except Exception as e:
            print(f"[PERSIST] cognition 恢复失败: {e}")

    # 1b. MoE 专家路由恢复（先确保已创建，再加载）
    if "moe" in data:
        try:
            if agi.moe is None:
                agi._ensure_moe()
            if agi.moe is not None:
                agi.moe.load_state_dict(data["moe"])
                print(f"[PERSIST] MoE 专家池已恢复 ({len(agi.moe.experts)} 专家)")
        except Exception as e:
            print(f"[PERSIST] moe 恢复失败: {e}")

    # 2. 价值系统
    if "value_system" in data:
        try:
            for name, entry in data["value_system"].items():
                agi.value_system.secondary_values[name] = entry
        except Exception:
            pass

    # 3. 空间记忆
    if "spatial_memory" in data and hasattr(agi.spatial_memory, 'map_nodes'):
        try:
            for node in data["spatial_memory"]["nodes"]:
                agi.spatial_memory.map_nodes[tuple(node["pos"])] = node["v"]
        except Exception:
            pass

    # 4. 概念库
    if "concept_bank" in data:
        try:
            from cognition.concept_bank import Concept
            for c in data["concept_bank"]["concepts"]:
                agi.concept_bank.concepts.append(
                    Concept(c["name"], c["kind"],
                            np.array(c["vector"]), c["freq"]))
        except Exception:
            pass

    # 5. 策略管理器
    if "strategy_mgr" in data:
        try:
            s = data["strategy_mgr"]
            agi.strategy_mgr.current = s["current"]
            agi.strategy_mgr.strategy_scores = s["scores"]
            agi.strategy_mgr.switch_count = s["switch_count"]
        except Exception:
            pass

    # 6. 身体状态
    if "body" in data:
        try:
            b = data["body"]
            agi.body.energy = b["energy"]
            agi.body.water = b["water"]
            agi.body.integrity = b["integrity"]
            agi.body.fatigue = b["fatigue"]
            agi.body.stress = b["stress"]
        except Exception:
            pass

    # 7. 元数据
    if "meta" in data:
        agi.tick = data["meta"].get("tick", 0)
        agi.survival_ticks = data["meta"].get("survival_ticks", 0)
        agi.peak_health = data["meta"].get("peak_health", 0.0)
        agi.alive = data["meta"].get("alive", True)

    print(f"[PERSIST] 已恢复: {path} "
          f"(tick={agi.tick}, survival={agi.survival_ticks})")
    return True


def list_checkpoints() -> list:
    """列出所有 checkpoint 文件"""
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    return sorted(os.listdir(CHECKPOINT_DIR))
