"""
P2: 经验 DNA — 跨代信息载体

session 结束时，不是复制权重（那是复读机），而是提取"规律/技能/价值倾向"
的压缩表示，写入 DNA 文件。新 session 加载 DNA 作为先验（贝叶斯初始化），
用自己的生活重新验证、修正、扩展。

DNA 结构：
  {
    "generation": N,
    "rules":    [{text, confidence, sources, context}],   # 规律库
    "skills":   [{text, proficiency, success, context}],  # 技能库
    "values":   {stimulus: {value, confidence}},          # 次级价值
    "growth":   {situations: [...], average_growths},     # 生长轨迹
    "fitness":  {survival, peak_health, ticks}            # 本代表现
  }
"""

import os
import json
import numpy as np

DNA_FILE = "experience_dna.json"


def _default_dna() -> dict:
    return {
        "generation": 0,
        "rules": [],
        "skills": [],
        "values": {},
        "growth": {"situations": [], "average_growths": 0},
        "fitness": {"survival": 0, "peak_health": 0.0, "ticks": 0},
    }


def load_dna(path: str = DNA_FILE) -> dict:
    if not os.path.exists(path):
        return _default_dna()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return _default_dna()


def save_dna(dna: dict, path: str = DNA_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dna, f, ensure_ascii=False, indent=2)


def extract_dna(agi, generation: int = None) -> dict:
    """
    从 AGI 实例提取经验 DNA（session 结束时调用）。
    提取的是"规律"，不是权重快照。
    """
    dna = load_dna()
    if generation is not None:
        dna["generation"] = generation
    else:
        dna["generation"] = dna.get("generation", 0) + 1

    # ─── 规律库：从空间记忆/概念库提取稳定规律 ───
    rules = []
    try:
        # 空间记忆中的高价值节点 → 资源位置规律
        if hasattr(agi, 'spatial_memory') and hasattr(agi.spatial_memory, 'map_nodes'):
            nodes = agi.spatial_memory.map_nodes
            if nodes:
                # 找出能量回报高的节点（正价值）
                high_val = [(list(k), v) for k, v in nodes.items() if v > 0.3]
                if high_val:
                    rules.append({
                        "text": "资源分布在特定空间位置（高价值节点）",
                        "confidence": min(1.0, len(high_val) / 10.0),
                        "sources": 1,
                        "context": "spatial",
                        "prototype": high_val[:5],
                    })
    except Exception:
        pass

    # 概念库中的高频概念 → 行为规律
    try:
        top_concepts = agi.concept_bank.get_stats().get("top_concepts", [])
        if top_concepts:
            rules.append({
                "text": f"高频行为概念: {top_concepts}",
                "confidence": 0.5,
                "sources": 1,
                "context": "concept",
                "prototype": top_concepts[:5],
            })
    except Exception:
        pass

    # 合并旧规则（保留来源数累计）
    old_rules = dna.get("rules", [])
    for r in rules:
        matched = False
        for old in old_rules:
            if old.get("text") == r["text"]:
                old["sources"] = old.get("sources", 1) + 1
                old["confidence"] = min(1.0, old.get("confidence", 0.5) + 0.1)
                matched = True
                break
        if not matched:
            old_rules.append(r)
    dna["rules"] = old_rules[:20]  # 规则数上限

    # ─── 技能库：从价值系统/决策统计提取 ───
    skills = []
    try:
        # 价值系统的高价值刺激 → 技能倾向
        for name, entry in agi.value_system.secondary_values.items():
            if entry["value"] > 0.5 and entry["updates"] > 5:
                skills.append({
                    "text": f"偏好刺激: {name}（价值 {entry['value']:.2f}）",
                    "proficiency": entry["confidence"],
                    "success": entry["updates"],
                    "context": "value",
                })
    except Exception:
        pass
    old_skills = dna.get("skills", [])
    for s in skills:
        matched = False
        for old in old_skills:
            if old.get("text") == s["text"]:
                old["success"] = old.get("success", 0) + s["success"]
                old["proficiency"] = min(1.0, old.get("proficiency", 0.5) + 0.1)
                matched = True
                break
        if not matched:
            old_skills.append(s)
    dna["skills"] = old_skills[:20]

    # ─── 价值倾向：次级价值表 ───
    try:
        for name, entry in agi.value_system.secondary_values.items():
            cur = dna["values"].setdefault(name, {"value": 0.5, "confidence": 0.1})
            # 跨代加权合并：新经验占 30%
            cur["value"] = 0.7 * cur["value"] + 0.3 * entry["value"]
            cur["confidence"] = min(1.0, cur["confidence"] + 0.1 * entry["confidence"])
    except Exception:
        pass

    # ─── 生长轨迹 ───
    try:
        gc = agi.cognition.growth_count if agi.cognition else 0
        dna["growth"]["average_growths"] = gc
        dna["growth"]["situations"].append({
            "ticks": agi.tick,
            "growths": gc,
            "survival": agi.survival_ticks,
        })
        dna["growth"]["situations"] = dna["growth"]["situations"][-10:]
    except Exception:
        pass

    # ─── fitness（选择压力数据） ───
    dna["fitness"] = {
        "survival": agi.survival_ticks,
        "peak_health": float(agi.peak_health),
        "ticks": agi.tick,
    }

    save_dna(dna)
    print(f"[DNA] 第 {dna['generation']} 代经验已提取: "
          f"{len(dna['rules'])} 规则, {len(dna['skills'])} 技能, "
          f"存活 {agi.survival_ticks} tick")
    return dna


def apply_dna(agi, path: str = DNA_FILE) -> bool:
    """
    新 session 初始化时应用 DNA 先验（不是冻结，是可修正的倾向）。
    """
    dna = load_dna(path)
    if dna["generation"] == 0 and not dna["rules"] and not dna["skills"]:
        return False

    # 应用价值倾向（次级价值初始值，可被新经验修正）
    try:
        for name, entry in dna["values"].items():
            if name in agi.value_system.secondary_values:
                v = agi.value_system.secondary_values[name]
                v["value"] = entry["value"]
                v["confidence"] = max(0.1, entry["confidence"] * 0.5)
    except Exception:
        pass

    print(f"[DNA] 应用第 {dna['generation']} 代先验: "
          f"{len(dna['rules'])} 规则, {len(dna['skills'])} 技能")
    return True
