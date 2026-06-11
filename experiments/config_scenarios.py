"""论文级泛化场景预设（eval-time，config 覆盖 + 可逆恢复）。

evaluate_on_env 每 episode 新建 VLEOSatelliteEnv，其 __init__ 实时读 config 字典，
故在评估前修改 config 即切换场景，退出 context 还原，不污染后续评估。

4 个场景（对齐论文泛化实验）：
  - sparse_comm        : 稀疏通信窗口（regional 站网 + 更高最低仰角）
  - energy_constrained : 能源受限（电池/太阳能砍小）
  - high_density       : 高任务密度（数据到达率 ×2.5）
  - sparse_high_value  : 稀疏高价值（高价值场景 arrival 大幅下调）
  - nominal            : 标称（不改 config，作对照）

用法：
    from experiments.config_scenarios import scenario, SCENARIOS
    with scenario("sparse_comm"):
        result = evaluate_on_env(fn, n_episodes=20, seed_offset=42)
"""
import sys, os
import copy
from contextlib import contextmanager

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _ROOT not in sys.path:
    sys.path.append(_ROOT)

import config as C

SCENARIOS = ["nominal", "sparse_comm", "energy_constrained", "high_density", "sparse_high_value"]

# 每个场景：{config_dict_name: {key: new_value}}；scene_profiles 用特殊处理（乘子）。
_HIGH_VALUE_SCENES = ("urban", "disaster", "military", "emergency_disaster")


def _overrides(name: str) -> dict:
    if name == "nominal":
        return {}
    if name == "sparse_comm":
        # regional 站网（5 站 vs global 19 站）+ 最低仰角 5°→12°，过顶窗口更短更稀。
        return {
            "GROUND_STATION_CONFIG": {
                "profile": "regional",
                "stations": C.GROUND_STATION_PROFILES["regional"],
                "min_elevation_deg": 12.0,
            },
        }
    if name == "energy_constrained":
        # 电池 500→320Wh、太阳能 800→560W：能量预算明显收紧但不必然坠毁。
        return {
            "ENERGY_CONFIG": {
                "battery_capacity_wh": 320.0,
                "solar_panel_power_w": 560.0,
            },
        }
    if name == "high_density":
        # 数据到达率 5.0→12.5 MB/s（×2.5）：处理/下传/队列压力全面抬升。
        return {
            "QUEUE_CONFIG": {"data_arrival_rate_mbs": 12.5},
        }
    if name == "sparse_high_value":
        # 高价值场景（urban/disaster/military/emergency）arrival_multiplier ×0.25：
        # 高价值任务变稀疏，考验策略对稀有高价值的捕获与交付。
        return {"__scene_arrival_scale__": {s: 0.25 for s in _HIGH_VALUE_SCENES}}
    raise ValueError(f"unknown scenario: {name}")


@contextmanager
def scenario(name: str):
    """进入场景：深拷贝快照受影响 config，应用覆盖；退出还原。"""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name}; choose from {SCENARIOS}")
    ov = _overrides(name)
    # 快照所有可能被改的 config 字典（深拷贝）
    touched = ["GROUND_STATION_CONFIG", "ENERGY_CONFIG", "QUEUE_CONFIG", "TASK_CONFIG"]
    snapshot = {n: copy.deepcopy(getattr(C, n)) for n in touched}
    try:
        for cfg_name, kv in ov.items():
            if cfg_name == "__scene_arrival_scale__":
                profiles = C.TASK_CONFIG["scene_profiles"]
                for scene, factor in kv.items():
                    if scene in profiles:
                        profiles[scene]["arrival_multiplier"] = (
                            float(profiles[scene].get("arrival_multiplier", 1.0)) * float(factor))
                continue
            target = getattr(C, cfg_name)
            target.update(kv)
        yield name
    finally:
        # 还原（in-place，保持其它模块持有的引用有效）
        for n, snap in snapshot.items():
            d = getattr(C, n)
            d.clear()
            d.update(snap)


if __name__ == "__main__":
    # 自检：进入/退出各场景后 config 完全还原
    import json
    base = {n: copy.deepcopy(getattr(C, n)) for n in
            ["GROUND_STATION_CONFIG", "ENERGY_CONFIG", "QUEUE_CONFIG", "TASK_CONFIG"]}
    for s in SCENARIOS:
        with scenario(s) as sc:
            gs = C.GROUND_STATION_CONFIG.get("profile")
            bat = C.ENERGY_CONFIG.get("battery_capacity_wh")
            arr = C.QUEUE_CONFIG.get("data_arrival_rate_mbs")
            mil = C.TASK_CONFIG["scene_profiles"]["military"]["arrival_multiplier"]
            print(f"[{s:18}] gs_profile={gs} battery={bat} arrival={arr} military_arr={mil:.3f}")
        # 还原校验
        for n in base:
            assert json.dumps(getattr(C, n), sort_keys=True, default=str) == \
                   json.dumps(base[n], sort_keys=True, default=str), f"{s}: {n} not restored!"
    print("OK: all scenarios apply + restore cleanly")
