"""物理模型可信度验证（顶刊审稿 Issue #2）。

审稿意见要求：不要只在 config 注释里写"对齐真实"，而要有一个**独立可复现的
物理模型验证章节**。本脚本用环境真实使用的同一套模型（OrbitalDynamics /
SolarPanelModel / BatteryModel / PowerSubsystem / 大气子系统 + THERMAL_CONFIG），
在固定输入下输出五组验证表格，全部写入 results/physics_validation_*.json：

  1. 轨道衰减验证 (orbital decay)
       固定高度/面积/质量/Cd/F10.7，输出 **真实时间** 每天衰减率，
       与 GOCE / SLATS / 文献量级对比。不含 orbital_time_compression。
  2. 推进维持验证 (propulsion maintenance)
       180/200/220/250/270/300 km 下维持轨道 (thrust=drag) 所需平均功率、
       推力、比冲、推进剂消耗 (g/day, kg/month)，以及是否在 P_prop_max 内可行。
  3. 能量闭合验证 (energy closure)
       每轨道周期太阳能输入 vs 阴影期放电 + 推进/CPU/TX/平台功耗，
       检查是否满足电池容量约束。
  4. 热模型验证 (thermal)
       稳态温度、时间常数、散热面积/吸收率/发射率的敏感性。
  5. 推进剂寿命主指标 (fuel)
       g/orbit、kg/day、kg/month、kg/delivered-GB、固定推进剂预算下的任务寿命。
       同时给出 episode 内（含 orbital_time_compression=C）与真实时间两套口径，
       直接回应"一个 episode 130kg 燃料不合理"的质疑——episode 数字是被
       时间压缩 C 放大后的结果，真实速率见 real_time 列。

用法（CPU 即可，纯解析计算，无需 GPU/训练）:
    python experiments/physics_validation.py
    python experiments/physics_validation.py --delivered_gb_per_day 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

# Windows GBK 控制台无法编码中文表格，强制 UTF-8 输出，避免打印阶段乱码/中断。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from config import (
    DRAG_CONFIG,
    ENERGY_CONFIG,
    ORBITAL_CONFIG,
    PROPELLANT_CONFIG,
    THERMAL_CONFIG,
)
from environment.atmosphere import SpaceWeatherState, make_atmosphere
from environment.energy_model import BatteryModel, PowerSubsystem, SolarPanelModel
from environment.orbital_dynamics import OrbitalDynamics

G0 = 9.80665                # 标准重力加速度 (m/s²)
SIGMA = 5.670374419e-8      # Stefan-Boltzmann (W/m²/K⁴)
SECONDS_PER_DAY = 86400.0
DAYS_PER_MONTH = 30.0

# 真实 VLEO 参考量级（用于"对齐真实"的对照，非精确复刻）。
# GOCE: ~255-270 km，太阳活动低年衰减 ~数百 m/day，离子推进维持。
# SLATS (つばめ): 167-271 km 阶梯式 VLEO 演示，霍尔/离子电推进维持。
REFERENCE_NOTES = {
    "GOCE": "255-270 km nominal; ion thruster drag-compensated; decay 量级 ~10²-10³ m/day @ low/high solar",
    "SLATS": "167-271 km step-down VLEO demo; ion thruster; 271 km 长期维持，167 km 短期演示",
    "ISS": "~400 km; periodic reboost; decay ~50-100 m/day (高于 VLEO 操作区)",
}


def _altitude_set_km() -> list[float]:
    return [150.0, 170.0, 180.0, 200.0, 220.0, 250.0, 270.0, 300.0]


def _make_atm_for_f107(f107: float):
    """构造与环境一致的大气模型，并钉死到指定 F10.7（quiet Ap，无风暴）。"""
    atm = make_atmosphere()
    atm.set_space_weather(SpaceWeatherState(
        f107_daily=float(f107),
        f107_81avg=float(f107),
        ap=4.0,
    ))
    return atm


def validate_orbital_decay() -> dict:
    """1. 轨道衰减验证（无推进，真实时间口径）。"""
    dyn = OrbitalDynamics()
    f107_levels = {
        "low_solar_f107_70": 70.0,
        "nominal_f107_150": 150.0,
        "high_solar_f107_250": 250.0,
    }
    rows = []
    for alt_km in _altitude_set_km():
        h_m = alt_km * 1e3
        entry = {"altitude_km": alt_km}
        for label, f107 in f107_levels.items():
            dyn.atm = _make_atm_for_f107(f107)
            rho = float(dyn.atm.density(h_m))
            drag_n = float(dyn.drag_force(h_m))
            # altitude_decay_rate 返回 m/s（真实时间，不含 episode 时间压缩 C）。
            decay_ms = float(dyn.altitude_decay_rate(h_m))      # 负值
            decay_km_day = decay_ms * SECONDS_PER_DAY / 1e3
            entry[label] = {
                "density_kg_m3": rho,
                "drag_force_N": drag_n,
                "decay_rate_m_per_s": decay_ms,
                "decay_km_per_day": decay_km_day,
            }
        rows.append(entry)
    return {
        "description": "无推进自由衰减；decay 为真实时间 (m/s, km/day)，不含 orbital_time_compression",
        "fixed_params": {
            "Cd": float(dyn.Cd),
            "area_m2": float(dyn.A),
            "mass_kg": float(dyn.m),
            "atmosphere_model": DRAG_CONFIG.get("atmosphere_model", "vallado"),
        },
        "f107_levels": f107_levels,
        "rows": rows,
        "reference_notes": REFERENCE_NOTES,
    }


def validate_propulsion_maintenance() -> dict:
    """2. 推进维持验证（thrust = drag 平衡点）。"""
    dyn = OrbitalDynamics()
    dyn.atm = _make_atm_for_f107(DRAG_CONFIG.get("f107_nominal", 150.0))
    isp_s = float(ENERGY_CONFIG.get("propulsion_isp_s", 1500.0))
    eff = float(ENERGY_CONFIG.get("propulsion_efficiency", 0.65))
    p_prop_max = float(ENERGY_CONFIG.get("power_propulsion_max_w", 720.0))
    solar_max = float(ENERGY_CONFIG.get("solar_panel_power_w", 800.0))

    rows = []
    for alt_km in [180.0, 200.0, 220.0, 250.0, 270.0, 300.0]:
        h_m = alt_km * 1e3
        drag_n = float(dyn.drag_force(h_m))
        # thrust = P·eff/(Isp·g0)。令 thrust == drag → 解出所需功率。
        p_required_w = drag_n * isp_s * G0 / max(eff, 1e-9)
        feasible = p_required_w <= p_prop_max
        # 实际能给到的推力（功率夹到 prop_max）。
        p_applied_w = min(p_required_w, p_prop_max)
        thrust_applied_n = dyn.propulsion_thrust(p_applied_w)
        thrust_required_n = drag_n
        # mdot = thrust/(Isp·g0)（按维持所需推力，真实时间）。
        mdot_kg_s = thrust_required_n / (isp_s * G0)
        period_s = float(np.sqrt((dyn.R_e + h_m) ** 3 / dyn.mu) * 2.0 * np.pi)
        rows.append({
            "altitude_km": alt_km,
            "drag_force_N": drag_n,
            "thrust_required_N": thrust_required_n,
            "power_required_W": p_required_w,
            "power_required_frac_of_prop_max": p_required_w / max(p_prop_max, 1e-9),
            "power_required_frac_of_solar": p_required_w / max(solar_max, 1e-9),
            "maintainable_within_prop_max": bool(feasible),
            "thrust_at_prop_max_N": float(dyn.propulsion_thrust(p_prop_max)),
            "mdot_kg_per_s": mdot_kg_s,
            "fuel_g_per_day": mdot_kg_s * SECONDS_PER_DAY * 1e3,
            "fuel_kg_per_month": mdot_kg_s * SECONDS_PER_DAY * DAYS_PER_MONTH,
            "orbital_period_s": period_s,
            "fuel_g_per_orbit": mdot_kg_s * period_s * 1e3,
        })
    return {
        "description": "维持轨道 (thrust=drag) 所需平均功率/推力/比冲/推进剂；真实时间口径",
        "params": {
            "isp_s": isp_s,
            "propulsion_efficiency": eff,
            "power_propulsion_max_w": p_prop_max,
            "solar_panel_power_w": solar_max,
            "thrust_at_prop_max_mN": float(dyn.propulsion_thrust(p_prop_max)) * 1e3,
        },
        "rows": rows,
    }


def validate_energy_closure() -> dict:
    """3. 能量闭合验证（每轨道周期太阳能 vs 放电/负载）。"""
    dyn = OrbitalDynamics()
    dyn.atm = _make_atm_for_f107(DRAG_CONFIG.get("f107_nominal", 150.0))
    solar = SolarPanelModel()
    isp_s = float(ENERGY_CONFIG.get("propulsion_isp_s", 1500.0))
    eff = float(ENERGY_CONFIG.get("propulsion_efficiency", 0.65))
    p_prop_max = float(ENERGY_CONFIG.get("power_propulsion_max_w", 720.0))
    p_baseline = float(ENERGY_CONFIG.get("power_baseline_w", 15.0))
    p_cpu_max = float(ENERGY_CONFIG.get("power_cpu_max_w", 25.0))
    p_tx_max = float(ENERGY_CONFIG.get("power_tx_max_w", 35.0))
    battery_wh = float(ENERGY_CONFIG.get("battery_capacity_wh", 500.0))
    usable_wh = battery_wh * (ENERGY_CONFIG.get("battery_max_soc", 0.95)
                             - ENERGY_CONFIG.get("battery_min_soc", 0.15))

    # 标称阴影占比来自 config（55min 日照 / 35min 阴影 / 90min 周期）。
    eclipse_frac = (ORBITAL_CONFIG.get("eclipse_duration_min", 35.0)
                    / ORBITAL_CONFIG.get("orbital_period_min", 90.0))

    rows = []
    for alt_km in [200.0, 250.0, 300.0]:
        h_m = alt_km * 1e3
        period_s = float(np.sqrt((dyn.R_e + h_m) ** 3 / dyn.mu) * 2.0 * np.pi)
        sunlit_s = period_s * (1.0 - eclipse_frac)
        eclipse_s = period_s * eclipse_frac

        drag_n = float(dyn.drag_force(h_m))
        p_prop_maintain = min(drag_n * isp_s * G0 / max(eff, 1e-9), p_prop_max)

        # 代表性占空：阴影期只维持轨道+平台（不成像/不下传）；日照期推进+CPU 半载。
        load_eclipse_w = p_prop_maintain + p_baseline
        load_sunlit_w = p_prop_maintain + p_baseline + 0.5 * p_cpu_max + 0.5 * p_tx_max

        # 太阳能：日照期峰值发电（正弦剖面平均约 2/π，但 output_power 已含相位强度，
        # 这里用平均日照强度 0.637≈2/π 近似积分平均）。
        solar_avg_w = solar.output_power(sunlit_fraction=2.0 / np.pi)
        energy_in_wh = solar_avg_w * (sunlit_s / 3600.0)
        energy_out_wh = (load_sunlit_w * sunlit_s + load_eclipse_w * eclipse_s) / 3600.0
        eclipse_discharge_wh = load_eclipse_w * eclipse_s / 3600.0

        rows.append({
            "altitude_km": alt_km,
            "orbital_period_s": period_s,
            "sunlit_s": sunlit_s,
            "eclipse_s": eclipse_s,
            "prop_maintain_power_W": p_prop_maintain,
            "load_sunlit_W": load_sunlit_w,
            "load_eclipse_W": load_eclipse_w,
            "solar_avg_power_W": solar_avg_w,
            "energy_in_per_orbit_Wh": energy_in_wh,
            "energy_out_per_orbit_Wh": energy_out_wh,
            "net_energy_per_orbit_Wh": energy_in_wh - energy_out_wh,
            "eclipse_discharge_Wh": eclipse_discharge_wh,
            "eclipse_discharge_frac_of_usable": eclipse_discharge_wh / max(usable_wh, 1e-9),
            "energy_positive_balance": bool(energy_in_wh >= energy_out_wh),
            "eclipse_within_battery": bool(eclipse_discharge_wh <= usable_wh),
        })
    return {
        "description": "每轨道周期能量收支；正平衡 + 阴影放电 < 可用容量 → 能量闭合可行",
        "params": {
            "battery_capacity_wh": battery_wh,
            "usable_energy_wh": usable_wh,
            "solar_panel_power_w": float(ENERGY_CONFIG.get("solar_panel_power_w", 800.0)),
            "eclipse_fraction": eclipse_frac,
        },
        "rows": rows,
    }


def validate_thermal() -> dict:
    """4. 热模型验证（稳态温度 + 时间常数 + 敏感性）。"""
    cfg = THERMAL_CONFIG
    C = float(cfg.get("thermal_capacity_j_per_k", 18000.0))
    area = float(cfg.get("radiator_area_m2", 0.22))
    emis = float(cfg.get("radiator_emissivity", 0.92))
    absorb = float(cfg.get("solar_absorptivity", 0.20))
    sun_area = float(cfg.get("sunlit_absorbing_area_m2", 0.08))
    solar_flux = float(cfg.get("solar_flux_w_m2", 1361.0))
    heat_frac = float(cfg.get("electronics_heat_fraction", 0.90))
    ambient_k = float(cfg.get("ambient_temp_c", -20.0)) + 273.15

    p_cpu_max = float(ENERGY_CONFIG.get("power_cpu_max_w", 25.0))
    p_tx_max = float(ENERGY_CONFIG.get("power_tx_max_w", 35.0))
    p_baseline = float(ENERGY_CONFIG.get("power_baseline_w", 15.0))

    def steady_state_temp_c(internal_w: float, sunlit: bool) -> float:
        """解 εσA(T⁴-T_amb⁴) = Q_in 的稳态温度。"""
        q_solar = absorb * solar_flux * sun_area if sunlit else 0.0
        q_in = heat_frac * internal_w + q_solar
        # T⁴ = T_amb⁴ + q_in/(εσA)
        t4 = ambient_k ** 4 + q_in / max(emis * SIGMA * area, 1e-12)
        return float(t4 ** 0.25 - 273.15)

    # 代表性负载场景。
    scenarios = {
        "idle_baseline": p_baseline,
        "cpu_half": p_baseline + 0.5 * p_cpu_max,
        "cpu_tx_full": p_baseline + p_cpu_max + p_tx_max,
    }
    steady = {}
    for name, load in scenarios.items():
        steady[name] = {
            "internal_load_W": load,
            "steady_temp_eclipse_C": steady_state_temp_c(load, sunlit=False),
            "steady_temp_sunlit_C": steady_state_temp_c(load, sunlit=True),
        }

    # 时间常数：围绕 cpu_tx_full 日照稳态点线性化 τ = C / (4εσA T³)。
    t_lin_k = steady_state_temp_c(scenarios["cpu_tx_full"], sunlit=True) + 273.15
    tau_s = C / max(4.0 * emis * SIGMA * area * t_lin_k ** 3, 1e-12)

    # 敏感性：散热面积 / 发射率 ±20% 对 cpu_tx_full 日照稳态温度的影响。
    base_temp = steady_state_temp_c(scenarios["cpu_tx_full"], sunlit=True)
    sensitivity = {}
    for frac in (0.8, 1.2):
        area_s = area * frac
        emis_s = min(emis * frac, 0.99)
        q_in = heat_frac * scenarios["cpu_tx_full"] + absorb * solar_flux * sun_area
        t_area = (ambient_k ** 4 + q_in / (emis * SIGMA * area_s)) ** 0.25 - 273.15
        t_emis = (ambient_k ** 4 + q_in / (emis_s * SIGMA * area)) ** 0.25 - 273.15
        sensitivity[f"x{frac}"] = {
            "radiator_area_m2": area_s,
            "steady_temp_C_area_scaled": float(t_area),
            "emissivity": emis_s,
            "steady_temp_C_emis_scaled": float(t_emis),
        }

    return {
        "description": "一阶集总热模型稳态/时间常数/敏感性；非仅一阶温度状态",
        "params": {
            "thermal_capacity_j_per_k": C,
            "radiator_area_m2": area,
            "radiator_emissivity": emis,
            "solar_absorptivity": absorb,
            "sunlit_absorbing_area_m2": sun_area,
            "electronics_heat_fraction": heat_frac,
            "warning_temp_c": cfg.get("warning_temp_c"),
            "max_temp_c": cfg.get("max_temp_c"),
            "critical_temp_c": cfg.get("critical_temp_c"),
        },
        "steady_state": steady,
        "time_constant_s": float(tau_s),
        "time_constant_min": float(tau_s / 60.0),
        "linearization_temp_C": float(t_lin_k - 273.15),
        "base_cpu_tx_full_sunlit_temp_C": float(base_temp),
        "sensitivity": sensitivity,
    }


def validate_fuel_metrics(delivered_gb_per_day: float | None = None) -> dict:
    """5. 推进剂寿命主指标（episode 压缩 vs 真实时间两套口径）。"""
    dyn = OrbitalDynamics()
    dyn.atm = _make_atm_for_f107(DRAG_CONFIG.get("f107_nominal", 150.0))
    isp_s = float(ENERGY_CONFIG.get("propulsion_isp_s", 1500.0))
    initial_kg = float(PROPELLANT_CONFIG.get("initial_mass_kg", 30.0))
    C = float(PROPELLANT_CONFIG.get("orbital_time_compression", 1.0))

    nominal_alt_km = float(ORBITAL_CONFIG.get("altitude_nominal_km", 250.0))
    h_m = nominal_alt_km * 1e3
    drag_n = float(dyn.drag_force(h_m))
    thrust_required_n = drag_n
    mdot_kg_s = thrust_required_n / (isp_s * G0)           # 真实时间消耗率
    period_s = float(np.sqrt((dyn.R_e + h_m) ** 3 / dyn.mu) * 2.0 * np.pi)

    real = {
        "fuel_g_per_orbit": mdot_kg_s * period_s * 1e3,
        "fuel_kg_per_day": mdot_kg_s * SECONDS_PER_DAY,
        "fuel_kg_per_month": mdot_kg_s * SECONDS_PER_DAY * DAYS_PER_MONTH,
        "mission_lifetime_days": initial_kg / max(mdot_kg_s * SECONDS_PER_DAY, 1e-12),
        "mission_lifetime_months": (initial_kg / max(mdot_kg_s * SECONDS_PER_DAY, 1e-12)) / DAYS_PER_MONTH,
    }
    # episode 口径：环境用 consumed = mdot·dt·C，等效真实速率被放大 C 倍。
    compressed = {
        "orbital_time_compression_C": C,
        "fuel_kg_per_day_episode_clock": real["fuel_kg_per_day"] * C,
        "mission_lifetime_days_episode_clock": real["mission_lifetime_days"] / max(C, 1e-9),
        "note": (
            "episode 内燃料速率被 orbital_time_compression=C 放大，"
            "因此 6h episode 内可消耗数十 kg——这是刻意的时间压缩，"
            "真实物理速率见 real_time。审稿应引用 real_time 列。"
        ),
    }
    if delivered_gb_per_day is not None and delivered_gb_per_day > 0:
        real["fuel_kg_per_delivered_gb"] = real["fuel_kg_per_day"] / float(delivered_gb_per_day)
        real["delivered_gb_per_day_assumed"] = float(delivered_gb_per_day)
    else:
        real["fuel_kg_per_delivered_gb"] = None
        real["delivered_gb_per_day_note"] = (
            "传 --delivered_gb_per_day 以计算 kg/GB；或由 multi_horizon_eval 的 24h 下传量推导"
        )

    return {
        "description": "推进剂主指标；real_time 为物理真值，episode_clock 为时间压缩后口径",
        "params": {
            "initial_propellant_kg": initial_kg,
            "nominal_altitude_km": nominal_alt_km,
            "isp_s": isp_s,
            "mdot_kg_per_s_real": mdot_kg_s,
        },
        "real_time": real,
        "episode_clock": compressed,
    }


def _print_section(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def run(delivered_gb_per_day: float | None = None) -> dict:
    report = {
        "__meta__": {
            "timestamp": datetime.now().isoformat(),
            "purpose": "顶刊 Issue#2 物理模型可信度验证",
        },
        "orbital_decay": validate_orbital_decay(),
        "propulsion_maintenance": validate_propulsion_maintenance(),
        "energy_closure": validate_energy_closure(),
        "thermal": validate_thermal(),
        "fuel_metrics": validate_fuel_metrics(delivered_gb_per_day),
    }

    _print_section("1. 轨道衰减验证 (真实时间, 无推进)")
    print(f"  {'高度(km)':>8} {'ρ@F107=150':>14} {'drag(N)':>10} "
          f"{'衰减(km/day)':>14}")
    for r in report["orbital_decay"]["rows"]:
        nom = r["nominal_f107_150"]
        print(f"  {r['altitude_km']:>8.0f} {nom['density_kg_m3']:>14.3e} "
              f"{nom['drag_force_N']:>10.4f} {nom['decay_km_per_day']:>14.2f}")

    _print_section("2. 推进维持验证 (thrust=drag)")
    print(f"  {'高度(km)':>8} {'drag(N)':>10} {'功率(W)':>10} "
          f"{'/prop_max':>10} {'g/day':>10} {'可维持':>7}")
    for r in report["propulsion_maintenance"]["rows"]:
        print(f"  {r['altitude_km']:>8.0f} {r['drag_force_N']:>10.4f} "
              f"{r['power_required_W']:>10.1f} "
              f"{r['power_required_frac_of_prop_max']:>10.2f} "
              f"{r['fuel_g_per_day']:>10.2f} "
              f"{'是' if r['maintainable_within_prop_max'] else '否':>7}")

    _print_section("3. 能量闭合验证 (每轨道周期)")
    print(f"  {'高度(km)':>8} {'入(Wh)':>10} {'出(Wh)':>10} {'净(Wh)':>10} "
          f"{'阴影放电(Wh)':>14} {'闭合':>6}")
    for r in report["energy_closure"]["rows"]:
        ok = "是" if (r["energy_positive_balance"] and r["eclipse_within_battery"]) else "否"
        print(f"  {r['altitude_km']:>8.0f} {r['energy_in_per_orbit_Wh']:>10.1f} "
              f"{r['energy_out_per_orbit_Wh']:>10.1f} "
              f"{r['net_energy_per_orbit_Wh']:>10.1f} "
              f"{r['eclipse_discharge_Wh']:>14.1f} {ok:>6}")

    _print_section("4. 热模型验证")
    th = report["thermal"]
    for name, s in th["steady_state"].items():
        print(f"  {name:<16} 内部负载={s['internal_load_W']:>5.1f}W  "
              f"稳态(阴影)={s['steady_temp_eclipse_C']:>6.1f}°C  "
              f"稳态(日照)={s['steady_temp_sunlit_C']:>6.1f}°C")
    print(f"  时间常数 τ ≈ {th['time_constant_min']:.1f} min "
          f"(线性化 @ {th['linearization_temp_C']:.1f}°C)")

    _print_section("5. 推进剂寿命主指标")
    fm = report["fuel_metrics"]
    rt = fm["real_time"]
    print(f"  [真实时间] g/orbit={rt['fuel_g_per_orbit']:.3f}  "
          f"kg/day={rt['fuel_kg_per_day']:.4f}  kg/month={rt['fuel_kg_per_month']:.3f}")
    print(f"  [真实时间] 固定 {fm['params']['initial_propellant_kg']:.0f}kg 推进剂寿命 ≈ "
          f"{rt['mission_lifetime_days']:.0f} 天 ({rt['mission_lifetime_months']:.1f} 月)")
    print(f"  [episode 口径] 时间压缩 C={fm['episode_clock']['orbital_time_compression_C']:.0f}× → "
          f"episode 时钟 kg/day={fm['episode_clock']['fuel_kg_per_day_episode_clock']:.1f}")
    if rt.get("fuel_kg_per_delivered_gb") is not None:
        print(f"  [真实时间] kg/delivered-GB = {rt['fuel_kg_per_delivered_gb']:.5f} "
              f"(假设 {rt['delivered_gb_per_day_assumed']:.1f} GB/day)")

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/physics_validation_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="物理模型可信度验证")
    parser.add_argument("--delivered_gb_per_day", type=float, default=None,
                        help="可选：用于计算 kg/delivered-GB 的日均下传量 (GB)")
    args = parser.parse_args()
    run(args.delivered_gb_per_day)
