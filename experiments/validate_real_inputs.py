"""
真实轨道、空间天气、地面站和硬件输入的轻量校验工具。
真实数据导入前的轻量校验工具。

用途：
  1. 检查 VLEO TLE 文件是否能解析出两行轨道根数
  2. 检查空间天气 CSV 是否包含 F10.7/F10.7a/Ap 或 Kp
  3. 检查硬件基准 JSON 是否包含延迟、参数量、FLOPs 等论文可报告字段
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    # 仅脚本直跑时追加项目路径。
    sys.path.append(_PROJECT_ROOT)

import argparse
import json
from datetime import datetime
from pathlib import Path

import math

from experiments.generate_trace_csv import (
    _extract_tle_triplet,
    _tle_altitude_in_range,
    _iter_tle_triplets,
    _load_ground_station_csv,
    _load_space_weather_csv,
    _read_tle_lines_from_file,
    _tle_epoch_to_datetime,
)
from config import ORBITAL_CONFIG


def _count_tle_triplets(lines: list) -> int:
    return sum(1 for _ in _iter_tle_triplets(lines))


def _all_tle_triplets(lines: list) -> list:
    return list(_iter_tle_triplets(lines))


def _altitude_summary(line2: str) -> dict | None:
    try:
        parts = line2.split()
        eccentricity = float("0." + parts[4])
        mean_motion = float(parts[7])
        mu = 3.986004418e14
        earth_radius_m = 6371e3
        n_rad_s = mean_motion * 2.0 * math.pi / 86400.0
        semi_major_m = (mu / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
        perigee_km = (semi_major_m * (1.0 - eccentricity) - earth_radius_m) / 1000.0
        apogee_km = (semi_major_m * (1.0 + eccentricity) - earth_radius_m) / 1000.0
        return {
            "mean_motion_rev_day": mean_motion,
            "eccentricity": eccentricity,
            "perigee_km": perigee_km,
            "apogee_km": apogee_km,
            "mean_altitude_km": 0.5 * (perigee_km + apogee_km),
        }
    except Exception:
        return None


def _vleo_candidates(lines: list, limit: int = 10) -> list:
    candidates = []
    for name, _line1, line2 in _all_tle_triplets(lines):
        alt = _altitude_summary(line2)
        if not alt:
            continue
        min_altitude_km = float(ORBITAL_CONFIG.get("altitude_min_km", 150.0))
        is_candidate = (
            min_altitude_km <= alt["perigee_km"] <= 450.0 or
            min_altitude_km <= alt["mean_altitude_km"] <= 450.0 or
            alt["mean_motion_rev_day"] >= 15.2
        )
        if is_candidate:
            item = {"name": name, **alt}
            candidates.append(item)
    candidates.sort(key=lambda x: x["mean_altitude_km"])
    return candidates[:limit]


def _altitude_range_count(lines: list,
                          min_altitude_km: float = None,
                          max_altitude_km: float = None) -> int:
    return sum(
        1
        for _name, _line1, line2 in _all_tle_triplets(lines)
        if _tle_altitude_in_range(line2, min_altitude_km, max_altitude_km)
    )


def _altitude_range_epochs(lines: list,
                           min_altitude_km: float = None,
                           max_altitude_km: float = None) -> list:
    epochs = []
    for _name, line1, line2 in _all_tle_triplets(lines):
        if not _tle_altitude_in_range(line2, min_altitude_km, max_altitude_km):
            continue
        epoch = _tle_epoch_to_datetime(line1)
        if epoch is not None:
            epochs.append(epoch)
    return epochs


def _check_tle(path: str, satellite_name: str = "",
               min_altitude_km: float = None,
               max_altitude_km: float = None) -> dict:
    lines = _read_tle_lines_from_file(path)
    name, line1, line2 = _extract_tle_triplet(
        lines,
        satellite_name=satellite_name,
        min_altitude_km=min_altitude_km,
        max_altitude_km=max_altitude_km,
    )
    selected_altitude = _altitude_summary(line2)
    epochs = [
        _tle_epoch_to_datetime(l1)
        for _name, l1, _l2 in _all_tle_triplets(lines)
    ]
    epochs = [e for e in epochs if e is not None]
    accepted_epochs = _altitude_range_epochs(lines, min_altitude_km, max_altitude_km)
    return {
        "path": path,
        "name": name,
        "requested_name": satellite_name or None,
        "satellite_count": _count_tle_triplets(lines),
        "altitude_filter_km": {
            "min": min_altitude_km,
            "max": max_altitude_km,
            "accepted_count": _altitude_range_count(lines, min_altitude_km, max_altitude_km),
            "accepted_epoch_first_utc": None if not accepted_epochs else min(accepted_epochs).isoformat(),
            "accepted_epoch_last_utc": None if not accepted_epochs else max(accepted_epochs).isoformat(),
        },
        "line1_ok": line1.startswith("1 "),
        "line2_ok": line2.startswith("2 "),
        "selected_epoch_utc": (
            None if _tle_epoch_to_datetime(line1) is None
            else _tle_epoch_to_datetime(line1).isoformat()
        ),
        "epoch_first_utc": None if not epochs else min(epochs).isoformat(),
        "epoch_last_utc": None if not epochs else max(epochs).isoformat(),
        "selected_altitude": selected_altitude,
        "vleo_like_candidates": _vleo_candidates(lines),
    }


def _check_space_weather(path: str) -> dict:
    rows = _load_space_weather_csv(path)
    keys = set()
    for row in rows:
        keys.update(row.keys())
    has_msise_driver = "ap" in keys and ("f107" in keys or "f107a" in keys)
    return {
        "path": path,
        "rows": len(rows),
        "fields": sorted(k for k in keys if k != "utc"),
        "has_msise_driver": bool(has_msise_driver),
        "first_utc": rows[0]["utc"].isoformat(),
        "last_utc": rows[-1]["utc"].isoformat(),
    }


def _check_ground_stations(path: str, min_elevation_deg: float = 10.0) -> dict:
    stations = _load_ground_station_csv(path, min_elevation_deg)
    return {
        "path": path,
        "stations": len(stations),
        "names": [station.get("name", "GS") for station in stations],
        "min_elevation_deg": [
            float(station.get("min_elevation_deg", min_elevation_deg))
            for station in stations
        ],
        "has_link_budget_columns": any(
            station.get("eirp_dbw") is not None or
            station.get("g_over_t_db_k") is not None or
            station.get("bandwidth_hz") is not None or
            station.get("frequency_hz") is not None
            for station in stations
        ),
    }


def _coverage_status(tle_report: dict, weather_report: dict) -> dict:
    tle_start = datetime.fromisoformat(tle_report["epoch_first_utc"])
    tle_end = datetime.fromisoformat(tle_report["epoch_last_utc"])
    weather_start = datetime.fromisoformat(weather_report["first_utc"])
    weather_end = datetime.fromisoformat(weather_report["last_utc"])
    overlap_start = max(tle_start, weather_start)
    overlap_end = min(tle_end, weather_end)
    covers = weather_start <= tle_start and weather_end >= tle_end
    return {
        "space_weather_covers_tle": covers,
        "overlap_start_utc": None if overlap_start > overlap_end else overlap_start.isoformat(),
        "overlap_end_utc": None if overlap_start > overlap_end else overlap_end.isoformat(),
        "missing_before_hours": max(0.0, (weather_start - tle_start).total_seconds() / 3600.0),
        "missing_after_hours": max(0.0, (tle_end - weather_end).total_seconds() / 3600.0),
    }


def _check_hardware_json(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    bench = data.get("onboard_benchmark", data)
    if isinstance(bench, dict) and "model" in bench:
        bench = bench["model"]

    required = [
        "latency_ms_mean",
        "latency_ms_p95",
        "latency_ms_p99",
        "model_params_total",
        "actor_flops_estimate",
    ]
    missing = [k for k in required if k not in bench]
    return {
        "path": path,
        "missing_fields": missing,
        "ready_for_paper_table": len(missing) == 0,
    }


def validate(args) -> dict:
    report = {}
    if args.tle_file:
        report["tle"] = _check_tle(
            args.tle_file,
            args.satellite_name,
            min_altitude_km=args.min_tle_altitude_km,
            max_altitude_km=args.max_tle_altitude_km,
        )
    if args.space_weather_csv:
        report["space_weather"] = _check_space_weather(args.space_weather_csv)
    if args.ground_station_csv:
        report["ground_stations"] = _check_ground_stations(
            args.ground_station_csv,
            min_elevation_deg=args.min_elevation_deg,
        )
    if args.hardware_json:
        report["hardware"] = _check_hardware_json(args.hardware_json)

    if "tle" in report and "space_weather" in report:
        report["coverage"] = _coverage_status(report["tle"], report["space_weather"])

    if not report:
        raise ValueError("请至少提供 --tle_file / --space_weather_csv / --hardware_json 之一")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate real data inputs for VLEO paper experiments")
    parser.add_argument("--tle_file", default=None, help="真实 VLEO TLE 文件")
    parser.add_argument("--satellite_name", default="",
                        help="多颗 3LE 文件中要校验/选择的卫星名")
    parser.add_argument("--min_tle_altitude_km", type=float, default=None,
                        help="minimum mean TLE altitude accepted")
    parser.add_argument("--max_tle_altitude_km", type=float, default=None,
                        help="maximum mean TLE altitude accepted")
    parser.add_argument("--space_weather_csv", default=None,
                        help="空间天气 CSV，需包含 utc_iso/date 与 f107/f107a/ap/kp")
    parser.add_argument("--ground_station_csv", default=None,
                        help="ground station CSV for link-budget trace generation")
    parser.add_argument("--min_elevation_deg", type=float, default=10.0,
                        help="default ground-station elevation mask")
    parser.add_argument("--hardware_json", default=None,
                        help="evaluate_optimized.py --benchmark_onboard 输出的 JSON")
    validate(parser.parse_args())
