"""
生成可用于 experiments/robustness.py 的 trace-driven 外生扰动 CSV。

输出列：
  utc_iso, step, sunlit, solar_scale, rho_scale, in_window, tx_capacity_mbps,
  altitude_km, lat_deg, lon_deg, f107a, f107, ap, kp

说明：
  1) 若提供 --tle_file 且安装了 skyfield，可基于轨道计算 altitude_km，
     并在提供 --ephemeris 时计算 sunlit。
  2) 若缺少 skyfield 或未提供 TLE，自动退化为周期性近似，不阻塞流程。
  3) rho_scale 默认采用指数大气近似（按高度变化），用于鲁棒性评估扰动。
  4) 若提供 --space_weather_csv，可逐时刻读取 F10.7/F10.7a/Ap/Kp 并驱动 MSISE。
"""

import argparse
import csv
import importlib
import math
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _import_skyfield_api():
    # Skyfield 是可选依赖；缺失时脚本仍可用周期近似生成扰动 trace。
    try:
        api = importlib.import_module("skyfield.api")
        return api.EarthSatellite, api.load, api.wgs84
    except Exception:  # pragma: no cover
        return None, None, None


EarthSatellite, load, wgs84 = _import_skyfield_api()


DEFAULT_GROUND_STATIONS = [
    # 中国站保留为本地区覆盖基线。
    {"name": "GS-Beijing", "lat_deg": 39.9, "lon_deg": 116.4, "alt_m": 50.0},
    {"name": "GS-Shanghai", "lat_deg": 31.2, "lon_deg": 121.5, "alt_m": 10.0},
    {"name": "GS-Guangzhou", "lat_deg": 23.1, "lon_deg": 113.3, "alt_m": 20.0},
    # 真实 trace 默认使用全球多站网，避免 SLATS/GOCE 在单一区域站网下过度稀疏。
    {"name": "GS-Svalbard", "lat_deg": 78.23, "lon_deg": 15.40, "alt_m": 460.0},
    {"name": "GS-Troll", "lat_deg": -72.01, "lon_deg": 2.54, "alt_m": 1270.0},
    {"name": "GS-Inuvik", "lat_deg": 68.32, "lon_deg": -133.55, "alt_m": 68.0},
    {"name": "GS-Alaska", "lat_deg": 64.98, "lon_deg": -147.51, "alt_m": 210.0},
    {"name": "GS-Kiruna", "lat_deg": 67.89, "lon_deg": 21.06, "alt_m": 420.0},
    {"name": "GS-Maspalomas", "lat_deg": 27.76, "lon_deg": -15.63, "alt_m": 205.0},
    {"name": "GS-Wallops", "lat_deg": 37.94, "lon_deg": -75.47, "alt_m": 10.0},
    {"name": "GS-Hawaii", "lat_deg": 19.01, "lon_deg": -155.67, "alt_m": 340.0},
    {"name": "GS-Santiago", "lat_deg": -33.45, "lon_deg": -70.67, "alt_m": 570.0},
    {"name": "GS-PuntaArenas", "lat_deg": -53.16, "lon_deg": -70.91, "alt_m": 34.0},
    {"name": "GS-Perth", "lat_deg": -31.80, "lon_deg": 115.89, "alt_m": 30.0},
    {"name": "GS-Singapore", "lat_deg": 1.35, "lon_deg": 103.82, "alt_m": 20.0},
    {"name": "GS-Tokyo", "lat_deg": 35.68, "lon_deg": 139.76, "alt_m": 40.0},
    {"name": "GS-Hartebeesthoek", "lat_deg": -25.89, "lon_deg": 27.71, "alt_m": 1400.0},
    {"name": "GS-Kourou", "lat_deg": 5.24, "lon_deg": -52.77, "alt_m": 30.0},
    {"name": "GS-Redu", "lat_deg": 50.00, "lon_deg": 5.15, "alt_m": 370.0},
]


def _import_nrlmsise_api():
    # NRLMSISE-00 也是可选依赖；没有安装时会退回指数大气近似。
    try:
        return importlib.import_module("nrlmsise00")
    except Exception:  # pragma: no cover
        return None


nrlmsise00 = _import_nrlmsise_api()


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(row: dict, keys: list, default=None):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _kp_to_ap(kp: float) -> float:
    kp_grid = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    # NOAA 常用 Kp→ap 近似表，输入允许小数并做线性插值。
    kp_grid = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    ap_grid = np.array([0, 4, 7, 15, 27, 48, 80, 140, 240, 400], dtype=np.float64)
    return float(np.interp(float(np.clip(kp, 0.0, 9.0)), kp_grid, ap_grid))


def _looks_like_gfz_kpdata_line(line: str) -> bool:
    parts = line.split()
    if len(parts) < 28:
        return False
    try:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        [float(x) for x in parts[7:15]]
        [float(x) for x in parts[15:23]]
        float(parts[23])
        float(parts[25])
        float(parts[26])
    except (TypeError, ValueError):
        return False
    return 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31


def _rolling_f107a(daily_rows, index: int, radius: int = 40) -> float:
    start = max(0, index - radius)
    end = min(len(daily_rows), index + radius + 1)
    values = [row["f107"] for row in daily_rows[start:end] if row.get("f107") is not None]
    if not values:
        return 120.0
    return float(sum(values) / len(values))


def _load_gfz_kpdata_rows(lines):
    daily_rows = []
    for line in lines:
        if not _looks_like_gfz_kpdata_line(line):
            continue
        parts = line.split()
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        date_utc = datetime(year, month, day, tzinfo=timezone.utc)
        daily_rows.append({
            "date": date_utc,
            "kp_values": [float(x) for x in parts[7:15]],
            "ap_values": [float(x) for x in parts[15:23]],
            "daily_ap": float(parts[23]),
            "f107_observed": float(parts[25]),
            "f107": float(parts[26]),
        })

    daily_rows.sort(key=lambda row: row["date"])
    rows = []
    for day_index, daily in enumerate(daily_rows):
        f107a = _rolling_f107a(daily_rows, day_index)
        for block_index in range(8):
            rows.append({
                "utc": daily["date"] + timedelta(hours=3 * block_index),
                "kp": daily["kp_values"][block_index],
                "ap": daily["ap_values"][block_index],
                "daily_ap": daily["daily_ap"],
                "f107": daily["f107"],
                "f107a": f107a,
                "f107_observed": daily["f107_observed"],
            })
    return rows


def _load_space_weather_csv(path: str):
    """
    读取空间天气 CSV。

    推荐列：
      utc_iso/date/time, f107, f107a, ap, kp
    若只有 Kp，脚本会用近似表换算 Ap，便于 NRLMSISE-00 使用。
    """
    if not path:
        return []

    path_obj = Path(path)
    lines = path_obj.read_text(encoding="utf-8-sig").splitlines()
    content_lines = [
        line for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if content_lines and _looks_like_gfz_kpdata_line(content_lines[0]):
        rows = _load_gfz_kpdata_rows(content_lines)
        if not rows:
            raise ValueError("space weather file did not contain valid GFZ kpdata rows")
        return rows

    rows = []
    with path_obj.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("space weather CSV 缺少表头")
        fields = {str(name).strip().lower(): name for name in reader.fieldnames if name}

        time_keys = ["utc_iso", "datetime", "time", "date"]
        f107_keys = ["f107", "f10.7", "f10_7", "daily_f107"]
        f107a_keys = ["f107a", "f10.7a", "f10_7a", "f107_81d", "f107a_81d"]
        ap_keys = ["ap", "ap_index", "planetary_ap"]
        kp_keys = ["kp", "kp_index", "planetary_kp"]

        for raw in reader:
            time_value = None
            for key in time_keys:
                col = fields.get(key)
                if col and raw.get(col):
                    time_value = raw.get(col)
                    break
            if not time_value:
                continue

            item = {"utc": _parse_utc(str(time_value))}
            for out_key, keys in [
                ("f107", f107_keys),
                ("f107a", f107a_keys),
                ("ap", ap_keys),
                ("kp", kp_keys),
            ]:
                for key in keys:
                    col = fields.get(key)
                    if col is None:
                        continue
                    val = _to_float(raw.get(col))
                    if val is not None:
                        item[out_key] = float(val)
                        break

            if "ap" not in item and "kp" in item:
                item["ap"] = _kp_to_ap(item["kp"])
            rows.append(item)

    rows.sort(key=lambda x: x["utc"])
    if not rows:
        raise ValueError("space weather CSV 未解析出有效行")
    return rows


def _space_weather_at(rows, utc_dt: datetime, defaults: dict) -> dict:
    if not rows:
        return dict(defaults)

    # 空间天气通常是小时/日尺度数据，使用不晚于当前时刻的最新记录；早于首行则取首行。
    selected = rows[0]
    for row in rows:
        if row["utc"] <= utc_dt:
            selected = row
        else:
            break

    out = dict(defaults)
    for key in ("f107a", "f107", "ap", "kp"):
        if key in selected:
            out[key] = selected[key]
    if "ap" not in out and "kp" in out:
        out["ap"] = _kp_to_ap(out["kp"])
    return out


def _parse_ground_station_spec(spec: str, default_min_elevation_deg: float) -> dict:
    parts = [part.strip() for part in str(spec).split(",")]
    if len(parts) < 3:
        raise ValueError(
            "ground station spec must be name,lat_deg,lon_deg[,alt_m[,min_elevation_deg]]"
        )

    if _to_float(parts[0]) is None:
        name = parts[0]
        lat_idx = 1
    else:
        name = f"GS-{parts[0]}-{parts[1]}"
        lat_idx = 0

    lat_deg = float(parts[lat_idx])
    lon_deg = float(parts[lat_idx + 1])
    alt_m = float(parts[lat_idx + 2]) if len(parts) > lat_idx + 2 else 0.0
    min_el = (
        float(parts[lat_idx + 3])
        if len(parts) > lat_idx + 3
        else float(default_min_elevation_deg)
    )
    return {
        "name": name,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "alt_m": alt_m,
        "min_elevation_deg": min_el,
    }


def _load_ground_station_csv(path: str, default_min_elevation_deg: float) -> list:
    stations = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("ground station CSV is missing a header")
        for idx, raw in enumerate(reader):
            row = {str(k).strip().lower(): v for k, v in raw.items() if k is not None}
            name = str(_first_present(row, ["name", "station", "id"], f"GS-{idx+1}"))
            lat = _to_float(_first_present(row, ["lat_deg", "latitude_deg", "lat", "latitude"]))
            lon = _to_float(_first_present(row, ["lon_deg", "longitude_deg", "lon", "longitude"]))
            if lat is None or lon is None:
                raise ValueError(f"ground station row {idx + 1} lacks lat/lon")
            stations.append({
                "name": name,
                "lat_deg": float(lat),
                "lon_deg": float(lon),
                "alt_m": float(_to_float(_first_present(
                    row, ["alt_m", "altitude_m", "elevation_m"], 0.0)) or 0.0),
                "min_elevation_deg": float(_to_float(_first_present(
                    row, ["min_elevation_deg", "min_el_deg", "mask_deg"],
                    default_min_elevation_deg)) or default_min_elevation_deg),
                "frequency_hz": _to_float(_first_present(row, ["frequency_hz", "freq_hz"])),
                "bandwidth_hz": _to_float(_first_present(row, ["bandwidth_hz", "bw_hz"])),
                "eirp_dbw": _to_float(_first_present(row, ["eirp_dbw", "satellite_eirp_dbw"])),
                "g_over_t_db_k": _to_float(_first_present(
                    row, ["g_over_t_db_k", "gt_db_k", "g_t_db_k"])),
                "losses_db": _to_float(_first_present(row, ["losses_db", "link_losses_db"])),
            })
    if not stations:
        raise ValueError("ground station CSV did not contain valid stations")
    return stations


def _load_ground_stations(args) -> list:
    if args.ground_station_csv:
        stations = _load_ground_station_csv(
            args.ground_station_csv, args.min_elevation_deg)
    elif args.ground_station:
        stations = [
            _parse_ground_station_spec(spec, args.min_elevation_deg)
            for spec in args.ground_station
        ]
    else:
        stations = [dict(item) for item in DEFAULT_GROUND_STATIONS]

    for station in stations:
        station.setdefault("alt_m", 0.0)
        station.setdefault("min_elevation_deg", float(args.min_elevation_deg))
    return stations


def _station_or_default(station: dict, key: str, default):
    value = station.get(key, None)
    return default if value is None else value


def _link_budget_arrays(slant_range_km,
                        elevation_deg,
                        station: dict,
                        args):
    frequency_hz = float(_station_or_default(
        station, "frequency_hz", args.link_frequency_hz))
    bandwidth_hz = float(_station_or_default(
        station, "bandwidth_hz", args.link_bandwidth_hz))
    eirp_dbw = float(_station_or_default(
        station, "eirp_dbw", args.satellite_eirp_dbw))
    g_over_t_db_k = float(_station_or_default(
        station, "g_over_t_db_k", args.ground_g_over_t_db_k))
    losses_db = float(_station_or_default(
        station, "losses_db", args.link_losses_db))

    slant_m = np.maximum(np.asarray(slant_range_km, dtype=np.float64) * 1000.0, 1.0)
    elevation_deg = np.asarray(elevation_deg, dtype=np.float64)
    fspl_db = 20.0 * np.log10(4.0 * np.pi * slant_m * frequency_hz / 299792458.0)

    low_el_factor = np.clip((30.0 - elevation_deg) / 25.0, 0.0, 1.0)
    elevation_loss_db = float(args.low_elevation_loss_db) * low_el_factor

    cn0_dbhz = (
        eirp_dbw
        + g_over_t_db_k
        - fspl_db
        - losses_db
        - elevation_loss_db
        + 228.6
    )
    snr_db = cn0_dbhz - 10.0 * np.log10(max(bandwidth_hz, 1.0)) - float(args.implementation_loss_db)
    snr_linear = np.maximum(10.0 ** (snr_db / 10.0), 0.0)
    capacity_mbps = (
        bandwidth_hz
        * np.log2(1.0 + snr_linear)
        * float(args.link_efficiency)
        / 1e6
    )
    if args.max_link_capacity_mbps is not None:
        capacity_mbps = np.minimum(capacity_mbps, float(args.max_link_capacity_mbps))
    margin_db = snr_db - float(args.required_snr_db)
    return capacity_mbps, margin_db


def _compute_ground_station_contacts(sat, ts, datetimes, stations: list, args) -> dict:
    if sat is None or wgs84 is None:
        raise ValueError("ground-station link budget requires Skyfield TLE propagation")

    t = ts.from_datetimes(datetimes)
    n_steps = len(datetimes)
    in_window = np.zeros(n_steps, dtype=np.int32)
    capacity = np.zeros(n_steps, dtype=np.float64)
    best_elevation = np.full(n_steps, np.nan, dtype=np.float64)
    best_slant = np.full(n_steps, np.nan, dtype=np.float64)
    best_margin = np.full(n_steps, np.nan, dtype=np.float64)
    best_station = np.full(n_steps, "", dtype=object)

    for station in stations:
        ground = wgs84.latlon(
            float(station["lat_deg"]),
            float(station["lon_deg"]),
            elevation_m=float(station.get("alt_m", 0.0)),
        )
        topocentric = (sat - ground).at(t)
        alt, _az, distance = topocentric.altaz()
        elevation_deg = np.asarray(alt.degrees, dtype=np.float64)
        slant_km = np.asarray(distance.km, dtype=np.float64)
        visible = elevation_deg >= float(station.get("min_elevation_deg", args.min_elevation_deg))
        station_capacity, margin_db = _link_budget_arrays(slant_km, elevation_deg, station, args)
        station_capacity = np.where(visible, station_capacity, 0.0)
        better = station_capacity > capacity
        if np.any(better):
            capacity[better] = station_capacity[better]
            best_elevation[better] = elevation_deg[better]
            best_slant[better] = slant_km[better]
            best_margin[better] = margin_db[better]
            best_station[better] = str(station.get("name", "GS"))

    in_window[capacity > 0.0] = 1
    return {
        "in_window": in_window,
        "capacity_mbps": capacity,
        "best_station": best_station,
        "elevation_deg": best_elevation,
        "slant_range_km": best_slant,
        "link_margin_db": best_margin,
    }


def _summarize_link_contacts(link_contacts: dict, dt_s: int) -> dict:
    in_window = np.asarray(link_contacts["in_window"], dtype=np.int32)
    capacity = np.asarray(link_contacts["capacity_mbps"], dtype=np.float64)
    active_capacity = capacity[capacity > 0.0]
    station_names = np.asarray(link_contacts["best_station"], dtype=object)
    active_names = station_names[station_names != ""]

    station_counts = []
    if active_names.size:
        names, counts = np.unique(active_names, return_counts=True)
        order = np.argsort(counts)[::-1]
        station_counts = [
            (str(names[idx]), int(counts[idx]))
            for idx in order
        ]

    return {
        "contact_steps": int(np.sum(in_window)),
        "contact_fraction": float(np.mean(in_window)) if in_window.size else 0.0,
        "contact_minutes": float(np.sum(in_window) * float(dt_s) / 60.0),
        "mean_capacity_mbps": float(np.mean(capacity)) if capacity.size else 0.0,
        "mean_active_capacity_mbps": (
            float(np.mean(active_capacity)) if active_capacity.size else 0.0
        ),
        "max_capacity_mbps": float(np.max(capacity)) if capacity.size else 0.0,
        "station_counts": station_counts,
    }


def _normalize_tle_name(name: str) -> str:
    text = (name or "").strip()
    if text.startswith("0 "):
        text = text[2:].strip()
    return text


def _tle_epoch_to_datetime(line1: str) -> datetime | None:
    try:
        epoch = line1[18:32].strip()
        year2 = int(epoch[:2])
        day_of_year = float(epoch[2:])
        year = 2000 + year2 if year2 < 57 else 1900 + year2
        return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1.0)
    except Exception:
        return None


def _tle_mean_altitude_km(line2: str) -> float | None:
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
        return 0.5 * (perigee_km + apogee_km)
    except Exception:
        return None


def _tle_altitude_in_range(line2: str,
                           min_altitude_km: float = None,
                           max_altitude_km: float = None) -> bool:
    if min_altitude_km is None and max_altitude_km is None:
        return True
    altitude_km = _tle_mean_altitude_km(line2)
    if altitude_km is None:
        return False
    if min_altitude_km is not None and altitude_km < float(min_altitude_km):
        return False
    if max_altitude_km is not None and altitude_km > float(max_altitude_km):
        return False
    return True


def _iter_tle_triplets(lines):
    """遍历 2LE/3LE 文件中的所有 TLE 记录。"""
    clean = [ln.rstrip("\n") for ln in lines if ln.strip()]
    i = 0
    while i < len(clean) - 1:
        cur = clean[i].strip()
        nxt = clean[i + 1].strip()
        nxt2 = clean[i + 2].strip() if i + 2 < len(clean) else ""
        if cur.startswith("0 ") and nxt.startswith("1 ") and nxt2.startswith("2 "):
            yield _normalize_tle_name(cur), nxt, nxt2
            i += 3
            continue
        if cur.startswith("1 ") and nxt.startswith("2 "):
            satnum = cur[2:7].strip()
            yield f"SAT-{satnum}", cur, nxt
            i += 2
            continue
        i += 1


def _tle_matches(name: str, line1: str, satellite_name: str) -> bool:
    target = _normalize_tle_name(satellite_name).lower()
    if not target:
        return True
    satnum = line1[2:7].strip().lower()
    intl_designator = line1[9:17].strip().lower()
    aliases = {
        _normalize_tle_name(name).lower(),
        satnum,
        f"sat-{satnum}",
        intl_designator,
    }
    return target in aliases


def _extract_tle_triplet(lines, satellite_name: str, target_utc: datetime = None,
                         min_altitude_km: float = None,
                         max_altitude_km: float = None,
                         max_epoch_gap_hours: float = None):
    # 支持两行 TLE、三行 3LE，以及同一颗卫星的历史多 epoch TLE。
    clean = [ln.rstrip("\n") for ln in lines if ln.strip()]
    if len(clean) < 2:
        raise ValueError("TLE content must have at least 2 non-empty lines")

    candidates = [
        (name, line1, line2)
        for name, line1, line2 in _iter_tle_triplets(clean)
        if _tle_matches(name, line1, satellite_name)
        and _tle_altitude_in_range(line2, min_altitude_km, max_altitude_km)
    ]
    if not candidates:
        raise ValueError(
            f"Cannot find TLE for satellite_name={satellite_name!r} "
            f"within altitude range [{min_altitude_km}, {max_altitude_km}] km"
        )

    if target_utc is not None:
        def distance_seconds(item):
            epoch_dt = _tle_epoch_to_datetime(item[1])
            if epoch_dt is None:
                return float("inf")
            return abs((epoch_dt - target_utc).total_seconds())
        candidates.sort(key=distance_seconds)
        if max_epoch_gap_hours is not None:
            gap_seconds = distance_seconds(candidates[0])
            if gap_seconds > float(max_epoch_gap_hours) * 3600.0:
                selected_epoch = _tle_epoch_to_datetime(candidates[0][1])
                raise ValueError(
                    f"Closest TLE epoch {selected_epoch} is "
                    f"{gap_seconds / 3600.0:.2f} hours from start_utc={target_utc}; "
                    f"limit is {max_epoch_gap_hours} hours"
                )
    return candidates[0]

    raise ValueError("Cannot parse TLE lines from provided content")


def _read_xlsx_first_column_lines(path: str):
    """
    从 xlsx 第一张表第一列读取 3LE/TLE 文本。

    这样可以直接使用从网页复制到 Excel 的 3LE 清单，不依赖 openpyxl。
    """
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    lines = []
    with zipfile.ZipFile(path) as z:
        shared_strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared_strings.append("".join(
                    t.text or ""
                    for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                ))

        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        for row in sheet.findall(".//a:sheetData/a:row", ns):
            cells = row.findall("a:c", ns)
            if not cells:
                continue
            cell = cells[0]
            value = cell.find("a:v", ns)
            if value is None:
                continue
            text = value.text or ""
            if cell.get("t") == "s" and text:
                text = shared_strings[int(text)]
            text = str(text).strip()
            if text:
                lines.append(text)
    return lines


def _read_tle_lines_from_file(tle_file: str):
    path = Path(tle_file)
    if path.suffix.lower() == ".xlsx":
        return _read_xlsx_first_column_lines(str(path))
    return path.read_text(encoding="utf-8").splitlines()


def _load_tle_from_file(tle_file: str, satellite_name: str = "",
                        target_utc: datetime = None,
                        min_altitude_km: float = None,
                        max_altitude_km: float = None,
                        max_epoch_gap_hours: float = None):
    lines = _read_tle_lines_from_file(tle_file)
    return _extract_tle_triplet(
        lines, satellite_name=satellite_name, target_utc=target_utc,
        min_altitude_km=min_altitude_km, max_altitude_km=max_altitude_km,
        max_epoch_gap_hours=max_epoch_gap_hours)


def _load_tle_from_url(tle_url: str, satellite_name: str,
                       target_utc: datetime = None,
                       min_altitude_km: float = None,
                       max_altitude_km: float = None,
                       max_epoch_gap_hours: float = None):
    with urllib.request.urlopen(tle_url, timeout=30) as r:
        text = r.read().decode("utf-8", errors="replace")
    return _extract_tle_triplet(
        text.splitlines(), satellite_name=satellite_name, target_utc=target_utc,
        min_altitude_km=min_altitude_km, max_altitude_km=max_altitude_km,
        max_epoch_gap_hours=max_epoch_gap_hours)


def _build_satellite(ts, tle_file: str = None,
                     tle_url: str = None,
                     satellite_name: str = "",
                     target_utc: datetime = None,
                     min_altitude_km: float = None,
                     max_altitude_km: float = None,
                     max_epoch_gap_hours: float = None):
    # 优先使用本地 TLE；只有显式传 URL 时才尝试从网络读取。
    if tle_file:
        name, line1, line2 = _load_tle_from_file(
            tle_file, satellite_name=satellite_name, target_utc=target_utc,
            min_altitude_km=min_altitude_km, max_altitude_km=max_altitude_km,
            max_epoch_gap_hours=max_epoch_gap_hours)
    elif tle_url:
        name, line1, line2 = _load_tle_from_url(
            tle_url, satellite_name=satellite_name, target_utc=target_utc,
            min_altitude_km=min_altitude_km, max_altitude_km=max_altitude_km,
            max_epoch_gap_hours=max_epoch_gap_hours)
    else:
        return None, None
    sat = EarthSatellite(line1, line2, name, ts)
    return sat, name


def _sunlit_fallback(step_idx: int, dt_s: int, orbit_period_s: int) -> int:
    phase = (step_idx * dt_s) % orbit_period_s
    return 1 if phase < 0.62 * orbit_period_s else 0


def _in_window_flag(step_idx: int, dt_s: int,
                    contact_period_s: int,
                    contact_offset_s: int,
                    contact_duration_s: int) -> int:
    t = (step_idx * dt_s - contact_offset_s) % contact_period_s
    return 1 if 0 <= t < contact_duration_s else 0


def _rho_scale_from_altitude(altitude_km: float,
                             ref_altitude_km: float,
                             scale_height_km: float,
                             rho_min_scale: float,
                             rho_max_scale: float) -> float:
    # rho(h) ~ exp(-(h-h0)/H)；转为相对比例后裁剪，避免极端值
    rho_rel = math.exp(-(altitude_km - ref_altitude_km) / max(scale_height_km, 1e-6))
    return float(np.clip(rho_rel, rho_min_scale, rho_max_scale))


def _rho_scale_from_msise(utc_dt: datetime,
                          altitude_km: float,
                          lat_deg: float,
                          lon_deg: float,
                          rho_ref_kg_m3: float,
                          f107a: float,
                          f107: float,
                          ap: float,
                          rho_min_scale: float,
                          rho_max_scale: float):
    if nrlmsise00 is None:
        return None
    try:
        density, _ = nrlmsise00.msise_model(
            time=utc_dt,
            alt=float(altitude_km),
            lat=float(lat_deg),
            lon=float(lon_deg),
            f107a=float(f107a),
            f107=float(f107),
            ap=float(ap),
        )
        # NRLMSISE 输出总质量密度单位为 g/cm^3
        rho_g_cm3 = float(density[5])
        rho_kg_m3 = rho_g_cm3 * 1000.0
        if rho_ref_kg_m3 <= 0:
            return None
        scale = rho_kg_m3 / rho_ref_kg_m3
        return float(np.clip(scale, rho_min_scale, rho_max_scale))
    except Exception:
        return None


def generate_trace(args):
    # trace 文件用于 robustness.py，将真实/近似外生扰动固定下来，保证鲁棒性实验可复现。
    start_utc = _parse_utc(args.start_utc)
    n_steps = max(1, int(args.duration_hours * 3600 / args.dt_s))
    space_weather_rows = _load_space_weather_csv(args.space_weather_csv)
    space_weather_defaults = {
        "f107a": float(args.f107a),
        "f107": float(args.f107),
        "ap": float(args.ap),
        "kp": None,
    }

    datetimes = [start_utc + timedelta(seconds=i * args.dt_s) for i in range(n_steps)]

    altitude_km = np.full(n_steps, args.altitude_km, dtype=np.float64)
    lat_deg = np.zeros(n_steps, dtype=np.float64)
    lon_deg = np.zeros(n_steps, dtype=np.float64)
    sunlit = np.zeros(n_steps, dtype=np.int32)
    satellite_name_used = None
    sat_model = None
    ts_model = None
    link_contacts = None
    ground_stations = []

    used_skyfield = False
    if (args.tle_file or args.tle_url) and EarthSatellite is not None and load is not None and wgs84 is not None:
        try:
            # 有 TLE 且 Skyfield 可用时，用轨道传播得到高度、经纬度和可选日照状态。
            ts = load.timescale()
            sat, satellite_name_used = _build_satellite(
                ts,
                tle_file=args.tle_file,
                tle_url=args.tle_url,
                satellite_name=args.satellite_name,
                target_utc=start_utc,
                min_altitude_km=args.min_tle_altitude_km,
                max_altitude_km=args.max_tle_altitude_km,
                max_epoch_gap_hours=args.max_tle_epoch_gap_hours,
            )
            if sat is None:
                raise ValueError("TLE input is missing")
            sat_model = sat
            ts_model = ts
            t = ts.from_datetimes(datetimes)
            g = sat.at(t)
            subp = wgs84.subpoint(g)
            altitude_km = np.asarray(subp.elevation.km, dtype=np.float64)
            lat_deg = np.asarray(subp.latitude.degrees, dtype=np.float64)
            lon_deg = np.asarray(subp.longitude.degrees, dtype=np.float64)

            # 若提供本地星历则优先使用；否则允许 skyfield 自动下载 de421.bsp
            # 星历失败不应影响真实轨道计算，只回退 sunlit 计算。
            used_ephemeris = False
            try:
                if args.ephemeris and Path(args.ephemeris).exists():
                    eph = load(args.ephemeris)
                    sunlit = np.asarray(g.is_sunlit(eph), dtype=np.int32)
                    used_ephemeris = True
                elif args.auto_download_ephemeris:
                    eph = load("de421.bsp")
                    sunlit = np.asarray(g.is_sunlit(eph), dtype=np.int32)
                    used_ephemeris = True
            except Exception as eph_err:
                print(f"[WARN] ephemeris unavailable, fallback sunlit model: {eph_err}")

            if not used_ephemeris:
                for i in range(n_steps):
                    sunlit[i] = _sunlit_fallback(i, args.dt_s, args.orbit_period_s)
            used_skyfield = True
        except Exception as e:
            if args.require_tle or args.min_tle_altitude_km is not None or args.max_tle_altitude_km is not None:
                raise
            print(f"[WARN] skyfield path failed, fallback to periodic trace: {e}")

    use_link_budget = (
        bool(args.use_link_budget)
        or bool(args.ground_station_csv)
        or bool(args.ground_station)
    )
    if use_link_budget:
        ground_stations = _load_ground_stations(args)
        if not used_skyfield:
            raise ValueError("ground-station link budget requires a valid TLE/Skyfield path")
        link_contacts = _compute_ground_station_contacts(
            sat_model, ts_model, datetimes, ground_stations, args)

    if not used_skyfield:
        # 依赖缺失或 TLE 失败时退化为周期模型，保证脚本不会因为外部数据不可用而中断。
        for i in range(n_steps):
            sunlit[i] = _sunlit_fallback(i, args.dt_s, args.orbit_period_s)

    rows = []
    for i in range(n_steps):
        # 每一行代表一个环境控制步的外部扰动：日照、阻力比例、通信窗口和轨道位置。
        s = int(sunlit[i])
        solar_scale = args.base_solar_scale if s == 1 else args.eclipse_solar_scale
        sw = _space_weather_at(space_weather_rows, datetimes[i], space_weather_defaults)
        f107a = float(sw.get("f107a", args.f107a))
        f107 = float(sw.get("f107", args.f107))
        ap = float(sw.get("ap", args.ap))
        kp = sw.get("kp", None)

        rho_scale = None
        if args.use_nrlmsise:
            # 如果启用 MSISE，则优先用大气模型计算 rho_scale；失败再退回指数近似。
            rho_scale = _rho_scale_from_msise(
                utc_dt=datetimes[i],
                altitude_km=float(altitude_km[i]),
                lat_deg=float(lat_deg[i]),
                lon_deg=float(lon_deg[i]),
                rho_ref_kg_m3=float(args.rho_ref_kg_m3),
                f107a=f107a,
                f107=f107,
                ap=ap,
                rho_min_scale=args.rho_min_scale,
                rho_max_scale=args.rho_max_scale,
            )

        if rho_scale is None:
            rho_scale = _rho_scale_from_altitude(
                altitude_km=float(altitude_km[i]),
                ref_altitude_km=args.ref_altitude_km,
                scale_height_km=args.scale_height_km,
                rho_min_scale=args.rho_min_scale,
                rho_max_scale=args.rho_max_scale,
            )

        best_station = ""
        elevation_deg = ""
        slant_range_km = ""
        link_margin_db = ""
        if link_contacts is not None:
            in_window = int(link_contacts["in_window"][i])
            tx_capacity_mbps = float(link_contacts["capacity_mbps"][i])
            best_station = str(link_contacts["best_station"][i])
            elevation = float(link_contacts["elevation_deg"][i])
            slant = float(link_contacts["slant_range_km"][i])
            margin = float(link_contacts["link_margin_db"][i])
            elevation_deg = "" if np.isnan(elevation) else elevation
            slant_range_km = "" if np.isnan(slant) else slant
            link_margin_db = "" if np.isnan(margin) else margin
        else:
            in_window = _in_window_flag(
                step_idx=i,
                dt_s=args.dt_s,
                contact_period_s=args.contact_period_s,
                contact_offset_s=args.contact_offset_s,
                contact_duration_s=args.contact_duration_s,
            )
            tx_capacity_mbps = float(args.window_capacity_mbps) if in_window else 0.0

        rows.append({
            "utc_iso": datetimes[i].isoformat(),
            "step": i,
            "sunlit": s,
            "solar_scale": float(solar_scale),
            "rho_scale": float(rho_scale),
            "in_window": int(in_window),
            "tx_capacity_mbps": tx_capacity_mbps,
            "altitude_km": float(altitude_km[i]),
            "lat_deg": float(lat_deg[i]),
            "lon_deg": float(lon_deg[i]),
            "best_station": best_station,
            "elevation_deg": elevation_deg,
            "slant_range_km": slant_range_km,
            "link_margin_db": link_margin_db,
            "f107a": f107a,
            "f107": f107,
            "ap": ap,
            "kp": "" if kp is None else float(kp),
        })

    output = Path(args.output)
    # 输出目录不存在时自动创建，方便批处理生成多组 trace。
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utc_iso", "step", "sunlit", "solar_scale", "rho_scale", "in_window",
                        "tx_capacity_mbps", "altitude_km", "lat_deg", "lon_deg",
                        "best_station", "elevation_deg", "slant_range_km", "link_margin_db",
                        "f107a", "f107", "ap", "kp"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] trace saved: {output}")
    print(f"[INFO] rows={len(rows)}, skyfield={'yes' if used_skyfield else 'no'}")
    if satellite_name_used:
        print(f"[INFO] satellite={satellite_name_used}")
    print(f"[INFO] nrlmsise={'yes' if (args.use_nrlmsise and nrlmsise00 is not None) else 'no'}")
    print(f"[INFO] space_weather={'yes' if space_weather_rows else 'no'}")
    if link_contacts is not None:
        contact_summary = _summarize_link_contacts(link_contacts, args.dt_s)
        contact_fraction = contact_summary["contact_fraction"]
        top_stations = ", ".join(
            f"{name}:{count}"
            for name, count in contact_summary["station_counts"][:5]
        ) or "none"
        print(f"[INFO] link_budget=yes, stations={len(ground_stations)}, "
              f"contact_fraction={contact_fraction:.3f}, "
              f"contact_minutes={contact_summary['contact_minutes']:.1f}, "
              f"mean_capacity_mbps={contact_summary['mean_capacity_mbps']:.3f}, "
              f"max_capacity_mbps={contact_summary['max_capacity_mbps']:.3f}")
        print(f"[INFO] top_contact_stations={top_stations}")
        if contact_fraction < float(args.min_contact_fraction_warn):
            print(
                f"[WARN] contact_fraction={contact_fraction:.3f} is sparse; "
                "use a larger/more global --ground_station_csv or longer --duration_hours "
                "for primary real-trace pressure tests."
            )
    else:
        print("[INFO] link_budget=no")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trace-driven perturbation CSV")
    parser.add_argument("--output", required=True, help="output CSV path")
    parser.add_argument("--start_utc", default="2026-01-01T00:00:00+00:00")
    parser.add_argument("--duration_hours", type=float, default=6.0)
    parser.add_argument("--dt_s", type=int, default=10)

    # Skyfield inputs (optional)
    parser.add_argument("--tle_file", default=None, help="optional TLE file path (2 or 3 lines)")
    parser.add_argument("--tle_url", default=None,
                        help="optional TLE source URL")
    parser.add_argument("--satellite_name", default="",
                        help="optional satellite name, catalog number, or international designator")
    parser.add_argument("--min_tle_altitude_km", type=float, default=None,
                        help="minimum mean TLE altitude accepted when selecting a TLE")
    parser.add_argument("--max_tle_altitude_km", type=float, default=None,
                        help="maximum mean TLE altitude accepted when selecting a TLE")
    parser.add_argument("--max_tle_epoch_gap_hours", type=float, default=None,
                        help="maximum allowed gap between start_utc and selected TLE epoch")
    parser.add_argument("--require_tle", action="store_true",
                        help="fail instead of falling back when TLE propagation is unavailable")
    parser.add_argument("--ephemeris", default=None, help="optional local bsp file (e.g. de421.bsp)")
    parser.add_argument("--auto_download_ephemeris", action="store_true",
                        help="allow skyfield to auto-download de421.bsp when local ephemeris is missing")

    # Period/contact fallback controls
    parser.add_argument("--orbit_period_s", type=int, default=5400)
    parser.add_argument("--contact_period_s", type=int, default=5400)
    parser.add_argument("--contact_offset_s", type=int, default=1800)
    parser.add_argument("--contact_duration_s", type=int, default=600)
    parser.add_argument("--window_capacity_mbps", type=float, default=100.0,
                        help="fallback/contact trace window capacity when in_window=1")
    parser.add_argument("--use_link_budget", action="store_true",
                        help="compute in_window and tx_capacity_mbps from ground-station geometry")
    parser.add_argument("--ground_station_csv", default=None,
                        help="CSV with name,lat_deg,lon_deg,alt_m,min_elevation_deg and optional link columns")
    parser.add_argument("--ground_station", action="append", default=[],
                        help="ground station spec: name,lat_deg,lon_deg[,alt_m[,min_elevation_deg]]")
    parser.add_argument("--min_elevation_deg", type=float, default=5.0,
                        help="default ground-station elevation mask")
    parser.add_argument("--link_frequency_hz", type=float, default=2.2e9,
                        help="downlink carrier frequency used by the link budget")
    parser.add_argument("--link_bandwidth_hz", type=float, default=20e6,
                        help="downlink bandwidth used by the link budget")
    parser.add_argument("--satellite_eirp_dbw", type=float, default=12.0,
                        help="satellite downlink EIRP in dBW")
    parser.add_argument("--ground_g_over_t_db_k", type=float, default=16.0,
                        help="ground-station receive G/T in dB/K")
    parser.add_argument("--link_losses_db", type=float, default=3.0,
                        help="fixed implementation, pointing, polarization, and feeder losses")
    parser.add_argument("--implementation_loss_db", type=float, default=2.0,
                        help="SNR margin lost to coding/modem implementation")
    parser.add_argument("--low_elevation_loss_db", type=float, default=2.0,
                        help="extra attenuation applied near the elevation mask")
    parser.add_argument("--required_snr_db", type=float, default=3.0,
                        help="minimum SNR used only for reporting link_margin_db")
    parser.add_argument("--link_efficiency", type=float, default=0.75,
                        help="fraction of Shannon capacity available after waveform overheads")
    parser.add_argument("--max_link_capacity_mbps", type=float, default=1000.0,
                        help="optional cap on computed link capacity")
    parser.add_argument("--min_contact_fraction_warn", type=float, default=0.03,
                        help="warn when link-budget contact fraction is below this threshold")

    # Scaling controls
    parser.add_argument("--base_solar_scale", type=float, default=1.0)
    parser.add_argument("--eclipse_solar_scale", type=float, default=0.15)
    parser.add_argument("--altitude_km", type=float, default=350.0)
    parser.add_argument("--ref_altitude_km", type=float, default=350.0)
    parser.add_argument("--scale_height_km", type=float, default=45.0)
    parser.add_argument("--rho_min_scale", type=float, default=0.2)
    parser.add_argument("--rho_max_scale", type=float, default=5.0)
    parser.add_argument("--use_nrlmsise", action="store_true",
                        help="use NRLMSISE-00 density model to compute rho_scale")
    parser.add_argument("--space_weather_csv", default=None,
                        help="optional CSV with utc_iso/date, f107, f107a, ap or kp columns")
    parser.add_argument("--rho_ref_kg_m3", type=float, default=4.89e-11,
                        help="reference density for converting MSISE density to rho_scale")
    parser.add_argument("--f107a", type=float, default=120.0,
                        help="81-day average F10.7 index for NRLMSISE")
    parser.add_argument("--f107", type=float, default=120.0,
                        help="daily F10.7 index for NRLMSISE")
    parser.add_argument("--ap", type=float, default=10.0,
                        help="planetary Ap index for NRLMSISE")

    generate_trace(parser.parse_args())
