"""
Generate the paper-ready GOCE/SLATS real trace CSVs with a global ground network.

This wrapper keeps the exact input files and start dates explicit, so the results
folder can be regenerated without relying on shell history.
"""

import argparse
import subprocess
import sys
from pathlib import Path


TRACE_SPECS = {
    "goce": {
        "label": "GOCE",
        "start_utc": "2012-12-31T00:00:00+00:00",
        "tle_file": "real_data/GOCE_tle.txt",
        "space_weather_csv": "real_data/GOCE_kpdata.txt",
        "output_name": "goce_real_trace_link.csv",
    },
    "slats": {
        "label": "SLATS",
        "start_utc": "2019-07-31T00:00:00+00:00",
        "tle_file": "real_data/SLATS_tle.txt",
        "space_weather_csv": "real_data/SLATS_kpdata.txt",
        "output_name": "slats_real_trace_link.csv",
    },
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _selected_specs(selection: str) -> list[tuple[str, dict]]:
    if selection == "all":
        return list(TRACE_SPECS.items())
    return [(selection, TRACE_SPECS[selection])]


def _build_command(args, spec: dict) -> list[str]:
    root = _project_root()
    output = root / args.output_dir / spec["output_name"]
    script = root / "experiments" / "generate_trace_csv.py"

    command = [
        sys.executable,
        str(script),
        "--output",
        str(output),
        "--start_utc",
        spec["start_utc"],
        "--duration_hours",
        str(args.duration_hours),
        "--dt_s",
        str(args.dt_s),
        "--tle_file",
        str(root / spec["tle_file"]),
        "--space_weather_csv",
        str(root / spec["space_weather_csv"]),
        "--ground_station_csv",
        str(root / args.ground_station_csv),
        "--use_link_budget",
        "--require_tle",
        "--min_elevation_deg",
        str(args.min_elevation_deg),
        "--link_bandwidth_hz",
        str(args.link_bandwidth_hz),
        "--max_link_capacity_mbps",
        str(args.max_link_capacity_mbps),
        "--min_contact_fraction_warn",
        str(args.min_contact_fraction_warn),
    ]
    if not args.no_nrlmsise:
        command.append("--use_nrlmsise")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate GOCE/SLATS real traces with a global ground-station network"
    )
    parser.add_argument("--only", choices=["all", "goce", "slats"], default="all")
    parser.add_argument("--duration_hours", type=float, default=72.0,
                        help="real trace length; 72h avoids one-day phase bias")
    parser.add_argument("--dt_s", type=int, default=10)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--ground_station_csv", default="data/ground_stations_global.csv")
    parser.add_argument("--min_elevation_deg", type=float, default=5.0)
    parser.add_argument("--link_bandwidth_hz", type=float, default=20e6)
    parser.add_argument("--max_link_capacity_mbps", type=float, default=1000.0)
    parser.add_argument("--min_contact_fraction_warn", type=float, default=0.03)
    parser.add_argument("--no_nrlmsise", action="store_true",
                        help="fall back to the exponential density approximation")
    args = parser.parse_args()

    root = _project_root()
    station_csv = root / args.ground_station_csv
    if not station_csv.exists():
        raise FileNotFoundError(f"ground station CSV not found: {station_csv}")

    for key, spec in _selected_specs(args.only):
        print(f"[SUITE] generating {spec['label']} trace ({key})", flush=True)
        subprocess.run(_build_command(args, spec), check=True)


if __name__ == "__main__":
    main()
