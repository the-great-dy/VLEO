"""用于训练和实验的统一指标日志记录。

日志记录器将相同的清理指标流写入JSONL、CSV模式和可选的
TensorBoard / WandB接收器。控制台输出故意与指标日志记录分离，
以便从一个清洁的模式生成论文表格。
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np


LOGGER = logging.getLogger("vleo.metrics")


def configure_console_logging(level: int = logging.INFO) -> logging.Logger:
    """配置项目范围的控制台日志记录器一次并返回它。"""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(message)s")
    LOGGER.setLevel(level)
    return LOGGER


class MetricJSONEncoder(json.JSONEncoder):
    """用于numpy / torch标量值的JSON编码器。"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                tensor = obj.detach().cpu()
                return tensor.item() if tensor.numel() == 1 else tensor.tolist()
        except ImportError:
            pass
        return super().default(obj)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        import torch
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    except ImportError:
        pass
    return value


def _flatten_metrics(metrics: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in (metrics or {}).items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, name))
        else:
            flat[name] = _to_serializable(value)
    return flat


def _is_scalar_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating, np.bool_))


class MetricLogger:
    """训练、消融和评估脚本使用的单个指标接收器。"""

    def __init__(
        self,
        log_dir: str,
        *,
        run_name: str = "train",
        enable_tensorboard: bool = False,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        wandb_config: dict | None = None,
    ):
        self.log_dir = log_dir
        self.run_name = run_name
        os.makedirs(log_dir, exist_ok=True)

        self.start_time = datetime.now()
        stamp = f"{self.start_time:%Y%m%d_%H%M%S}"
        self.jsonl_path = os.path.join(log_dir, f"{run_name}_{stamp}.jsonl")
        self.csv_path = os.path.join(log_dir, f"{run_name}_{stamp}.csv")
        self.schema_path = os.path.join(log_dir, f"{run_name}_metric_schema.json")
        self.summary_path = os.path.join(log_dir, "summary.json")

        self.data = defaultdict(list)
        self.records: list[dict] = []
        self.schema: list[str] = []
        self.schema_set: set[str] = set()
        self.console = configure_console_logging()

        self.tb_writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=log_dir)
            except Exception as exc:
                self.console.warning(f"[MetricLogger] TensorBoard disabled: {exc}")

        self.wandb = None
        self.wandb_run = None
        if wandb_project:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config=wandb_config or {},
                    reinit=True,
                )
            except Exception as exc:
                self.console.warning(f"[MetricLogger] WandB disabled: {exc}")

    def log_step(self, step: int, metrics: dict, *, namespace: str | None = None):
        flat = _flatten_metrics(metrics)
        if namespace:
            flat = {f"{namespace}/{key}": value for key, value in flat.items()}

        safe = {key: _to_serializable(value) for key, value in flat.items()}
        record = {"step": int(step), **safe}
        self.records.append(record)

        for key in record:
            if key not in self.schema_set:
                self.schema_set.add(key)
                self.schema.append(key)

        for key, value in safe.items():
            self.data[key].append(value)

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, cls=MetricJSONEncoder, ensure_ascii=False) + "\n")

        self._write_optional_sinks(int(step), safe)

    def log_eval(self, step: int, metrics: dict):
        self.log_step(step, metrics, namespace="eval")

    def _write_optional_sinks(self, step: int, metrics: dict) -> None:
        numeric = {
            key: float(value)
            for key, value in metrics.items()
            if _is_scalar_number(value)
        }
        if self.tb_writer is not None:
            for key, value in numeric.items():
                self.tb_writer.add_scalar(key, value, step)
        if self.wandb is not None:
            self.wandb.log(numeric, step=step)

    def save(self):
        summary = {}
        for key, values in self.data.items():
            numeric_values = []
            for value in values[-100:]:
                if value is None or not _is_scalar_number(value):
                    continue
                numeric_values.append(float(value))
            if numeric_values:
                summary[key] = float(np.mean(numeric_values))
                summary[f"{key}_std"] = float(np.std(numeric_values))

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(self.schema_path, "w", encoding="utf-8") as f:
            json.dump({"columns": self.schema}, f, indent=2, ensure_ascii=False)

        if self.records:
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.schema)
                writer.writeheader()
                for record in self.records:
                    writer.writerow({key: record.get(key, "") for key in self.schema})

        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()

        self.console.info(f"[MetricLogger] saved metrics: {self.log_dir}")


TrainingLogger = MetricLogger
