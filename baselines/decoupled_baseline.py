"""
解耦堆叠基线 (DECOUPLED) —— "联合优化 > 模块堆叠" 假设 H1 的对照靶子。

设计语义 (见 docs/joint_optimization_experiment_design.md §3.1):
  两段逻辑互不知道对方,刻意制造"无协调的模块堆叠":
    1) 解析维轨控制器:只看高度/阻力,按高度误差闭环算 α_prop,目标维持标称高度带。
       对任务队列/价值/通信窗口 **完全无感**。维轨优先占电。
    2) 任务调度内核:用一个现成调度器 (默认 ValueAwareHeuristicBaseline) 决定
       α_cpu/α_tx 及价值/紧迫/丢弃维度。对维轨占了多少电 **完全无感**。

  与 Ours 的区别:Ours 用单一联合策略在共享功率预算上动态权衡 prop vs 任务;
  本基线两模块各自贪心,冲突交给环境的功率裁剪层被动仲裁 —— 高阻力下要么
  维轨过度占电饿死任务、要么任务抢电导致维轨不足跌入 unsafe (假设 H2)。

接口对齐其它基线 (baselines/heuristic_baseline.py):
  实例提供 .schedule(state, env) -> 8/9 维分组动作;
  在 experiments/compare_all.py 中按
      d = DecoupledOrbitSchedulerBaseline()
      def decoupled_fn(state, env): return d.schedule(state, env)
      results["DECOUPLED-Heur"] = evaluate_on_env(_pointed(decoupled_fn), ...)
  注册即可 (内核换 MPCBaseline 即得 DECOUPLED-MPC 变体)。

⚠️ 骨架状态 (v1):α_prop→功率 的精确换算在 env / safety.power_manager 里,
   这里用"按高度分档的 prop 强度 + 对 cpu/tx 做剩余预算缩放"的**近似**实现
   coupling-blind 堆叠,足以跑 §10.2 sanity check。标 [TODO-EXACT-POWER] 处
   接入 env 真实功率模型后即为论文最终口径。
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, "") and _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import numpy as np
from config import ORBITAL_CONFIG, ENERGY_CONFIG
from environment.satellite_env import OBSERVATION_FEATURES
from utils.action_space import default_grouped_action
from baselines.heuristic_baseline import ValueAwareHeuristicBaseline

_FEATURE_INDEX = {name: idx for idx, name in enumerate(OBSERVATION_FEATURES)}


def _feature(state: np.ndarray, name: str, default: float = 0.0) -> float:
    idx = _FEATURE_INDEX.get(name)
    if idx is None or idx >= state.shape[0]:
        return float(default)
    return float(state[idx])


class AnalyticOrbitKeeper:
    """高度-误差闭环维轨控制器 (载荷无感)。

    控制律 (分档 P 控制,贴合 ORBITAL_CONFIG 安全带):
      h >= nominal           : 仅最小站位保持 (抵消标称阻力)
      warning <= h < nominal : 线性 ramp,误差越大推得越多
      min     <= h < warning : 高推力
      h < min  (unsafe)      : 满推力保命
    输出 α_prop ∈ [0, 1]。完全不看任务/队列/窗口/电量收益。
    """

    def __init__(self,
                 station_keeping_floor: float = 0.18,
                 nominal_km: float | None = None,
                 warning_km: float | None = None,
                 min_km: float | None = None):
        self.h_nominal = float(nominal_km if nominal_km is not None
                               else ORBITAL_CONFIG["altitude_nominal_km"])
        self.h_warning = float(warning_km if warning_km is not None
                               else ORBITAL_CONFIG.get("altitude_warning_km", 200.0))
        self.h_min = float(min_km if min_km is not None
                           else ORBITAL_CONFIG["altitude_min_km"])
        self.h_max = float(ORBITAL_CONFIG["altitude_max_km"])
        # 标称高度抵消阻力所需的基础推力占比 (站位保持)。
        self.station_keeping_floor = float(station_keeping_floor)

    def altitude_km(self, state: np.ndarray, env=None) -> float:
        if env is not None and hasattr(env, "altitude_m"):
            return float(env.altitude_m) / 1e3
        h_norm = _feature(state, "altitude_norm", 0.5)
        return self.h_min + h_norm * (self.h_max - self.h_min)

    def alpha_prop(self, state: np.ndarray, env=None) -> float:
        h = self.altitude_km(state, env)
        if h >= self.h_nominal:
            return self.station_keeping_floor
        if h >= self.h_warning:
            # nominal..warning 之间线性 ramp: floor → 0.6
            frac = (self.h_nominal - h) / max(self.h_nominal - self.h_warning, 1e-6)
            return float(np.clip(self.station_keeping_floor + frac * (0.6 - self.station_keeping_floor), 0.0, 1.0))
        if h >= self.h_min:
            # warning..min 之间 ramp: 0.6 → 0.95
            frac = (self.h_warning - h) / max(self.h_warning - self.h_min, 1e-6)
            return float(np.clip(0.6 + frac * 0.35, 0.0, 1.0))
        # unsafe: 满推力
        return 1.0


class DecoupledOrbitSchedulerBaseline:
    """解耦堆叠基线:解析维轨 (载荷无感) + 独立任务调度内核 (维轨无感)。

    scheduler_kernel: 任何提供 .schedule(state)->分组动作 的实例;
                      默认 ValueAwareHeuristicBaseline。换 MPCBaseline 即得 DECOUPLED-MPC。
    """

    def __init__(self, scheduler_kernel=None, orbit_keeper: AnalyticOrbitKeeper | None = None):
        self.kernel = scheduler_kernel if scheduler_kernel is not None else ValueAwareHeuristicBaseline()
        self.orbit = orbit_keeper if orbit_keeper is not None else AnalyticOrbitKeeper()
        self.last_diagnostics: dict = {}

    def schedule(self, state: np.ndarray, env=None) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)

        # ── 模块 A:解析维轨 (对任务负载无感) ───────────────────────
        alpha_prop = self.orbit.alpha_prop(state, env)

        # ── 模块 B:独立任务调度内核 (对维轨占电无感) ────────────────
        # 内核按"自己拥有全部预算"决定 cpu/tx 及价值/紧迫/丢弃维度。
        try:
            kernel_action = np.asarray(
                self.kernel.schedule(state, env), dtype=np.float32
            ).reshape(-1)
        except TypeError:
            kernel_action = np.asarray(
                self.kernel.schedule(state), dtype=np.float32
            ).reshape(-1)
        if kernel_action.size < 9:
            kernel_action = np.pad(kernel_action, (0, 9 - kernel_action.size),
                                   mode="constant", constant_values=0.5)
        action = kernel_action.copy()

        # ── 堆叠仲裁:维轨优先占电,任务只能用"剩余预算" ──────────────
        # [TODO-EXACT-POWER] 论文最终口径应调用 env / safety.power_manager 把
        # α_prop 换算成 W、从 available_power_w 扣除后,按真实剩余预算缩放 cpu/tx。
        # 骨架近似:用 (1 - alpha_prop) 作为剩余预算比例,线性压低 cpu/tx。
        # 这刻意保留 coupling-blind 失配:内核不知道自己被压、维轨不知道压了多少。
        remaining = float(np.clip(1.0 - alpha_prop, 0.0, 1.0))
        action[0] = float(np.clip(alpha_prop, 0.0, 1.0))   # prop 由维轨控制器接管
        action[1] = float(np.clip(kernel_action[1] * remaining, 0.0, 1.0))  # cpu
        action[2] = float(np.clip(kernel_action[2] * remaining, 0.0, 1.0))  # tx
        # 价值/紧迫/丢弃维度 (3..7) 沿用内核;指向 (8) 留给 _pointed 注入。

        self.last_diagnostics = {
            "decoupled_alpha_prop": action[0],
            "decoupled_remaining_budget": remaining,
            "decoupled_altitude_km": self.orbit.altitude_km(state, env),
        }
        return action.astype(np.float32)


def make_decoupled_baseline(kernel: str = "heuristic"):
    """工厂:'heuristic' → DECOUPLED-Heur;'mpc' → DECOUPLED-MPC。

    返回 (name, scheduler_fn(state, env))。scheduler_fn 仍需在 compare_all 里
    用 _pointed(...) 包裹后传给 evaluate_on_env。
    """
    if kernel == "mpc":
        from baselines.mpc_baseline import MPCBaseline

        class _MPCKernelAdapter:
            def __init__(self):
                self._mpc = MPCBaseline()

            def schedule(self, state, env=None):
                # MPCBaseline.schedule 需要 (raw_state, env);此适配器只用于
                # 拿 cpu/tx 倾向,env=None 时退化为保守均衡 (骨架,待接 env)。
                try:
                    if env is not None:
                        return self._mpc.schedule(
                            state,
                            env.battery.soc,
                            env.altitude_m,
                            env.orbit_sim.is_sunlit(env.time_s),
                            env.solar.output_power(
                                env.orbit_sim.sunlit_fraction(env.time_s)
                            ),
                            time_s=env.time_s,
                            env=env,
                        )
                    return self._mpc.schedule(state, None)
                except Exception:
                    return default_grouped_action([0.3, 0.4, 0.4])

        d = DecoupledOrbitSchedulerBaseline(scheduler_kernel=_MPCKernelAdapter())
        return "DECOUPLED-MPC", (lambda state, env: d.schedule(state, env))

    d = DecoupledOrbitSchedulerBaseline()
    return "DECOUPLED-Heur", (lambda state, env: d.schedule(state, env))
