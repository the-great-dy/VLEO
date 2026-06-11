# VLEO 安全/性能自动调参实验日志（Agent 接力）

> 设备：RTX 4070 Ti SUPER（CUDA）。目标：`episode_safety_rate≥0.90`、`violation_rate≤0.10`、`delivered≥95% baseline`。

---

## SAFE_BUDGET_FALLBACK 轮（2026-06-07）

### 背景：window_util 与 episode_safety 的跷跷板
冻结基线 `final_safe_baseline.pt` survival=1.0 但 window_util≈0.098（对日充电保安全，少下传）。
重训得到 `best_optimized.pt`（step 400k，用旧激进 mission_pointing_fallback 脚手架训练）。

20-seed×5ep 终评发现**训练/评估开关不一致**：
- fallback OFF（eval 默认）：ep_safe 0.97 但 window 0.086（拐杖被抽走）
- fallback ON（=训练口径）：window 0.324 但 ep_safe 崩到 0.76、worst_seed 0.40、出现 crash

### 诊断（experiments/diagnose_fallback_safety.py，min_soc=0.55）
- fallback_trigger 0.597；**to_image 0.317（最大头）**、to_downlink 0.110、to_charge(safety_guard_sun) 0.170。
- 崩前回溯：SOC 被钉在 0.55 门控线反复 SUN↔IMAGE churn（强制成像放电→回充→再成像），
  建不起能量缓冲，一遇扰动即破线；疯狂成像同时把 proc/dl 推到 4.16。
- **根因**：旧 `daylit_image` 规则只看当前 SOC，不估动作后 SOC、不看 eclipse 充电机会、不看 backlog。

### 修复：SAFE_BUDGET_FALLBACK（config.SAFE_BUDGET_FALLBACK_CONFIG + satellite_env._apply_safe_budget_fallback）
安全优先级硬序：`hard_safety > charge > downlink-in-contact > image/process > idle`。
- `_project_soc_through_eclipse(mode)`：估计候选指向下穿越下一次阴影后的 SOC，< reserve 则禁该动作改充电（前瞻能量预算，旧 fallback 完全没有）。
- 充电机会前瞻：未来 lookahead 内日照不足（长阴影）→ 抬高 reserve（eclipse_reserve_bonus）。
- data_pressure = onboard_mb / future_downlink_capacity_mb：>1.5 不主动成像，>2.0 禁成像+禁处理（alpha_cpu→0），直击 proc/dl。
- 窗口内 SOC 裕度不足只允许低功率下传（cap alpha_tx）。
- 回填 `mission_pointing_fallback_applied`，使 eval 聚合与训练 AP-BC（mission_pointing_bc_weight）以安全壳动作为模仿目标 → 策略共适应。

### 小规模扫描（experiments/scan_safe_budget.py，best_optimized eval-time，3seed×2ep）
| config | ep_safe | worst | window | downlink | proc_dl |
|---|---|---|---|---|---|
| fallback_off | 1.00 | 1.00 | 0.081 | 1953 | 4.08 |
| aggressive_on | 0.83 | 0.50 | 0.384 | 6312 | 3.78 |
| **safe_budget(0.60)** | **1.00** | **1.00** | **0.463** | 7482 | 3.12 |
| safe_budget(0.65) | 1.00 | 1.00 | 0.424 | 7281 | 3.10 |
| safe_budget(0.70) | 1.00 | 1.00 | 0.346 | 6202 | 3.64 |
→ soft=0.60 在安全和吞吐上同时碾压旧激进档。

### 20-seed×5ep 全量确认（results/multiseed_safebudget_soft060_20260607.json，eval-time）
ep_safe **0.96**(worst 0.80) / survival 1.0 / crash 0 / step_safe 1.000 /
window **0.389**(worst 0.258) / downlink 6816 / delivered 7587 / hi_del 0.456 / proc_dl 2.79 / tx_in_contact 0.636。
- 对比 fallback OFF（0.97/0.086）与 ON（0.76/0.324）：**安全守住 + window 4.5× + downlink +230%**。
- 6 条成功标准除 proc/dl≤2.0（=2.79，但明显低于激进档 4.16）外全部达标。

### 决策与改动
1. **固化交付**：`SAFE_BUDGET_FALLBACK_CONFIG.enabled=True, soft_min_soc=0.60` 定为永久部署安全壳。
   交付方案 = `best_optimized.pt` + SAFE_BUDGET 壳（eval-time，无需重训即达标）。
2. **checkpoint 选择安全门**：`DRL_CONFIG.checkpoint_min_episode_safety_rate=0.90`，
   `train._selection_tuple` 把 ep_safe≥0.90 加入可行集硬门（杜绝高吞吐但不安全的模型被选成 best）。
3. **后台重训对比**：from-scratch 540k，SAFE_BUDGET 全程开（策略共适应），输出到
   `checkpoints_safebudget/`（不覆盖交付基线），目标把 proc/dl 压到≤2.0、抬高 worst-seed ep_safe/hi_del。

### Gate 缺陷修复（2026-06-08）
重训后发现 checkpoint gate 用**单 seed periodic eval** 选 best：把 step200k 躺平模型
（canonical window=0.036/downlink=713/delivered=927，但 ep_safe=0.99）选成 best_optimized.pt。
periodic eval 已反复被证明不可靠（best_optimized periodic ep_safe=0.1 / latest periodic 60-70%，
canonical 分别是 0.96 / 1.00）。修复：
- `config.CHECKPOINT_SELECTION_CONFIG`：canonical 口径的 safety floor + utility floor
  (window≥0.20/downlink≥3500/delivered≥4000/proc_dl≤3.0) + anti-conservative filter
  (window<0.10→collapse) + safety-constrained utility score + 锁定基线 + 替换条件。
- `experiments/select_best_checkpoint.py`：**权威 post-hoc 选择器，只认 canonical 20-seed eval**；
  无合格者保留交付基线，绝不退化选保守躺平模型。
- `train.py._selection_tuple`：加 anti-conservative utility floor 防躺平模型混入候选；
  注释明确 periodic 仅供候选保存，权威选择用 canonical 工具。

### 重训 canonical 20-seed 裁决（results/checkpoint_selection_final_20260608.json）
| 候选 | ep_safe(worst) | window | downlink | delivered | proc_dl | floors | score |
|---|---|---|---|---|---|---|---|
| **delivery_baseline** | 0.96(0.80) | **0.389** | **6816** | **7587** | 2.79 | PASS/PASS | **1.68 ✅best** |
| retrain_best(200k) | 0.99(0.80) | 0.036 | 713 | 927 | 2.58 | PASS/FAIL collapse | — |
| retrain_latest(540k) | 1.00(1.00) | 0.203 | 3817 | 4246 | 2.66 | PASS/PASS | -0.20 |

**结论：保留交付基线，重训不替换。** retrain_latest 安全更好(ep_safe 1.0)但吞吐砍半
(downlink/delivered -44%)、proc_dl 几乎没动(2.79→2.66，远未到≤2.0)。
**SAFE_BUDGET 全程共适应把策略训得更保守，未达成"保吞吐+降 proc_dl"目标。**
periodic eval 在 400k 显示的 proc_dl=2.13 是 seed42 假象，canonical 未兑现。

### 最终交付（锁定）
`checkpoints_optimized/best_optimized.pt` + `SAFE_BUDGET_FALLBACK_CONFIG(enabled=True, soft=0.60)`
eval-time 安全壳：ep_safe 0.96 / survival 1.0 / crash 0 / window 0.389 / downlink 6816 /
delivered 7587 / hi_del 0.456 / proc_dl 2.79。
proc_dl≤2.0 仍未达成（唯一软肋）；若后续追求，需换思路（如只在 eval 用壳、或用 reward shaping
定向降 proc_dl，而非让策略全程共适应整个壳），不再盲训。

### proc/dl 定向 ablation（2026-06-08，config/gate 级，不重训）
experiments/ablation_proc_dl.py，best_optimized eval-time，6 变体 3seed×2ep 快筛：
| 变体 | ep_safe | window | downlink | delivered | proc_dl |
|---|---|---|---|---|---|
| current | 1.0 | 0.463 | 7482 | 8701 | 3.12 |
| dp_throttle@1.75 | 1.0 | 0.460 | 7479 | 8846 | 3.14 |
| dp_throttle@1.5 | 1.0 | 0.426 | 6958 | 10208 | 3.28 |
| dp_throttle@1.25 | 1.0 | 0.447 | 7264 | 7975 | **2.92** |
| strong_proc_penalty | 1.0 | 0.428 | 7046 | 9124 | 3.10 |
| future_contact_gate | 1.0 | 0.463 | 7482 | 8701 | 3.12（与 current 逐字节相同） |

**结论：eval-time 门控无法把 proc/dl 明显降下来。**
- data_pressure 节流是反应式（backlog 起来才压），但策略在 dp 还低时就过度处理 → 太晚，最多 -6%。
- future_contact_gate **完全 no-op**：它按瞬时队列水位 ratio≥0.55 触发，而 processed 快速过期/下传
  使队列水位一直低 → 门永不触发。它防的是"队列溢出"，不是"处理速率 > 下传速率"。
- 根因：proc/dl≈3 是**学到的处理速率 ≫ 下传速率**（且发生在 backlog 未堆积时），现有 eval-time
  机制都够不到。所有变体安全/吞吐全稳（ep_safe 1.0/window>0.42/downlink>6900）。
新增 lever（no-op 默认，不改锁定基线）：SAFE_BUDGET_FALLBACK_CONFIG.cpu_throttle_pressure +
process_cap_alpha_soft（分级 soft CPU cap），env._apply_safe_budget_fallback 分级门控。
**proc/dl≤2.0 只能靠 (a) 新增 running-ratio 处理门（cap 每步 processed≤T×累计 downlinked，
直接把 episode proc/dl 钉到 ~T，仍是 eval-time 无需重训）或 (b) 定向 reward shaping 重训。**

### credit-bucket 处理门（2026-06-08，eval-time，治 proc/dl 根因）
naive `processed≤T×downlinked` 会在前期无窗口时锁死处理。改 **leaky-bucket 流控**：
credit = initial + gain·累计下传 − 累计处理；credit≤0→禁处理/禁成像（窗口内改下传，否则充电），
credit≤soft→节流 alpha_cpu；渐近把 episode proc/dl 钉到 ~target，initial 缓冲防前期锁死。
实现：config.SAFE_BUDGET_FALLBACK_CONFIG（enable_credit_gate 等，默认 no-op）+
satellite_env._apply_safe_budget_fallback（gate）+ step 末 credit 更新 + info 诊断。
harness：experiments/ablation_credit_gate.py。

快筛（3seed×2ep）proc/dl 全部从 3.12 降到 2.05~2.37、expired 25291→9000~19000，安全全稳。
**canonical 20-seed×5ep 确认（t=2.5 bigInit，同口径对比，results/multiseed_creditgate_t25bigInit_20260608.json）**：
| 指标 | 旧基线(SAFE_BUDGET) | 新(+credit gate) |
|---|---|---|
| episode_safety | 0.96(worst 0.80) | **1.00(worst 1.00)** |
| comm_window | 0.389 | 0.372 |
| downlink | 6816 | 6422(−6%) |
| delivered | 7587 | 7120(−6%) |
| **proc/dl** | 2.79 | **2.16(−23%)** |
| intervention | 0.239 | 0.216 |
| expired_value_rate | 0.248 | 0.245（率不变，绝对降因总处理少） |
| high_value_delivery | 0.456 | 0.456 |

**用户裁决：替换。** proc/dl 达 ≤2.5 目标 + 完美安全(ep_safe 1.0)，代价 −6% 吞吐仍过所有 floor。

### 最终交付（更新锁定）
`checkpoints_optimized/best_optimized.pt` +
`SAFE_BUDGET_FALLBACK_CONFIG(enabled=True, soft=0.60, enable_credit_gate=True, target=2.5, initial_credit_factor=2.5)`。
canonical 20-seed：ep_safe **1.00** / survival 1.0 / crash 0 / window 0.372 / downlink 6422 /
delivered 7120 / proc/dl **2.16** / hi_del 0.456。
注：expired_value_rate(~0.245) 与 high_value_delivery(0.456) 是结构性指标，credit gate 未改变；
proc/dl≤2.0 若要继续追，需定向 reward-shaping 重训（本阶段按"不盲训"约束未做）。
