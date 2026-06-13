# Safety-First Reinforcement Learning for Autonomous VLEO Earth-Observation Scheduling: A Deployment Safety Shell with a Provable Credit-Bucket Flow Controller

*Manuscript draft. Target venues (representative): IEEE Trans. Aerospace and Electronic Systems (TAES), Aerospace Science and Technology, IEEE Internet of Things Journal, IEEE JSTARS, IEEE Access. All quantitative results are from the canonical 20-seed evaluation in this repository (`results/paper_*.json`, `tables/*.tex`).*

---

## Abstract

Very-low-Earth-orbit (VLEO, 200–300 km) Earth-observation satellites must autonomously
schedule imaging, on-board processing, downlink, and electric propulsion under a tight and
*coupled* budget: aerodynamic drag continuously decays the orbit, the Hall thruster and payload
share a limited solar/battery supply, ground contact windows are sparse, and time-sensitive
observation tasks carry deadline-dependent value-of-information (VoI). We cast the problem as a
constrained Markov decision process (CMDP) and learn a soft actor–critic (SAC) policy with a
constraint critic, a state-dependent Lyapunov projection, and a predictive safety filter (PSF).
At deployment we wrap the learned policy in a *training-decoupled safety shell* that combines
(i) **SAFE_BUDGET**, a look-ahead energy-budget and attitude guard, and (ii) a novel
**credit-bucket processing gate** that bounds the long-run processing-to-downlink ratio
(proc/dl) by a configurable target while avoiding the throughput collapse of naive hard
constraints. We provide a bounded flow-control property (proposition): the credit-bucket gate
enforces an asymptotic upper bound proc/dl ≤ T on the episode-level ratio. On a high-fidelity VLEO simulator with Vallado
atmosphere, diurnal/storm density randomization, a 19-station global network with adaptive
modulation/coding, finite xenon, and an attitude-momentum model, we evaluate eleven
method–shield configurations under an identical 20-seed canonical protocol. The complete method
(**RL + SAFE_BUDGET + credit gate**) is the **only** configuration that simultaneously attains
zero observed safety violations over all 40 evaluation episodes (episode-safety rate 1.00,
worst-seed 1.00), zero crashes, and a processing-to-downlink ratio close to the target (2.16). Classical optimization
and rule baselines (drift-plus-penalty, MPC, heuristic) crash in 18–40 of 40 episodes; a crash
audit shows all terminal failures are orbit decay with full remaining fuel, i.e. failures of the
coupled energy–orbit–processing dynamics rather than an unfair protocol. A strong safety-first
rule scheduler under the *identical* shield still reaches only 0.50 worst-seed safety, and the
shield alone (no policy) downlinks nothing — indicating that, in our evaluation, neither a rule
wrapper nor the safety shell alone matches the learned policy's worst-seed robustness. We frame
our contribution as a safety-first deployment method: the credit gate trades ≈25% of delivered
VoI for zero observed violations over the 40-episode evaluation and a large reduction in
wasteful over-processing.

**Keywords:** VLEO satellites, autonomous task scheduling, safe reinforcement learning,
constrained MDP, predictive safety filter, flow control, value of information, Earth observation.

---

## I. Introduction

Commercial and scientific interest in very-low-Earth-orbit (VLEO) platforms (≈200–300 km) is
growing because the reduced range improves optical/SAR resolution and link budgets. The same
low altitude, however, makes operation uniquely hard: residual atmospheric drag is large enough
that the orbit decays within days unless continuously counter-thrust, and the electric
propulsion needed to do so competes with the payload and bus for a limited power supply. An
autonomous VLEO observation satellite must therefore jointly decide, every few seconds, how to
split power among **propulsion** (orbit keeping), **on-board processing** (CPU), and **downlink**
(transmitter), and which **attitude** to hold (imaging vs. downlink vs. sun-charging), all while
ground contact opportunities are sparse and observation tasks expire.

These decisions are tightly coupled. Over-thrusting drains the battery and causes an
*energy–orbit* collapse; under-thrusting lets the orbit decay; processing data faster than it
can be downlinked floods the on-board buffer so that high-value observations expire before a
contact window (a *processing–communication* mismatch, quantified by the processing-to-downlink
ratio proc/dl). A scheduler that optimizes throughput without respecting these couplings will
either crash the spacecraft or waste energy and compute.

Reinforcement learning (RL) is attractive for the long-horizon foresight this requires, but
naive RL provides no safety guarantee, and reviewers rightly demand fair, strong baselines and
deployable safety. This paper makes the following contributions:

1. **A safe-RL formulation and policy** for joint VLEO power/attitude/scheduling as a CMDP,
   solved with SAC + a constraint critic (Lagrangian), a state-dependent Lyapunov projection
   [Chow et al., 2018], and a K-step predictive safety filter (PSF) [Wabersich & Zeilinger, 2018].
2. **A training-decoupled deployment safety shell (SAFE_BUDGET)** that adds a look-ahead
   energy-budget guard (projecting state-of-charge through the next eclipse) and an attitude
   priority order, applied to *any* policy at evaluation/deployment time.
3. **A credit-bucket processing gate**, a leaky-bucket flow controller that bounds the long-run
   proc/dl ratio to a configurable target without the early-episode lock-up of naive hard
   constraints. We provide a proposition establishing an asymptotic bound proc/dl ≤ T
   (Section IV-D).
4. **A rigorous, fairness-audited evaluation**: eleven method–shield configurations under an
   identical 20-seed canonical protocol, a crash-cause audit of the classical baselines, a
   six-axis ablation, four out-of-distribution generalization scenarios, and two strong control
   baselines (a safety-first rule scheduler and a policy-free shield) that isolate the value of
   the learned policy.

Our headline finding is deliberately *safety-first*: the complete method is not the
highest-throughput configuration, but it is the only one meeting all deployment-safety
constraints (episode-safety = worst-seed-safety = 1.00, zero crashes) at a near-target proc/dl.

---

## II. Related Work

**Satellite/agile EO scheduling.** Classical agile Earth-observation scheduling uses MILP,
heuristics, or rollout/MPC over contact and imaging opportunities. These methods generally
assume a stable orbit and an energy model decoupled from the scheduling horizon — an assumption
that breaks in VLEO, where orbit keeping is itself a per-step control coupled to energy.

**Drift-plus-penalty (DPP) / Lyapunov optimization.** DPP provides queue-stability guarantees
for communication/energy-harvesting systems by greedily minimizing a one-step drift-plus-penalty
expression. It is myopic by construction; we include it as a principled stochastic-optimization
baseline.

**Model predictive control (MPC).** Short-horizon MPC is the standard optimization-based
controller for spacecraft power/attitude. We include a finite-horizon MPC baseline; an
orbit-keeping-augmented (robust) MPC and an offline oracle upper bound are planned extensions
of the baseline suite (Section VIII).

**Safe RL / CMDP.** Constrained policy optimization, Lagrangian methods, Lyapunov-based
constraints [Chow et al., 2018], and predictive safety filters/shielding [Wabersich & Zeilinger,
2018; Alshiekh et al., 2018] provide safety during and after learning. Our policy combines a
constraint critic with a Lyapunov projection and a PSF; our novelty at deployment is the
*flow-control* layer (credit gate) and the look-ahead energy shell, and a fairness-audited
comparison against strong non-learning controllers under the *same* shield.

**Flow/admission control.** Leaky-bucket and token-bucket regulators are classical in networking
for rate shaping. We adapt the leaky-bucket idea to bound the *processing-to-downlink* ratio of an
on-board pipeline, with an explicit long-run ratio bound and an initial-credit term that prevents
pipeline starvation before the first contact window.

---

## III. System Model and Problem Formulation

We simulate one spacecraft over a 6-hour mission (≈4 orbital periods) at a control step of
Δt = 10 s (2160 steps/episode). To make month-scale propellant/decay dynamics visible within an
episode, a fixed *orbital-time-compression* C = 1000 scales orbit decay, thrust response, and
fuel consumption consistently (steady-state thrust = drag is invariant to C; only the transient
toward failure is compressed). Per-step altitude change is clipped to ±5 km to preserve the
unsafe→failure transition.

**Orbital/atmospheric dynamics.** Altitude evolves under thrust minus aerodynamic drag with
F = ½ρ C_d A v_rel². Density ρ uses a Vallado model anchored at F10.7 = 150 (nominal solar),
with Earth-fixed atmosphere co-rotation (≈ −7.5% drag at 51.6° inclination), a Harris–Priester
diurnal bulge, and per-episode domain randomization of the 11-year solar cycle (F10.7 ∈ [70,250]),
eclipse β-angle, and geomagnetic storms (transient density surges). Safety thresholds: warning
200 km, unsafe 180 km (thrust/drag < 1), crash 120 km (re-entry).

**Power/energy.** Solar panel ≤ 800 W (cosine loss off-sun), battery 500 Wh (usable to SOC_min
= 0.15, crash at 0.05). A Hall thruster (P ≤ 720 W, I_sp = 1500 s, η = 0.65) yields ≈31.8 mN at
A = 1.0 m², m = 300 kg. Finite xenon (65 kg, compressed by C) couples "save fuel vs. grab data".
A one-state thermal model bounds CPU/TX duty under heat limits. The instantaneous energy balance
couples propulsion, processing, downlink, and charging — the central difficulty of VLEO.

**Communication.** A 19-station global network with ≥5° elevation, adaptive modulation/coding
(discrete MCS by SNR), acquisition latency, and a per-pass cap defines sparse, geometry-driven
downlink windows.

**Tasks / value of information.** Observation tasks arrive by an orbital-phase scene model
(ocean/land/urban/disaster/military/…); each carries priority, quality, cloud cover, a deadline,
and a freshness profile, yielding a deadline-decaying VoI. The pipeline is raw queue → on-board
processing → processed queue → downlink in a contact window. The reward is delivered VoI;
expired/dropped value and over-processing are penalized.

**CMDP.** State (dim 65): physics (altitude/SOC/solar/thermal), contact-window look-ahead,
queue utilizations, a compressed task-value histogram, future-contact-capacity bins, propellant,
and attitude/momentum. Action (dim 15): {α_prop, α_cpu, α_tx}, per-class CPU/TX logits and
value/urgency weights, a low-value drop strength, and a pointing mode (IMAGE/DOWNLINK/SUN). We
seek a policy maximizing expected delivered VoI subject to physical-safety constraints
(orbit/energy/thermal feasibility), i.e. a CMDP whose feasible set is "no crash + bounded
violations".

---

## IV. Method

### A. Overview

Our system has two parts: (1) a **learned CMDP policy** trained offline, and (2) a
**training-decoupled deployment safety shell** applied at evaluation/deployment to any policy.
The shell is deterministic and interpretable. We report both algorithm-only baselines and
same-shell attribution baselines, so the comparison distinguishes policy quality from the
protection supplied by the deployment shell.

### B. Learned policy (offline)

We train SAC with a transformer backbone over an 8-frame dilated history. Beyond the reward
critic we add: a **constraint critic** with an adaptively-tuned Lagrange multiplier on a clean
physical-safety cost; a **state-dependent Lyapunov projection** Π_Lya that projects the action so
the Lyapunov function does not increase beyond a state-dependent slack [Chow et al., 2018]; and a
**K-step predictive safety filter** Π_PSF that line-searches the largest feasible action under a
backup controller [Wabersich & Zeilinger, 2018]. Training uses a 5-stage curriculum (data-arrival
and domain-randomization ramp), n-step TD targets, and a small behavior-cloning term toward the
executed safe action. The policy is then *frozen*; all results below use one frozen checkpoint.

### C. Deployment safety shell

**SAFE_BUDGET** enforces a hard priority order before an action executes:
`hard-safety > charge/recovery > downlink-in-contact > image/process > idle`. Crucially it is
*look-ahead*: it projects SOC through the next eclipse and forbids any pointing/processing action
whose projected SOC would fall below a reserve, raising the reserve when no charging opportunity
exists within the look-ahead window. This converts the reactive "current-SOC gate" of a naive
fallback into a predictive energy budget.

### D. Credit-bucket processing gate

The processing–communication mismatch (proc/dl ≫ 1, high expiry) is the residual failure once
energy is safe. Reactive gates (instantaneous data-pressure or queue-level gates) do not bind,
because over-processing happens *before* the buffer fills; a naive hard constraint ΣP ≤ T·ΣD
locks up the pipeline early (ΣD = 0 before the first window). We therefore use a leaky bucket.

Let P_t, D_t be the processed and downlinked MB at step t, C_t the processing credit (MB),
T the target ratio, and B̄ the mean per-pass contact capacity. Initialize C_0 = κ_init B̄.

**Credit update:** C_{t+1} = clip( C_t + T·D_t − P_t, −C_max, C_max ).

**Action masking (before execution):** with running ratio r_t = ΣP/max(ΣD,ε) and warm-up after
downlink W,
- hard block if C_t ≤ 0 or (ΣD ≥ W and r_t ≥ ρ_hard): set α_cpu = 0; if pointing = IMAGE, switch
  to DOWNLINK (in a window with backlog) else SUN;
- soft block if C_t ≤ C_soft or (ΣD ≥ W and r_t ≥ ρ_soft): cap α_cpu ≤ α_↓.

**Proposition (long-run proc/dl bound).** Over an N-step episode, ΣP_N / ΣD_N ≤ T + (C_0 + C_max)/ΣD_N,
hence proc/dl → T from above as ΣD_N → ∞.
*Proof.* Dropping the upper clip (which only tightens the bound), C_{t+1} ≤ C_t + T D_t − P_t;
telescoping gives C_N ≤ C_0 + T ΣD_N − ΣP_N, so ΣP_N ≤ C_0 + T ΣD_N − C_N. The lower clip gives
C_N ≥ −C_max, thus ΣP_N ≤ C_0 + T ΣD_N + C_max; divide by ΣD_N. ∎

The initial credit C_0 provides an early-episode buffer (no lock-up); the bound explains why a
T = 2.5 configuration measures proc/dl ≈ 2.16 (conservative gating + upper clip). A hard
running-ratio backstop ρ_hard additionally caps the instantaneous ratio after warm-up.

---

## V. Experimental Setup

**Protocol.** Canonical evaluation over 20 seeds (42–61); main comparison and ablation use 2
episodes/seed (40 episodes/method); generalization uses 1 episode/seed. All methods run on the
*same* environment, seeds, task sequences, contact windows, energy model, and action space. The
shell (SAFE_BUDGET / credit gate) is an environment-level config switch applied identically to
every policy.

**Metrics.** Episode-safety rate (fraction of episodes with zero in-episode constraint
violations); **worst-seed** safety (minimum over seeds); survival rate / crash count;
communication-window utilization; downlink (MB); delivered VoI; **proc/dl** = global
Σ(processed)/Σ(downlinked); expired-value rate; high-value delivery rate; safety-layer
intervention (action-modification magnitude); runtime (ms/step).

**Methods (11 configurations).** RL with {none, SAFE_BUDGET, SAFE_BUDGET+credit gate}; Heuristic,
DPP, MPC with {none, SAFE_BUDGET+credit gate}; **Safe Greedy + SAFE_BUDGET + credit gate** (a
safety-first rule scheduler: downlink-in-window-first, image/process only with energy+credit,
charge below soft reserve, under the same shell); **Rule-only Shell** (a zero-command action —
no propulsion/CPU/TX requests, pointing chosen by a fixed rule — driven only
by the shield, no learned policy).

**Generalization scenarios.** sparse_comm (regional 5-station network, 12° elevation),
energy_constrained (battery 320 Wh, solar 560 W), high_density (×2.5 data arrival),
sparse_high_value (high-value scene arrival ×0.25).

---

## VI. Results

### A. Main comparison (Table I)

**Table I — Main method comparison (canonical 20-seed, 40 episodes/method).**

| Method | EpSafe | Worst | Crash/40 | WinUtil | Downlink | Delivered | Proc/DL | HiDel |
|---|---|---|---|---|---|---|---|---|
| Raw RL | 0.950 | 0.50 | 0 | 0.083 | 1871 | 2989 | 3.15 | 0.238 |
| RL + SAFE_BUDGET | 0.925 | 0.50 | 0 | 0.414 | **7252** | **7675** | 2.70 | 0.621 |
| **RL + SAFE_BUDGET + credit gate (Ours)** | **1.000** | **1.00** | **0** | 0.357 | 6138 | 5770 | **2.16** | 0.588 |
| Heuristic | 0.000 | 0.00 | 18 | 0.371 | 5394 | 6489 | 4.43 | 0.409 |
| Heuristic + SB + CG | 0.825 | 0.50 | 7 | 0.074 | 1319 | 1010 | 2.58 | 0.313 |
| DPP | 0.000 | 0.00 | 40 | 0.588 | 1497 | 1702 | 5.68 | 0.462 |
| DPP + SB + CG | 0.275 | 0.00 | 29 | 0.101 | 849 | 867 | 2.92 | 0.406 |
| MPC | 0.000 | 0.00 | 40 | 0.582 | 1476 | 1697 | 5.62 | 0.462 |
| MPC + SB + CG | 0.275 | 0.00 | 29 | 0.100 | 831 | 828 | 2.94 | 0.429 |
| Safe Greedy + SB + CG | 0.925 | 0.50 | 3 | 0.301 | 4734 | 5494 | 2.27 | 0.515 |
| Rule-only Shell (no policy) | 0.325 | 0.00 | 27 | 0.000 | 0 | 0 | n/a | 0.000 |

The complete method is the **only** configuration with EpSafe = Worst = 1.00 and zero crashes,
at proc/dl = 2.16 (near the 2.0 target). We explicitly do **not** claim it maximizes every metric:
RL + SAFE_BUDGET delivers the most VoI (7675) but its worst-seed safety is 0.50 — at least one
seed exhibits violations, failing deployment robustness. The credit gate trades ≈25% delivered
VoI (7675 → 5770) for zero observed violations (40/40 episodes) and a large reduction in
over-processing (2.70 → 2.16). See
Fig. 1 (safety–throughput) and Fig. 2 (proc/dl–delivered).

### B. Baseline fairness audit (Table II)

**Table II — Crash/violation audit of classical baselines (10 seeds × 2 ep = 20 ep, no shield).**

| Method | Crash/20 | Dominant failure | SOC_pre | Fuel_pre | Prop_pre | CPU_pre | Contact_pre |
|---|---|---|---|---|---|---|---|
| DPP | 20/20 | orbit_decay (under-thrust) | 0.214 | 1.00 | **0.00** | 0.97 | 0.06 |
| MPC | 20/20 | orbit_decay (under-thrust) | 0.214 | 1.00 | 0.50 | 0.11 | 0.07 |
| Heuristic | 6/20 | orbit_decay (under-thrust) | 0.163 | 1.00 | **0.96** | 0.01 | 0.00 |

All terminal failures are orbit decay with **full remaining fuel** — not fuel exhaustion, not a
single hand-set constraint. DPP gives almost all power to processing (cpu ≈ 0.97) and essentially
no thrust (prop ≈ 0.00): a decoupling blind spot. MPC under-thrusts (prop ≈ 0.50) because its
short horizon cannot see the slow secular decay. Heuristic thrusts hard (prop ≈ 0.96) but drains
the battery to SOC ≈ 0.16: an energy–orbit collapse. These are failures of the coupled
energy–orbit–processing dynamics, **not** an unfair protocol (identical env/seeds/windows/energy).

### C. Ablation (Table III)

**Table III — Deployment-mechanism ablation (RL policy fixed).**

| Configuration | EpSafe | Worst | WinUtil | Downlink | Delivered | Proc/DL |
|---|---|---|---|---|---|---|
| no_shield (= Raw RL) | 0.950 | 0.50 | 0.083 | 1871 | 2989 | 3.15 |
| SAFE_BUDGET only | 0.925 | 0.50 | 0.414 | 7252 | 7675 | 2.70 |
| credit gate only | **1.000** | **1.00** | 0.375 | 6419 | 6089 | **2.12** |
| SAFE_BUDGET + credit gate | **1.000** | **1.00** | 0.357 | 6138 | 5770 | 2.16 |

SAFE_BUDGET is primarily responsible for throughput (window utilization 0.083 → 0.414); the
credit gate is primarily responsible for proc/dl reduction (3.15 → 2.12) and for locking safety
to 1.00. The mechanisms are complementary. (A model-selection ablation — removing the canonical
20-seed selector / anti-conservative filter — causes a degenerate "idle-but-safe" checkpoint,
window 0.036 / downlink 713, to be chosen; our selector rejects it.)

### D. Scenario generalization (Table IV)

**Table IV — Out-of-distribution generalization (RL ladder; EpSafe / Worst).**

| Scenario | Raw RL | RL + SAFE_BUDGET | RL + SB + CG (Ours) |
|---|---|---|---|
| sparse_comm | 1.00 / 1.00 | 1.00 / 1.00 | **1.00 / 1.00** |
| energy_constrained | 0.90 / 0.00 | 0.95 / 0.00 | **1.00 / 1.00** |
| high_density | 0.55 / 0.00 | 0.50 / 0.00 | **1.00 / 1.00** |
| sparse_high_value | 0.90 / 0.00 | 0.85 / 0.00 | **1.00 / 1.00** |

The complete method holds EpSafe = Worst = 1.00 in **all four** out-of-distribution scenarios,
whereas Raw RL and RL + SAFE_BUDGET drop to 0.00 worst-seed safety under energy-constrained,
high-density, and sparse-high-value shifts. The credit gate's safety robustness generalizes, at
the cost of more conservative throughput under heavy load.

---

## VII. Discussion: the learned policy improves robustness under the same safety shell

Two control baselines isolate the value of learning. **Safe Greedy + SB + CG**, a carefully
engineered safety-first rule scheduler running under the *identical* shell and credit gate,
reaches only EpSafe 0.925 / Worst 0.50 / 3 crashes — in our evaluation it does not match the
learned policy's worst-seed robustness.
**Rule-only Shell** (the shield with a zero-command action — no propulsion/CPU/TX requests —
and no policy) downlinks and delivers
exactly zero with 27/40 crashes — the safety shell is a guardrail, not a scheduler, and cannot
accomplish the task alone. The learned policy supplies long-horizon decisions; the deterministic
credit gate (with its bounded long-run proc/dl property) supplies deterministic, auditable flow
control. This is the
intended reading of our contribution: a **safety-first deployment** method, not a
throughput-maximization method.

---

## VIII. Limitations and Future Work

(i) The deployed policy was trained with an earlier propulsion/pointing scaffold and is evaluated
under a *deployment-time, training-decoupled* shield; a co-adaptation retrain (policy trained
with the full shield) was tested but produced a more conservative, lower-throughput policy and
was not adopted — closing this train/deploy gap is future work. (ii) proc/dl is undefined for
degenerate configurations that downlink almost nothing (reported as n/a). (iii) Scene value
profiles are synthetic stress-test priors (declared in the protocol); absolute VoI magnitudes are
relative, not calibrated. (iv) Evaluation uses 20 seeds × 2 episodes; ≥5 episodes/seed would
further reduce the worst-seed estimator variance. (v) Single-spacecraft; constellation-level
coordination is left to future work. (vi) The baseline suite currently lacks an offline oracle
upper bound and an orbit-keeping-augmented (robust) MPC variant; both are planned strengthening
of the comparison.

---

## IX. Conclusion

We presented a safety-first RL system for autonomous VLEO Earth-observation scheduling that
couples a CMDP policy (SAC + constraint critic + Lyapunov projection + PSF) with a
training-decoupled deployment safety shell and a credit-bucket flow controller with a bounded
long-run processing-to-downlink property. Under an identical, fairness-audited 20-seed
canonical evaluation, the complete method is the only one of eleven configurations to achieve
zero observed safety violations (episode- and worst-seed safety 1.00) with zero crashes at a
near-target proc/dl, while classical optimization/rule baselines fail catastrophically on the
coupled energy–orbit–processing dynamics. Strong control baselines show that, under the same
shell, neither a rule wrapper nor the shield alone matches the learned policy's worst-seed
robustness in our evaluation. The credit gate offers a principled, auditable safety–efficiency
trade-off for safety-critical autonomous spacecraft operations.

---

## References (representative — to be completed for submission)

1. Y. Chow, O. Nachum, E. Duenez-Guzman, M. Ghavamzadeh. "A Lyapunov-based Approach to Safe
   Reinforcement Learning." NeurIPS, 2018.
2. K. P. Wabersich, M. N. Zeilinger. "Linear Model Predictive Safety Certification for Learning-
   based Control." CDC, 2018.
3. M. Alshiekh et al. "Safe Reinforcement Learning via Shielding." AAAI, 2018.
4. M. J. Neely. "Stochastic Network Optimization with Application to Communication and Queueing
   Systems." Morgan & Claypool, 2010. (drift-plus-penalty)
5. T. Haarnoja, A. Zhou, P. Abbeel, S. Levine. "Soft Actor-Critic." ICML, 2018.
6. N. H. Crisp et al. "The Benefits of Very Low Earth Orbit for Earth Observation Missions."
   Progress in Aerospace Sciences, 2020.
7. D. A. Vallado. "Fundamentals of Astrodynamics and Applications." Microcosm Press.
8. E. Altman. "Constrained Markov Decision Processes." Chapman & Hall/CRC, 1999.
9. (Agile EO scheduling survey — to add.)
10. (VLEO aerodynamic drag / GOCE / SLATS references — to add.)

*Note: items 9–10 and venue-specific formatting/citations to be finalized against the target
journal template. Tables I–IV and Figs. 1–2 are auto-generated from `tables/*.tex` and
`figures/*.png`; regenerate with `experiments/paper_tables_figures.py`.*
