# VERIFLOW — Master Research and Execution Plan

**Verifier Error Heterogeneity and Learning Dynamics**
Plan version 1.0 · 2026-10-31 · Author of record: the student researcher (solo). Drafted with Claude Code.

> Status legend used throughout: **[VERIFIED]** = checked against a primary source on 2026-10-31; **[UNVERIFIED]** = could not be confirmed, must be checked before relying on it; **[PROVED]** = elementary proof given here, checked numerically; **[TOY]** = result from a small tabular simulation, not evidence about LLMs; **[TO MEASURE]** = must be measured on the SIT cluster.

> The two attached documents (original proposal and revised critique) were **not present** in `E:\Veriflow` when this plan was written. The critique points restated in the request were used instead. Put both files in `literature/source_docs/` and diff them against §5–§8 before Week 1 ends.

---

## 1. Executive decision

**Proceed, with a modified question, CPU-first.**

1. **Novelty: the exact experiment is open, but the original framing is not defensible as written.** No paper found holds aggregate FPR *and* TPR fixed while redistributing false acceptances across error modes.
   - Egashira et al. (COLM 2026) already show that outcomes are not determined by the aggregate rate or Youden's J. "Beyond aggregate rate" is therefore their result, not ours.
   - Rad et al.'s theory assumes all incorrect modes share one FPR ("block symmetry"). Dropping that assumption is the theoretical opening, and it is a modest step.
2. **Mathematics: variance is not the full story, so it cannot be presented as "the" predictor.** In the idealized replicator model, true accuracy has an exact closed form governed by the **cumulant-generating function** of the per-mode false-acceptance probabilities (§7, Prop. 2; verified numerically to 1e-15).
   - Variance $V$ is its leading term at **short** horizon (Cor. 3a).
   - The worst mode ($f_{\max}$) governs the **long** horizon (Prop. 4).
   - Under **vanilla** policy-gradient geometry, the relevant dispersion is $q^2$-weighted, and its sign can flip (Prop. 7, toy sweep).
   - Whether $V$ predicts anything in a real transformer trained with Adam under GRPO is an open, two-sided empirical question.
3. **Feasibility: moderate.**
   - Qwen2.5-0.5B GRPO on short arithmetic fits one L40 comfortably.
   - The binding constraints are (i) scheduled GPU time shared with ErrorEcho and (ii) whether the base model produces ≥3 identifiable error modes at adequate frequency (Gate 3).
   - All compute numbers are estimates until `veriflow-bench` runs.
4. **Order of work:** theory and simulation gates (CPU, Weeks 1–3) → construct gate (census, Week 5) → feasibility (Week 6) → pilot (Weeks 7–8) → pre-registration → confirmatory grid (January) → analysis (February).
   - The project produces a publishable output even if the GPU arm is killed: a theory-plus-simulation note on which verifier statistics matter in which regime.

## 2. Final research question

**Primary (confirmatory):** among programmatic verifiers matched exactly on aggregate false-acceptance rate (weighted by the base policy's error-mode distribution) and on correct-answer acceptance rate, does the base-weighted variance $V$ of mode-specific false-acceptance probabilities predict the true held-out accuracy reached by GRPO-trained Qwen2.5-0.5B-Instruct, after controlling for the worst-mode acceptance rate $f_{\max}$, and does it do so on held-out verifier configurations?

**Secondary (pre-registered competitors):** does a one-parameter cumulant-generating-function predictor, derived from idealized replicator dynamics, out-predict both $V$ and $f_{\max}$? And how do $J$, trigger frequency and Egashira-style conditional advantage compare when aggregate rates are allowed to vary?

## 3. Scientific contribution

All three are conditional on the gates passing.

1. **A controlled instrument:** matched multi-mode programmatic verifiers that hold $\overline F$ and $\tau$ exactly while varying the *shape* of false acceptance across error modes, with mode-identity randomisation. This is the first exact-matching design of this kind that was found (§5).
2. **A regime theory:** replicator and Gibbs results, standard in evolutionary game theory, applied to verifier error. They show that the per-mode FPR distribution acts on learning through its CGF: variance at short horizon, worst mode at long horizon, and the KL coefficient setting the horizon. They also show this ordering depends on optimisation geometry. This generalises Rad et al.'s block-symmetric analysis and is presented as such, not as new mathematics.
3. **A predictive test:** out-of-configuration forecasting of GRPO outcomes from verifier statistics, comparing $V$, $f_{\max}$, trigger frequency/conditional advantage, $J$ and the CGF predictor. A null result ("dispersion adds nothing beyond the worst mode at this scale") is a valid and reportable answer.

What this project will **not** show: that real-world verifiers (unit tests, LLM judges) behave like these synthetic ones, or anything about models above 1.5B.

## 4. Final title and project name

- **Project:** VERIFLOW (Verifier Error Heterogeneity and Learning Dynamics). Unchanged.
- **Working title:** *Same Error Rate, Different Fate? Error-Mode Structure of Verifier False Positives and the Dynamics of RLVR*
- **Alternative title** (if the CGF predictor wins): *The Shape of Verifier Error: A Cumulant View of Reward Hacking in RLVR*
- The original title ("Beyond Aggregate Verifier Quality: Does Error-Mode Heterogeneity Predict RLVR Failure?") is retired. Its "beyond aggregate quality" claim is Egashira et al.'s established result.

---

## 5. Novelty audit

**Method and its limits.** WebSearch/WebFetch failed in this session. Papers were retrieved through the arXiv export API and full-PDF/HTML text, and read including appendices, on 2026-10-31. OpenReview and general web search were not used, so differently-worded or non-arXiv work may have been missed. **Re-run a Google Scholar / Semantic Scholar search in Week 1** for: "error mode" verifier RLVR, "false positive" heterogeneity GRPO, "per-category" verifier noise.

### 5.1 The five questions from the brief

1. **Is the exact matched-rate contrast published?** **No, as far as could be found.**
   - Egashira et al. use one false-positive trigger per run and track only global FPR/FNR. Their closest contrast (§4.4: asymmetric vs symmetric relative-error tolerance) compares verifiers with *different* initial FPR. The asymmetric one has the lower FPR and collapses.
   - Plesner et al. (2604.07666) match the rate (ε=0.10) across four noise *structures*, but those are response-independent correlation patterns, not error-mode heterogeneity.
   - No paper found holds $\overline F$ and $\tau$ fixed while redistributing $f_m$ across error modes.
2. **Does Rad's theory already imply the predictor?** **No, it assumes it away.**
   - Rad et al. analyse K good and M bad modes under **block symmetry** (App. E): every bad mode has the same advantage $a_b(s)=-J(1-s)/\sigma(s)$, i.e. one shared FPR.
   - Their "heterogeneity" expansion (Eq. 17, Thm I.7) is over initial *policy-mass* heterogeneity ($\|v_0\|^2,\|w_0\|^2$), not acceptance-rate heterogeneity.
   - Their dynamics have vanilla-PG-type $[p(1-p)]^2$ and $\|z\|^2$ factors. These are the $q^2$-weighted terms of §7 Prop. 7, which independently supports the geometry point.
   - **Consequence:** dropping block symmetry is a natural and modest theoretical step. Reviewers will call the theory incremental, and that framing should be adopted rather than fought.
3. **Does Egashira already evaluate the same statistic?** **No, but they own the motivation and a mandatory baseline.**
   - Their explanatory statistic is the per-trigger **conditional advantage** $C(t)=\frac1{|S_t|}\sum_{y\in S_t}A^\*(y)$ together with **initial trigger frequency** (§4.3, Fig. 4). It is descriptive, with no held-out forecasting.
   - Their abstract and §5 already claim outcomes are not determined by the overall error rate or Youden's J.
   - VERIFLOW therefore cannot claim "beyond aggregate rate" as its headline. That is established. What remains open is *which property of a multi-mode error distribution* matters, under *exact* matching, with *out-of-configuration* prediction.
4. **Does Moya derive equivalent dynamics?** **No.** Moya et al. use a single hack set, no false negatives and gradient flow on a log-linear policy. Their Prop. 3.2 gives a log-odds drift for the hacked share driven by leakage and hack bias. There is no per-mode FPR, no dispersion and no CGF.
5. **Is $V$ just a reparameterisation?**
   - **In the idealized model, partly yes.** By Prop. 2, the outcome is an exact function of the CGF $K_0$ of $f$ under $q(0)$. $V$ is its second cumulant, so it is the leading term at short horizon. At long horizon the outcome is governed by $f_{\max}$ and its trigger mass.
   - $V$ is therefore not an independent primitive but a regime-specific summary. **Under vanilla-PG geometry it is not even the right summary** (Prop. 7).
   - This is the main reason the research question must be modified (below).

### 5.2 Gate 0 verdict: **SURVIVES, REQUIRES MODIFICATION**

The exact experiment is open. Gupta & Joshi's 2026 survey (2610.17824, §VIII.D) explicitly lists a theory of response-dependent, exploitable errors under group-relative gradients as an open problem. But the original framing ("variance predicts hacking beyond aggregate rate") is weak on two counts:
- the "beyond aggregate rate" part is already shown by Egashira;
- "variance" is one cumulant of an object (the CGF) that the idealized theory says is the complete predictor.

**Modified framing:** treat the per-mode false-acceptance distribution as the object. Test which of its summaries — **dispersion ($V$)**, **worst-case ($f_{\max}$, $J_{\text{worst}}$)**, **Egashira-style trigger frequency × conditional advantage**, or the **CGF plug-in** — predicts GRPO outcomes at exactly matched $\overline F$ and $\tau$, on held-out configurations. The theory predicts which should win as a function of horizon and optimisation geometry. $V$ remains the pre-registered primary statistic (H1), and the competitors are pre-registered (H2–H3).

### 5.3 Mandatory baselines added by the audit

- **Egashira conditional advantage, per mode:** $C_m=\mathbb E[A^\*\mid\text{mode }m]$ under the base policy with an oracle reward, plus trigger frequency $q_m$. Aggregate them as $\max_m q_m f_m$ and as $\max_m f_m\cdot[\text{sign of }C_m]$, defined before any data.
- **Rad's $J$** (constant within the matched arm; tested in the auxiliary arm).
- **Cai/El Mansouri noise-corrected reward** $(r-\overline F)/(\tau-\overline F)$ as an *extended-design* training ablation. Under exchangeable FPR it removes bias. Under heterogeneous $f_m$, Cai et al.'s App. B.6 shows a residual $\mathrm{Cov}(w,G\mid R^\*)$. Showing that correction fails exactly when $V$ (or $f_{\max}$) is large is a natural, cheap secondary result.

## 6. Related-work matrix

Venue is marked ✔ only when stated on arXiv/official proceedings; otherwise it is "UNVERIFIED".

| Paper (arXiv) | Model / task / algo | Verifier error model | Per-mode FPR? | Matched $\overline F$ & $\tau$, varied dispersion? | Theory | Held-out prediction? | Relation to VERIFLOW |
|---|---|---|---|---|---|---|---|
| Egashira, Vero, Dekoninck, Dorner, Staab, Vechev — *Delay, Plateau, or Collapse* (2605.02909; COLM 2026 ✔ per arXiv) | Qwen3-1.7B-Base, OLMo3-7B (+Qwen2.5-1.5B-Instr. App. B.2); Reasoning Gym decimal chain-sum; DAPO in TRL; 500 steps | random FP/FN flips 20/50 %; systematic triggers (format, language, tolerance intervals, words/trigrams, length) | No (one trigger/run) | No (§4.4 nearest; FPR differs) | none (empirical) | No (descriptive Fig. 4) | Closest empirical work; owns "rate ≠ fate" motivation; its conditional advantage is a baseline; its code is the reuse candidate |
| Rad, Filom, Keivan, Mohajerin Esfahani, Kamalinejad — *Rate or Fate? RLVεR* (2601.04411; ICML 2026 UNVERIFIED) | Qwen2.5-3B; Python code (OpenR1 subset); GRPO in VeRL; 1,410 steps; 5 seeds | class-conditional random flips (TPR, FPR) | No — **block symmetry** across bad modes | No (varies TPR/FPR split at fixed J) | mean-field GRPO; $J$ = net discriminative power; phase transition at $J=0$ | No | The null theory VERIFLOW generalises by dropping block symmetry |
| Moya, Thornley, Lin — *Verifier Errors in RLVR* (2609.35677; preprint) | bandits; Qwen2-0.5B LoRA digit task; GRPO 20 rounds | fixed verifier, one hack set, no FN | No | No | gradient-flow hacking criterion; impossibility of detection from RLVR record; audit control | No | Hacking definition and impossibility results to cite; not a competitor |
| Cai, Wang, Liu et al. — *RLVR yet Noisy Rewards under Imperfect Verifiers* (2510.00915; venue UNVERIFIED) | Qwen2.5-Math-1.5B/7B etc.; DeepScaleR; GRPO (VERL) | asymmetric instance-independent $\rho_0,\rho_1$ | No (varies FN by format only) | No | backward/forward correction; residual $\mathrm{Cov}(w,G\mid R^\*)$ under response-dependent errors (App. B.6) | No | Correction baseline; App. B.6 predicts the failure VERIFLOW can measure |
| El Mansouri, Izzati, Seddik, Lahlou — *Noise-corrected GRPO* (2510.18924; ICML 2026 UNVERIFIED) | Qwen3-0.6B–4B, Llama-3.2; GSM8K/MATH/code | Bernoulli flips $\rho^\pm(q)$ | No | No | gradient shrink $\alpha=(1-\rho^+-\rho^-)\sigma/\tilde\sigma$; debiasing | No | Correction baseline (exchangeable FPR) |
| Helff, Delfosse, Steinmann et al. — *LLMs Gaming Verifiers* (2604.15149; UNVERIFIED) | GPT-5 family, Olmo3; SLR-Bench ILP; Olmo-3 RLVR ~500 steps | one structural FP class (enumeration) | No | No | none | No | Real-world single-mode hacking evidence |
| Wang, Yang, Tian et al. — *CURE* (2506.03136; NeurIPS 2025 Spotlight ✔ per arXiv) | 7B/14B coders; co-evolved unit tests | $p_u, p_{01}, p_{00}$ | No | No | precision bound via $\mu$ | No | Peripheral (learned verifier) |
| Gupta, Joshi — *Checking the Checkers* (2610.17824; AISSA 2026 ✔ per arXiv) | survey of 58 studies | $\rho^\pm$, $J$; random vs systematic, "exploitable" | — | — | none new; names the open theory gap (§VIII.D) | — | Cite as the field's statement that VERIFLOW's gap is open |
| Ray — *Before the Model Learns the Bug* (2606.01066; UNVERIFIED) | fuzzing math/JSON/code verifiers; tabular REINFORCE | per-bug-class FPR/FNR | Measured, not controlled | No | none | No | Evidence that real FPs are mode-structured; taxonomy inspiration |
| Kim, Garg, Peng, Garg — *Correlated Errors in LLMs* (2506.07962; ICML 2025 ✔ per arXiv) | 350+ LLMs, leaderboards | pairwise error agreement | — | — | none | — | Why LLM-verifier FPs concentrate on modes correlated with the policy (external-validity argument) |
| Plesner, Guzmán, Athalye — *An Imperfect Verifier is Good Enough* (2604.07666; UNVERIFIED) | Qwen3-8B; MBPP | ε=0.10 under 4 noise structures | No | **Rate matched, structure varied — but response-independent** | none | No | Nearest "matched-rate" design; must be cited and distinguished |
| Zhang — *When the Reward Suite Is Leaky…* (2607.11022; UNVERIFIED; abstract-level) | GRPO on MBPP vs MBPP+ (preregistered) | natural per-task FPs | Natural, not controlled | No | none | static audit predicts FP mass (ρ=0.80) | Shows selection of existing modes (consistent with Prop. 1) |
| Xin — *Are Verifier Errors Independent Within a GRPO Group?* (2609.06386; UNVERIFIED) | GRPO groups | within-group error correlation 0.53 | — | — | none | — | Motivates group-level stochasticity (B1/B3) |
| Xin — category-level verifier audit (2609.01354; UNVERIFIED, abstract-level) | verifier audit | per-category errors | Measured | No RL | none | — | Similar aggregate error, opposite failure reasons |
| Shao et al. — *Spurious Rewards* (2506.10947) | Qwen2.5-Math-7B; MATH | random/incorrect rewards | — | — | none | — | Model-specific priors can dominate reward signal; caution for Qwen |
| Huang et al. — *Pitfalls of rule- and model-based verifiers* (2505.22203; EMNLP 2026 per arXiv) | math RL | model-based verifier FPs | Partly | No | none | — | Real verifier FP modes |
| Liu et al. — *Dr. GRPO* (2503.20783) | — | — | — | — | GRPO length/std biases | — | Justifies pinning loss_type/scale_rewards |

Local copies of the extracted full texts used for this audit are in the session scratchpad and should be replaced by PDFs in `literature/` (Week 1).

---

## 7. Mathematical formulation

### 7.1 Objects

| Symbol | Definition |
|---|---|
| $x \sim \mathcal D$ | arithmetic prompt; splits $\mathcal D_{\text{census}}, \mathcal D_{\text{train}}, \mathcal D_{\text{eval}}$ are disjoint by operand pair |
| $y \sim \pi_\theta(\cdot\mid x)$ | completion; $a(y)$ = extracted final answer (or ∅) |
| $c(x,y)\in\{0,1\}$ | true correctness: $a(y)$ parses to the integer answer of $x$ |
| $\mu(x,y)\in\{\mathsf C, 1,\dots,M, \mathsf U, \mathsf I\}$ | outcome class: correct, designed error mode $m$, unclassified wrong integer, invalid/unparseable |
| $\mathcal E = \{1..M,\mathsf U,\mathsf I\}$ | all incorrect classes |
| $p_k(t)$ | policy probability of class $k$ at training step $t$ (averaged over $x\sim\mathcal D_{\text{eval}}$ unless stated) |
| $P_E(t)=\sum_{k\in\mathcal E}p_k(t)$ | error probability; $A(t)=p_{\mathsf C}(t)=1-P_E(t)$ = true accuracy |
| $q_k(t)=p_k(t)/P_E(t)$ | conditional error-mode distribution; $q_k(0)$ = base-policy census |
| $\tau=\Pr(R=1\mid\mathsf C)$ | correct-answer acceptance (TPR) |
| $f_k=\Pr(R=1\mid k)$, $k\in\mathcal E$ | mode-specific false-acceptance probability |
| $q^{\text{ref}}$ | **reference distribution** = census point estimate of $q(0)$ on $\mathcal D_{\text{census}}$, frozen before any training (§11) |
| $\overline F=\sum_k q^{\text{ref}}_k f_k$ | aggregate FPR under the reference distribution (**primary aggregate definition**) |
| $\overline F_{\text{macro}}=\frac1{|\mathcal E|}\sum_k f_k$ | unweighted macro FPR (reported, never matched on) |
| $V=\sum_k q^{\text{ref}}_k (f_k-\overline F)^2$ | weighted heterogeneity (candidate primary statistic) |
| $V_{\text{unw}}$, $f_{\max}=\max_k f_k$, $\kappa_3=\sum_k q^{\text{ref}}_k(f_k-\overline F)^3$, IQR, $H_q(f)$ | competing dispersion/concentration statistics |
| $K(s)=\log\sum_k q^{\text{ref}}_k e^{s f_k}$ | cumulant-generating function (CGF) of $f$ under $q^{\text{ref}}$ |
| $J=\tau-\overline F$ | Youden index / net discriminative power at the reference distribution |
| $J_{\text{worst}}=\tau-f_{\max}$ | worst-case conditional advantage of correct over the most-accepted error mode |
| $\pi_{\text{trig}}=q^{\text{ref}}_{\arg\max f}$ | initial trigger frequency of the most-accepted mode |
| $\bar R(t)$ | mean assigned reward on training rollouts at step $t$ |
| $\Gamma(t)=\bar R(t)-A_{\text{train}}(t)$ | reward–accuracy divergence (assigned minus true, same rollouts) |
| $r$ | verifier configuration $(\tau, f_1..f_M, f_{\mathsf U}, f_{\mathsf I})$; $s$ = training seed; a **run** is a pair $(r,s)$ |

Verifier: $R\mid(x,y)\sim\text{Bernoulli}(\rho(\mu(x,y)))$ with $\rho(\mathsf C)=\tau$, $\rho(k)=f_k$.

**Fixed conventions (pre-registered now, cannot change after confirmatory data):**
1. The primary aggregate rate is $\overline F$ weighted by the frozen census $q^{\text{ref}}$, not by the evolving $q(t)$ and not macro-averaged.
2. $f_{\mathsf U}=\overline F$ and $f_{\mathsf I}=0$ in every condition. $\mathsf U$ therefore contributes zero to $V$, and invalid outputs are never rewarded. Designed heterogeneity lives only in modes $1..M$, and the matching constraint is solved including $\mathsf U$'s mass.
3. $\tau$ is identical across matched configurations.

### 7.2 Idealized model (I): mode-level replicator dynamics

**Assumptions A1–A5.**
A1 Single prompt type (or prompt-homogeneous class probabilities).
A2 The policy is a free categorical distribution over classes (no shared parameters).
A3 Update = natural policy gradient / multiplicative weights in continuous time: $\dot\theta_k = \lambda(t)\,(\rho_k-\bar\rho)$ with logits $\theta$ and any positive scalar $\lambda(t)$.
A4 Exact expected rewards (infinite group size, no sampling noise).
A5 No KL term, no clipping, and classes are fixed (no new modes appear).

Under A1–A5, $\dot p_k=\lambda p_k(\rho_k-\bar\rho)$ (replicator equation). Write $\tilde t=\int_0^t\lambda$.

**Proposition 1 (closure of the error-mode distribution) [PROVED].**
$\dot q_k=\lambda q_k\,(f_k-F(t))$, with $F(t)=\sum_k q_k(t)f_k$. Hence $q_k(\tilde t)=q_k(0)e^{f_k\tilde t}/\sum_j q_j(0)e^{f_j\tilde t}$, independent of $p_{\mathsf C}$ and of $\tau$.
*Proof.* $\dot P_E=\sum_{k\in\mathcal E}\dot p_k=\lambda P_E(F-\bar\rho)$. Then $\dot q_k=(\dot p_kP_E-p_k\dot P_E)/P_E^2=\lambda q_k[(f_k-\bar\rho)-(F-\bar\rho)]$. ∎

**Proposition 2 (exact accuracy law) [PROVED].**
$$\operatorname{logit}A(\tilde t)=\operatorname{logit}A(0)+\tau\,\tilde t-K_0(\tilde t),\qquad K_0(s)=\log\sum_k q_k(0)e^{s f_k}.$$
*Proof.* $\frac{d}{dt}\log(p_{\mathsf C}/P_E)=\lambda[(\tau-\bar\rho)-(F-\bar\rho)]=\lambda(\tau-F)$, and by Prop. 1, $F(\tilde t)=K_0'(\tilde t)$. Integrate. ∎
Numerical check: max |ODE − closed form| = 2.8e-15 (`docs/preliminary/cgf_check.py`).

**Proposition 3 (Fisher/Price identity) [PROVED].** $\frac{dF}{d\tilde t}=\operatorname{Var}_{q(t)}(f)$. At $t=0$ the effective FPR rises at rate exactly $V_0=\operatorname{Var}_{q(0)}(f)$. Checked numerically: 0.0157851 vs 0.0157851.

**Corollary 3a (matched-rate contrast).** For verifiers A and B with equal $\tau$ and equal $\overline F$ (with $q^{\text{ref}}=q(0)$), the cumulant expansion $K_0(s)=\overline F s+V s^2/2+\kappa_3 s^3/6+\dots$ gives
$$\operatorname{logit}A_B(\tilde t)-\operatorname{logit}A_A(\tilde t)=-(V_B-V_A)\tfrac{\tilde t^2}{2}-(\kappa_{3,B}-\kappa_{3,A})\tfrac{\tilde t^3}{6}-\dots$$
**Variance is the leading-order term at short effective horizon, and only that.**

**Proposition 4 (long-horizon behaviour) [PROVED].** As $\tilde t\to\infty$, $K_0(\tilde t)=f_{\max}\tilde t+\log q_{\arg\max}(0)+o(1)$. Hence:
- if $f_{\max}>\tau$, $A\to0$ (collapse onto the most-accepted mode);
- if $f_{\max}<\tau$, $A\to1$, with an asymptotic logit delay of $-\log q_{\arg\max}(0)$ relative to ideal;
- accuracy peaks at the $\tilde t^\*$ solving $K_0'(\tilde t^\*)=\tau$, which exists iff $f_{\max}>\tau$.

At long horizon the outcome is governed by $f_{\max}$, its trigger frequency, and $\tau$, not by $V$. **Peak step is a function of the whole CGF and of the unknown time change $\lambda$, which is why peak step must not be the primary outcome.**

**Proposition 5 (time-change invariance) [PROVED].** Any positive common multiplier $\lambda(t)$ (learning-rate schedule, or GRPO's per-prompt std normalisation when A1 holds) changes only the clock $\tilde t$, not the orbit. Consequence: in the idealized model, two verifiers at the same *step* can differ partly because their $\lambda(t)$ differ (GRPO's $1/\sigma(\bar\rho)$ depends on the state). This is a known nuisance; secondary analyses index outcomes by cumulative $\int\lambda$ estimated from logged advantage magnitudes.

**Proposition 6 (KL-regularized optimum).** For the objective $\mathbb E_\pi[\rho]-\beta\,\mathrm{KL}(\pi\|\pi_0)$ over free categoricals, $\pi^\*_k\propto\pi_{0,k}e^{\rho_k/\beta}$ (standard Gibbs variational result). So $\operatorname{logit}A^\*=\operatorname{logit}A(0)+\tau/\beta-K_0(1/\beta)$: the same law with horizon $1/\beta$. A KL coefficient therefore sets the effective horizon of Corollary 3a.

**Proposition 7 (geometry dependence) [PROVED + TOY].** Under *vanilla* softmax policy gradient on logits, $\dot\theta_k=p_k(\rho_k-\bar\rho)$ (not A3),
$$\frac{dF}{dt}=P_E\sum_k q_k^2\,(f_k-F)(f_k-\bar\rho),$$
which is a $q^2$-weighted covariance and can be negative even when $V>0$.
**[TOY]** In 150–180 random matched 5-mode configurations (`docs/preliminary/sweep.py`; $\tau=0.9$, $\overline F=0.3$, $A(0)=0.4$), the Spearman correlation between late-window accuracy and $V$ was:

| Horizon | Natural PG | Vanilla PG | Adam |
|---|---|---|---|
| Short | −1.00 | **+0.11** | −0.79 |
| Medium | −0.93 | −0.38 | −0.65 |
| Long | −0.80 | −0.85 | −0.45 |

The CGF plug-in $\tau s-K(s)$ was the best or tied-best predictor in the natural-PG and Adam settings at all horizons. Under vanilla PG it was poor (+0.14, −0.09, −0.51).

**Interpretation that drives the whole project:** whether, and with which sign, $V$ predicts outcomes depends on the effective optimisation geometry of the real network. That geometry is unknown for a transformer trained with Adam under GRPO. This is a genuine empirical question with a non-obvious answer, which is what makes a null result informative.

> Honesty note: Props. 1–4 are textbook replicator-dynamics facts (Fisher's fundamental theorem; constant-fitness replicator solutions; Gibbs posterior for Prop. 6). The novelty is **not** the mathematics. It is (i) using them to identify which verifier statistics should matter in which regime and (ii) testing that empirically. Present them as "standard results applied", with citations to evolutionary game theory (Hofbauer & Sigmund 1998) and to mirror descent / multiplicative weights.

### 7.3 Where the idealized model stops applying to neural GRPO

The actual GRPO update for prompt $x$ with group $\{y_i\}_{i=1}^G$ and stochastic rewards $R_i$ is
$$\hat A_i=\frac{R_i-\operatorname{mean}(R)}{\operatorname{std}(R)+\epsilon}\quad(\text{or without std, for Dr.GRPO-style variants}),$$
$$\nabla\mathcal L=\frac1{\sum_i|y_i|}\sum_i\sum_{t}\nabla\min\big(\varrho_{i,t}\hat A_i,\operatorname{clip}(\varrho_{i,t},1\pm\epsilon_c)\hat A_i\big)-\beta\nabla\mathrm{KL}.$$
Break points, each mapped to a test:

| # | Departure | Effect on the idealized law | Where tested |
|---|---|---|---|
| B1 | Group-relative advantage with **finite G** | All-equal groups give zero gradient with probability $\bar\rho_x^G+(1-\bar\rho_x)^G$. The mean/std estimates are biased at small G. Effect is a state-dependent time change plus noise. | `sim_grpo_tabular.py` (finite-G Monte Carlo) |
| B2 | **Std normalisation** $1/\sigma(\bar\rho_x)$ | Per-prompt time change. With many prompts the multipliers differ by prompt, so the orbit is no longer invariant. | same |
| B3 | **Stochastic verifier decisions** | Adds Bernoulli noise; unbiased in expectation, so it changes variance, not drift. | same |
| B4 | **Policy-ratio clipping** | With `num_iterations=1` the ratio is 1 at the update and clipping is inactive. Becomes relevant only if μ>1. | config pin: μ=1 |
| B5 | **KL regularisation** | Bounds the horizon at $1/\beta$ (Prop. 6). | β=0 primary; β>0 ablation (extended design) |
| B6 | **Shared parameters across prompts and modes** | The class logits are not free. Gradient on mode $m$ for prompt $x$ moves mode $m'$ and other prompts. Replicator closure (Prop. 1) fails. | `sim_shared_policy.py` |
| B7 | **Prompt heterogeneity** $q(0\mid x)$ | Even under A2–A4 per prompt, the population accuracy is an average of per-prompt laws: $\mathbb E_x[\sigma(\cdot - K_{0,x}(\tilde t))]\neq\sigma(\cdot-K_0(\tilde t))$. $V$ computed from the pooled $q^{\text{ref}}$ can mis-rank configurations. | census reports per-prompt $q$; sim with prompt mixtures |
| B8 | **Optimiser geometry (Adam, token-level)** | Prop. 7: the relevant statistic may be $q^2$-weighted or sign-like rather than $V$. | `sim_shared_policy.py` with SGD vs Adam |
| B9 | **Mode creation/drift** | RL can create errors absent at $t=0$ (they land in $\mathsf U$) and change lengths. | per-checkpoint mode census; $\mathsf U$ fraction is a secondary outcome |
| B10 | **Sequence-level classes** | A "mode" is a set of token sequences. Credit is token-level; length differences change per-token gradient scale (loss_type matters). | log per-mode lengths; fix loss_type across conditions |

Levels of claim:
1. **Analytical theorem:** Props. 1–6 under A1–A5.
2. **Idealized model:** replicator / Gibbs.
3. **Numerical simulation:** finite-G GRPO on tabular and shared-parameter policies (§12).
4. **Empirical neural observation:** Qwen2.5 GRPO runs (§13–14).

The paper must never move a claim up a level.

### 7.4 Choice of primary heterogeneity statistic

The primary confirmatory statistic is $V$ (the $q^{\text{ref}}$-weighted variance), because:
(a) it is the unique leading-order term of Corollary 3a;
(b) it is what the proposal set out to test;
(c) it is defined before any GPU data.

$f_{\max}$, $\kappa_3$, $V_{q^2}$ and the CGF plug-in $Z(s)=\tau s-K(s)$ are pre-registered **competitors**, not fallbacks. **Lock rule:** if Gate 2 (shared-policy simulation) shows $V$ is dominated by $V_{q^2}$ under every Adam-trained shared-parameter setting, switching the primary statistic is allowed *only before* Gate 5 data exist, and the switch is logged in `docs/DECISIONS.md` with the simulation evidence.

---

## 8. Theoretical assumptions and limitations

1. **Synthetic verifier.** The verifier *knows* the true error mode and accepts by design. This is a controlled instrument, as in systematic-error studies, not a realistic checker. External validity to real buggy verifiers (regex matchers, unit tests, LLM judges) is not established by this project and must be stated as a limitation.
2. **Arithmetic only.** Modes are clean in arithmetic because the answer is an integer. Transfer to math word problems or code is out of scope for the confirmatory study.
3. **Reference distribution.** $\overline F$ and $V$ are matched under the census $q^{\text{ref}}$. Under the policy's evolving $q(t)$ the realised FPR diverges by design (that divergence *is* the mechanism). Realised rates are logged but never re-matched.
4. **Small models.** Qwen2.5-0.5B (and 1.5B replication). Dynamics at 7B+ may differ, especially mode creation.
5. **Fixed horizon.** Conclusions are horizon-specific (Corollary 3a vs Prop. 4). The horizon is fixed from the pilot *before* the confirmatory grid.
6. **The theory does not predict the sign under Adam with shared parameters.** The confirmatory test is therefore two-sided.

---

## 9. Dataset and error taxonomy

### 9.1 Task

Primary: **integer addition of two $n$-digit operands**, answer-only format with a short scratch allowance. The final answer goes on a line `Answer: <integer>`. Addition is preferred over multiplication for construct validity: every candidate error mechanism (dropped carry, digit slip, transposition, truncation) has a closed-form counterfactual answer that can be matched exactly. Multiplication errors (partial-product misalignment) are much harder to attribute deterministically.

- $n$ is chosen in the census (§11) from {5, 6, 7, 8, 9, 10} as the smallest $n$ with base accuracy (temperature 1.0) in [0.25, 0.60] and designed-mode coverage ≥ 70 % of errors. Rationale: room to improve and room to collapse, with enough errors to estimate $q$.
- **Relation to ETH's setup [VERIFIED, commit `913e3bf642747d9e178775a8a84f0e59fc9567f0`, MIT]:** Egashira et al. use reasoning-gym `decimal_chain_sum` (3–6 terms, 3–6 digits, 3–6 decimal places) with `\boxed{}` answers. VERIFLOW deliberately uses **two-operand integer** addition instead, because multi-term decimal chains do not admit a clean deterministic error taxonomy (a dropped carry in which term?). This is a documented departure, and it makes direct numerical comparison with their results impossible. A small replication of one of their configurations on our stack is an **extended-design** item (it validates the training stack against a published result).
- Prompt template is frozen and hashed. No few-shot examples (they shift $q(0)$).
- Operand pairs are sampled with a split-specific seed. Splits are disjoint on the unordered operand pair: census 2,000 prompts; train 50,000; eval 1,000 (fixed); eval-mode 1,000 (for per-checkpoint mode census).

### 9.2 Mode taxonomy v1 (deterministic, hypothesis-matching)

For prompt $(a,b)$ with true sum $S$, compute a **hypothesis set** of counterfactual answers. Classify the parsed answer $\hat S$ by the first matching rule in this priority order:

| Priority | Code | Rule (exact) |
|---|---|---|
| 0 | `C` | $\hat S=S$ (after stripping commas/spaces/leading `+`; leading zeros allowed) |
| 1 | `I` | no parseable integer on the `Answer:` line, or several conflicting `Answer:` lines |
| 2 | `ECHO` | $\hat S\in\{a,b\}$ or $\hat S$ = digit concatenation of $a$ and $b$ |
| 3 | `CARRY` | $\hat S$ equals the sum computed with **any non-empty subset of carries dropped** (enumerated; $\le 2^{n}$ cases, $n\le10$) |
| 4 | `TRUNC` | digits of $\hat S$ are a proper prefix or proper suffix of the digits of $S$ |
| 5 | `TRANS` | $\hat S$ equals $S$ with exactly one adjacent digit pair swapped (pairs with distinct digits only) |
| 6 | `DIGIT` | same length as $S$, differs in exactly one digit position, by any amount |
| 7 | `U` | any other integer |

Notes and handling:
- **Overlap:** a dropped single carry also makes one digit off by one. The priority puts `CARRY` before `DIGIT` because it is the more specific mechanism. Every classification records *all* matching rules (a multi-hot vector), not only the winner. The **ambiguity rate** (fraction of errors matching ≥2 rules) is reported per mode.
- **Parser failure** is `I`, never `U`.
- **Formatting-only differences** (commas, spaces) are treated as correct. Format is not a designed mode, because amplifying it is not semantic reward hacking and would confound the construct.
- **Taxonomy versioning:** `error_modes/TAXONOMY_VERSION = "v1"`. Any change bumps the version and invalidates every census that used the old version.

### 9.3 Construct-validity gate (part of Gate 3)

All criteria are set *before* the census is run and are justified by identifiability: a mode with tiny mass cannot carry designed heterogeneity.

1. **Coverage:** designed modes (ECHO, CARRY, TRUNC, TRANS, DIGIT) ≥ 70 % of incorrect outputs. Otherwise $\mathsf U$ dominates and the matched configurations differ mostly by $f$ on rare modes.
2. **Mass:** at least **3** designed modes each with $q^{\text{ref}}_k\ge 0.05$. Modes below 0.02 are merged into `U` *before* configurations are generated.
3. **Ambiguity:** ≤ 10 % of designed-mode assignments match more than one rule. Otherwise report a merged mode (e.g., CARRY∪DIGIT).
4. **Stability:** total-variation distance between $q$ estimated on two disjoint halves of the census ≤ 0.03.
5. **Synthetic recovery test:** for each mode, inject 1,000 synthetic errors of that mode into random problems. The classifier must recover the injected mode at ≥ 99 %, excluding cases that are provably identical under two rules, which are counted as ambiguous.
6. **Manual audit:** 200 random classified errors inspected by the researcher with the classifier label hidden. Disagreement > 5 % triggers redesign.

**If the gate fails:** first try a different $n$ or an answer-only prompt. Second, try 2-operand subtraction. Third, accept $M=3$ modes. If fewer than 3 modes reach 5 % mass, **stop**: the matched-heterogeneity design cannot be built on this model and task.

---

## 10. Verifier construction

### 10.1 Decision rule

**Reuse audit of `eth-sri/llm-verifier-noise` (pinned at `913e3bf`):**
- Noise is injected in `Mixup.mixup_rewards()` (`src/data/base.py`).
- Its `TARGETED` strategy applies per-item TPR/FPR from the first matching *selector bucket* `{path, op, value}`, then a Bernoulli flip.
- This is exactly the per-mode verifier abstraction VERIFLOW needs. **Reuse the design, not the file**, because:
  - the repo pins `vllm==0.11.2`, `transformers<5` and `trl>=0.25`, and subclasses GRPOTrainer internals (`GRPOTrainerDebug`), which conflicts with current TRL 1.17.x;
  - its draws use the global `np.random`, which is not an independent, replayable stream.
- VERIFLOW reimplements the bucket logic as a TRL reward function, with each taxonomy mode as one bucket. Credit the design in code comments and the paper.
- Also reuse their `scripts/conditional_advantage.sh` logic to compute the Egashira conditional-advantage baseline.

`verifiers/noisy_verifier.py`

```
decide(prompt_id, completion, step, sample_idx, config) -> VerifierRecord
  mode   = classify(prompt, completion)            # taxonomy v1
  rho    = config.tau if mode == "C" else config.f[mode]
  u      = philox_uniform(key=(config.verifier_seed, run_seed, step, prompt_id, sample_idx))
  reward = 1.0 if u < rho else 0.0
  return VerifierRecord(mode, all_matching_modes, correct=(mode=="C"), rho, u, reward)
```

- **Independent stream:** counter-based RNG (`numpy.random.Philox` keyed by a SHA-256 of the tuple). Verifier randomness is therefore independent of the sampling RNG and of the trainer, bit-reproducible, and replayable offline.
- **Per-call independent draws** are primary: they match the Bernoulli assumption in §7. A deterministic-per-string variant (`u` keyed on `sha256(completion)` instead of step/sample) is an extended-design ablation only.
- Every decision is appended to `runs/<run_id>/verifier_log.parquet` with fields `step, prompt_id, sample_idx, completion_sha, mode, modes_multi, correct, rho, u, reward`. **True correctness is stored separately from assigned reward.**

### 10.2 Matched configuration generator

`verifiers/matched_verifiers.py` solves, for designed modes $1..M$ with masses $q^{\text{ref}}$ (including fixed $f_{\mathsf U}=\overline F$ and $f_{\mathsf I}=0$):

$$\sum_k q^{\text{ref}}_kf_k=\overline F^\*,\quad \sum_kq^{\text{ref}}_k(f_k-\overline F^\*)^2=V^\*,\quad f_{\max}=f^\*_{\max}\ (\text{when targeted}),\quad 0\le f_k\le 1.$$

- Since $f_{\mathsf I}=0$ is fixed, $\overline F^\*$ must be reachable. The generator solves for the designed modes so the *total* weighted FPR, including `U` and `I`, equals $\overline F^\*$.
- Solved by projected optimisation (`scipy.optimize.minimize`, SLSQP) from multiple starts, then **verified exactly**. Unattainable targets raise an error.
- **De-confounding design:** $V$ and $f_{\max}$ are strongly correlated in random designs. The generator samples configurations on a 2-D grid of $(V, f_{\max})$, plus a $\kappa_3$ sign split, so their effects are separable.
- **Mode-identity randomisation:** for each target vector, the assignment of $f$-values to modes is a random permutation, with the constraint re-solved after permuting. This prevents "TRUNC is just easy to learn" from masquerading as a heterogeneity effect. The permutation is part of the configuration identity.

### 10.3 Rates distinguished in every log and table

1. **Configured** $\tau, f_k$.
2. **Empirical realised** rates: accepted/attempted per class, from `verifier_log`.
3. **Base-policy-weighted aggregate** $\overline F$ under $q^{\text{ref}}$ (the matched quantity).
4. **On-policy aggregate** $F(t)=\sum_kq_k(t)f_k$ (an outcome, not a control).
5. **Macro** average $\overline F_{\text{macro}}$.

### 10.4 Tests (`tests/test_verifiers.py`)

- Matching holds to 1e-12 for $\overline F$ and $V$ in every generated config; all $f\in[0,1]$.
- Realised rates over $10^6$ synthetic decisions per class fall within 4.5 binomial SEs of configured rates (family-wise ≈1e-5 false alarm).
- Replay determinism: identical keys give identical rewards across processes and machines.
- Stream independence: correlation of $u$ between adjacent sample indices < 3/√N; and the verifier makes no draws from the global numpy/torch RNG (asserted by snapshotting RNG state).
- Homogeneous config: $V=0$ exactly.
- `I` is never rewarded; `C` is accepted at rate τ.

---

## 11. Base-policy error census

`experiments/base_mode_census.py` (GPU job, about 0.2–0.5 GPU-h [TO MEASURE])

- Inputs: model + tokenizer at pinned revision, frozen prompt template, `D_census` (2,000 prompts), sampling exactly as in training (T=1.0, top_p=1.0, same max_completion_length), $n$ candidates {5..10}.
- **Sample size:** 2,000 prompts × 4 samples = 8,000 generations per $n$. With an error rate around 0.5, that gives ≈4,000 errors. The cluster-bootstrap SE (clustered by prompt) on a mode with $q=0.10$ is ≈0.006, enough to place $f$ vectors whose matched $\overline F$ is insensitive to $q$ error at the third decimal.
- Outputs (`runs/census_<hash>/`):
  - `generations.parquet`
  - `classification.parquet` (multi-hot)
  - `q_ref.json` (point estimate + 95 % cluster-bootstrap CI)
  - per-mode output-length distribution
  - per-prompt accuracy histogram (difficulty)
  - per-prompt mode entropy (prompt-level variation, B7)
  - split-half TV distance (stability)
  - coverage, ambiguity
- `q_ref.json` is **frozen** with its SHA-256 recorded in `configs/locked/q_ref.lock`. All configurations are generated from it.
- **Leakage control:** census, verifier-calibration (synthetic), train and eval prompts come from disjoint operand pairs and different seeds. The eval set is never used to choose $n$ or any hyperparameter. $n$ is chosen on the census split only.

---

## 12. Simulation plan (CPU, Gates 1–2)

### 12.1 `simulations/sim_replicator.py`

Functions (each unit-tested):
- `replicator_rhs(p, rho)`, `integrate(p0, rho, T, method="rk4"|"expm")`
- `closed_form_accuracy(A0, q0, tau, f, t)` (Prop. 2)
- `cgf(q, f, s)`, `cgf_prime`, `peak_time(q, f, tau)` (root of $K'=\tau$, or `inf`)
- `make_matched_family(q, Fbar, V_grid, fmax_grid, n_perm, rng)` → reuses `verifiers/matched_verifiers.py`, so simulation and experiment share one implementation
- `predictors(q, f, tau)` → dict {FPR, macroFPR, J, J_worst, fmax, trigger, V, V_unw, V_q2, kappa3, IQR, H, Z(s)}
- `forecast_eval(outcomes, predictors, cv="leave-one-config-out")` → MSE, Spearman, calibration slope

Sweeps (exported as Parquet plus `params.json`):
- $M\in\{3,4,6,8\}$
- $q(0)\sim$ Dirichlet(α) with α∈{0.3, 1, 5} (concentrated → flat)
- $\overline F\in\{0.1,0.2,0.3,0.4\}$
- $V$ from 0 to the feasible maximum
- $f_{\max}$ on a grid
- $\tau\in\{0.8, 0.9, 1.0\}$
- horizon $\tilde t\in\{0.5,1,2,4,8,16,32\}$
- seeds for the random families = 20

Outputs: Figure 3 panels and the "regime map" (best predictor as a function of horizon × concentration).

**Exact tests:**
- ODE vs Prop. 2 to 1e-10.
- $dF/dt|_0=V$ (finite difference, 1e-6).
- Homogeneous verifier gives logit accuracy linear in $t$ with slope $J$.
- $f_{\max}>\tau$ gives an interior peak at the root of $K'=\tau$.
- Prop. 6: Gibbs optimum vs long-run KL-regularised ODE to 1e-8.

### 12.2 `simulations/sim_grpo_tabular.py`

Finite-G Monte Carlo GRPO on a per-prompt tabular policy, with:
- P prompts with heterogeneous $q(0\mid x)$ drawn from census-like mixtures;
- G ∈ {4, 8, 16};
- Bernoulli verifier;
- std-normalised and non-normalised advantages;
- optional KL.

Tests B1–B3 and B7. Compare with the closed form after the time change.

### 12.3 `simulations/sim_shared_policy.py`

Shared-parameter policy: $\pi_\theta(k\mid x)=\operatorname{softmax}(W\phi(x,k))$ with features
$\phi(x,k)=[e_k;\ \psi(x)\otimes e_k;\ \text{shared "carry-skill" features}]$,
so that improving the correct class on one prompt partially transfers to others and some modes share features (e.g., CARRY and DIGIT share a "carry-handling" feature).

Variants:
- (i) linear softmax, SGD;
- (ii) linear softmax, Adam;
- (iii) 2-layer MLP feature map, Adam;
- (iv) L2 regularisation toward init (KL proxy);
- (v) mode-generalisation coupling strength ∈ {0, 0.3, 0.7}.

Each variant is trained with finite-G GRPO on 64–256 synthetic prompts.

Question answered: does $V$ add leave-one-config-out predictive information beyond $f_{\max}$ (and beyond $Z(s)$), and with what sign, across these geometries?

**This simulation decides whether the GPU study is worth running (Gate 2).**

---

## 13. Pilot (Gates 3–5)

Smallest defensible pilot, Qwen2.5-0.5B-Instruct only:

1. **Hyperparameter pilot (homogeneous verifier only, $V=0$, $\overline F^\*$, $\tau^\*$):**
   - learning rate ∈ {5e-7, 1e-6, 3e-6} × 1 seed, run to 800 steps;
   - plus a perfect verifier ($\tau=1$, all $f=0$) × 1 seed.
   - Choose the LR and fix the **horizon T** as the step at which the homogeneous run reaches 90 % of its maximum held-out accuracy gain, rounded up to a multiple of 50 (rule fixed now). This keeps the confirmatory window in a regime where both accuracy gains and heterogeneity-driven losses are visible.
2. **Heterogeneity pilot:** 4 configurations ($V$ ∈ {0, mid, high} plus one high-$f_{\max}$/low-$V$ config) × 3 seeds = 12 runs at horizon T.

Pilot data are **exploratory**: they are used only to estimate seed SD and configuration SD for the power analysis, to check that modes are amplified as logged, and to measure throughput. Pilot configurations are excluded from the confirmatory set. Three seeds are never presented as significance.

---

## 14. Confirmatory experiment

| Design | Matched configs ($\overline F=\overline F^\*$) | Aux. unmatched configs | Seeds/config | Runs (0.5B) | 1.5B replication |
|---|---|---|---|---|---|
| Minimum | 12 | 0 | 2 | 24 | none |
| **Standard (target)** | 24 | 8 ($\overline F$ ∈ {0.5×, 1.5×} of $\overline F^\*$, 4 each) | 2 | 64 | 8 configs × 1 seed |
| Extended | 36 | 12 | 2 | 96 | 12 × 2, + β>0 ablation (8 runs) |

- **Independent unit = verifier configuration.** Seeds are nested replicates. Checkpoints within a run are never independent observations.
- **Matched arm** tests whether $V$ predicts anything when aggregate rates are equal by construction.
- **Auxiliary arm** is needed to evaluate "beyond FPR / J". In the matched arm those predictors are constants, so "after controlling for FPR" is trivially satisfied there and must be tested where FPR varies.
- **Configurations are generated once**, from `q_ref.lock`, with a fixed generator seed, *before* any confirmatory run. They are split 2:1 into *training-configs* and **held-out configs** by stratified random assignment over the $(V,f_{\max})$ grid. Held-out configs are used only for the out-of-configuration evaluation in Figure 7.
- **Training budget** is identical across runs: T steps (from the pilot), fixed prompts per step, fixed G, fixed eval schedule.

**Primary outcome (one):** $Y_{r,s}$ = logit of the mean held-out true accuracy (greedy decoding, 1,000 eval prompts) averaged over the evaluation points in the window $[T/2, T]$. This is one scalar per run. A window mean was chosen over the final step alone (lower variance, robust to one bad eval) and over the peak step (unstable; see Prop. 4).

**Secondary outcomes:**
- final accuracy at T;
- integrated divergence $\int_0^T\Gamma(t)\,dt$;
- per-mode frequency trajectories $q_k(t)$ (from the eval-mode census every 50 steps);
- growth of the most-accepted mode, $\log q_{\arg\max f}(T)/q_{\arg\max f}(0)$;
- hacking indicator $H$ = 1[ $\max_{t\le T}A(t)-A(T)\ge 0.05$ and $\bar R(T)\ge\bar R(t^\*)$ ] (threshold = 2.5 × the pilot's eval SE, recomputed and locked after the pilot);
- peak-accuracy step (descriptive only);
- training instability (grad-norm spikes, NaN);
- $\mathrm{KL}(q(T)\|q(0))$;
- $\mathsf U$ fraction (mode creation).

---

## 15. Statistical analysis (pre-registration skeleton)

Pre-register on OSF (free) after the pilot and before the first confirmatory run. The exact text lives in `docs/PREREGISTRATION.md`.

**Hypotheses (two-sided, α = 0.05, Holm-adjusted over H1–H3):**
- **H1.** In the matched arm, config-mean outcome $\bar Y_r$ (mean over seeds) is associated with $V_r$ controlling for $f_{\max,r}$: OLS $\bar Y_r=\beta_0+\beta_V V_r+\beta_f f_{\max,r}+\varepsilon_r$ with HC3 standard errors. One row per configuration, so there is no pseudoreplication.
- **H2.** Out-of-configuration prediction: on held-out configs, the model {$f_{\max}$, $V$} fitted on training-configs has lower MSE than {$f_{\max}$} alone. Test: paired difference in squared error across held-out configs with a 10,000-resample config-level bootstrap CI. Also reported via leave-one-config-out CV on all configs.
- **H3.** The one-parameter CGF plug-in $Z(s)=\tau s-K(s)$ ($s$ fitted by CV on training-configs) has lower held-out MSE than the {$f_{\max}$, $V$} model.

**Secondary, all arms:** a hierarchical model $Y_{r,s}=\beta^\top X_r+u_r+\varepsilon_{r,s}$ ($u_r$ random config intercept) with predictors $\overline F$, $\tau$ (if varied), $J$, $J_{\text{worst}}$, $f_{\max}$, $\pi_{\text{trig}}$, $V$, $\kappa_3$, $V_{q^2}$, and the Egashira-style per-mode trigger-frequency × conditional-advantage aggregates (§5.3), computed from base-policy rollouts with oracle reward before training. Compare models by leave-one-config-out MSE, Spearman ρ and calibration slope. Report every predictor, not only the winners.

**Reporting:** effect sizes with 95 % CIs; ICC (config vs seed variance); the full prediction-error table. Exploratory analyses are labelled as exploratory.

**Power:** done by simulation after the pilot. Use the pilot seed SD $\hat\sigma_s$, the config-level residual SD $\hat\sigma_r$, and an effect size taken from the shared-policy simulation scaled to the pilot's observed time scale. Simulate the H1 OLS with R configs × 2 seeds. Choose R ∈ {12, 24, 36} as the smallest giving ≥ 80 % power. If even R = 36 gives < 50 % power, the confirmatory claim is downgraded to "estimation with CIs" and stated as such *in advance*. As a rough guide, with configs as units, a slope test needs SE ≈ $\sigma_{\bar Y}/(\mathrm{sd}(V)\sqrt{R-3})$.

**Missing data and failures:**
- *Infrastructure failure* (node crash, OOM, preemption): resume from the last checkpoint; same seed; at most 2 retries.
- *Training divergence* (NaN loss, or accuracy < 1 % for 3 consecutive evals): this is an **outcome**, not missing. The run is kept, with $Y$ computed from the evaluations that exist; divergences are counted per arm.
- A config with < 2 valid seeds uses its available seed. A sensitivity analysis drops such configs.
- No run is excluded for "looking wrong".

**Checkpoint selection:** none. Outcomes come from the pre-registered window, never from a best-checkpoint pick.

**Stopping rule:** the confirmatory grid runs to completion. There are no interim efficacy looks. A single futility look is allowed only for infrastructure feasibility (time), not for results.

---

## 16. HPC architecture (SIT cluster)

### 16.1 Isolation from ErrorEcho

```
~/VERIFLOW/                     # project root (never write outside it)
  .venv/                        # uv-managed, Python 3.11
  .cache/{hf,pip,uv,triton,torch,vllm}/
  data/  configs/  runs/  manifests/  logs/slurm/  checkpoints/
```

Every job script starts with:

```bash
set -euo pipefail
export VERIFLOW_ROOT="$HOME/VERIFLOW"
case "$(pwd -P)" in "$VERIFLOW_ROOT"*) ;; *) echo "refusing: not in VERIFLOW"; exit 2;; esac
export HF_HOME="$VERIFLOW_ROOT/.cache/hf"  HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$VERIFLOW_ROOT/.cache/uv"  PIP_CACHE_DIR="$VERIFLOW_ROOT/.cache/pip"
export TRITON_CACHE_DIR="$VERIFLOW_ROOT/.cache/triton"  TORCH_HOME="$VERIFLOW_ROOT/.cache/torch"
export VLLM_CACHE_ROOT="$VERIFLOW_ROOT/.cache/vllm"  TMPDIR="${SLURM_TMPDIR:-$VERIFLOW_ROOT/.tmp}"
source "$VERIFLOW_ROOT/.venv/bin/activate"
```

No script references `~/ErrorEcho-phase11-20`; this is checked by a `tests/test_isolation.py` grep. Job names all start with `veriflow-`. ErrorEcho jobs keep priority: before submitting, check `squeue -u $USER`. If an ErrorEcho job is pending or running on `sit-gpu`, VERIFLOW GPU jobs wait (`--dependency=afterany:<id>`), unless the node demonstrably has spare GPUs.

### 16.2 Step 0: read the scheduler policy (login node, read-only, no allocation)

```bash
sinfo -p gpu-nodes -o "%P %a %l %D %t %N %G %c %m"
scontrol show partition gpu-nodes
scontrol show node sit-gpu | egrep "Gres|CfgTRES|AllocTRES|RealMemory|CPUTot|State"
sacctmgr -nP show assoc user=$USER format=account,partition,qos,grptres,maxtres,maxwall,maxjobs
sacctmgr -nP show qos format=name,maxwall,maxtrespu,maxjobspu,maxsubmitpu,priority
squeue -p gpu-nodes -o "%.10i %.24j %.8u %.2t %.10M %.10l %.6D %b %R"
squeue -u $USER
cat /etc/slurm/slurm.conf 2>/dev/null | egrep -i "MaxTime|DefMemPer|MaxMemPer|PreemptMode|OverSubscribe" || true
```

Record the answers in `docs/CLUSTER_POLICY.md`:
- max walltime;
- GPUs on `sit-gpu` and whether they are shared or MIG;
- memory and CPU limits;
- preemption mode;
- whether compute nodes have internet (determines offline model download);
- the actual `--gres` string (`gpu:1` vs `gpu:l40:1`).

**Every resource number in the sbatch templates below is a placeholder until this file exists.**

### 16.3 Job ladder

| # | Job (`veriflow-*`) | Where | Purpose | Requests (placeholders) |
|---|---|---|---|---|
| 1 | `veriflow-preflight` | CPU (srun on gpu-nodes w/o GPU, or login if allowed) | env builds, imports, unit tests, simulation tests | 4 CPU, 16G, 30 min |
| 2 | `veriflow-gpucheck` | GPU | `nvidia-smi`, driver/CUDA, torch.cuda, bf16, flash-attn/vLLM import, 50-token generate | 1 GPU, 8 CPU, 32G, 15 min |
| 3 | `veriflow-smoke` | GPU | 1-step and 10-step GRPO with logging, checkpoint save + resume, verifier log written | 1 GPU, 8 CPU, 64G, 30 min |
| 4 | `veriflow-census` | GPU | §11 census for each $n$ | 1 GPU, 8 CPU, 64G, 1–2 h |
| 5 | `veriflow-bench` | GPU | throughput: steps/s for {HF generate, vLLM colocate} × G ∈ {8,16} × batch | 1 GPU, 8 CPU, 64G, 1 h |
| 6 | `veriflow-pilot` | GPU array | §13 | 1 GPU each, `--array=0-N%1` |
| 7 | `veriflow-grid` | GPU array | §14 | `--array=...%1` (or %k if k GPUs) |
| 8 | `veriflow-eval` | GPU | (re)evaluate checkpoints on eval sets | 1 GPU, 1 h |
| 9 | `veriflow-stats` | CPU | aggregation, bootstrap, figures | 8 CPU, 32G, 1 h |

### 16.4 `slurm/grpo_pilot.sbatch` (template)

```bash
#!/bin/bash
#SBATCH --job-name=veriflow-pilot
#SBATCH --partition=gpu-nodes
#SBATCH --nodelist=sit-gpu
#SBATCH --gres=gpu:1                 # confirm exact gres string in CLUSTER_POLICY.md
#SBATCH --cpus-per-task=8            # placeholder
#SBATCH --mem=64G                    # placeholder
#SBATCH --time=04:00:00              # placeholder ≤ MaxTime; set to 1.5× benchmarked run time
#SBATCH --array=0-11%1               # 12 pilot runs, one at a time
#SBATCH --signal=B:USR1@600          # 10 min warning → checkpoint
#SBATCH --requeue
#SBATCH --output=%x_%A_%a.out        # relative to submit dir = ~/VERIFLOW/logs/slurm
#SBATCH --error=%x_%A_%a.err
source "$HOME/VERIFLOW/slurm/_common.sh"          # the isolation block from §16.1
RUN_SPEC=$(python -m scripts.run_spec --table configs/pilot_runs.csv --index "$SLURM_ARRAY_TASK_ID")
RUN_DIR="$VERIFLOW_ROOT/runs/$(python -m scripts.run_id "$RUN_SPEC")"
mkdir -p "$RUN_DIR"
python -m scripts.write_manifest --spec "$RUN_SPEC" --out "$RUN_DIR/manifest.json"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 30 > "$RUN_DIR/gpu_monitor.csv" &
MON=$!
trap 'kill -USR1 $TRAIN_PID 2>/dev/null; wait $TRAIN_PID' USR1
python -m training.train_grpo --spec "$RUN_SPEC" --run-dir "$RUN_DIR" --resume auto &
TRAIN_PID=$!
wait $TRAIN_PID
STATUS=$?
kill $MON || true
python -m scripts.finalize_manifest --run-dir "$RUN_DIR" --exit-code $STATUS --slurm-job "$SLURM_JOB_ID"
exit $STATUS
```

**Checkpoint/resume:**
- `train_grpo.py` saves every 50 steps and on SIGUSR1: model, optimizer, scheduler, RNG states, trainer state, and verifier-stream counter (the stream is keyed by step, so it resumes exactly).
- `--resume auto` picks the latest complete checkpoint, marked by a `COMPLETE` sentinel file written last.
- Keep the last 2 checkpoints plus the step-T checkpoint. Delete the rest at run end to save disk; 0.5B full checkpoints with optimizer state are ≈6 GB each [TO MEASURE].
- A failed array task is resubmitted with `sbatch --array=<idx> slurm/grpo_pilot.sbatch`.

**Monitoring commands:**
- `squeue -u $USER -o "%.10i %.20j %.2t %.10M %R"`
- `sacct -j <id> --format=JobID,JobName%24,Elapsed,State,MaxRSS,AllocTRES%40`
- `seff <id>`
- inside a running allocation: `srun --jobid <id> --overlap nvidia-smi`
- `tail -f logs/slurm/veriflow-pilot_<A>_<a>.out`

### 16.5 Dependencies

Verified on PyPI/GitHub on 2026-10-31:
- **TRL 1.17.1** was released on 2026-10-30, and 1.16.0 on 2026-10-15.
- TRL 1.17.1 requires:
  - Python ≥3.11
  - torch ≥2.6
  - transformers ≥4.56.2
  - accelerate ≥1.11
  - datasets ≥4.7
  - extras: peft ≥0.17 and vllm ≥0.22,<0.33

Plan:
- **Python 3.11 or 3.12**, chosen in `veriflow-gpucheck` against the node's driver.
- **Pin TRL 1.17.x** at its latest patch on lock day. If a 1.17.2 has appeared by Week 4, prefer it. Release-day versions get one patch cycle before being frozen.
- Let `uv lock` resolve torch, transformers and vllm within TRL's ranges, then freeze. **Do not reuse ETH's or ErrorEcho's environment.**

GRPOConfig defaults in TRL 1.17.1 that matter to the theory are **pinned explicitly** in `configs/base.yaml`, never left to defaults:

| Setting | Value | Reason |
|---|---|---|
| `loss_type` | `"dapo"` | Same as Egashira |
| `beta` | `0.0` | Primary; §7 B5 |
| `scale_rewards` | `"group"` | |
| `num_iterations` | `1` | Clipping inactive (B4) |
| `num_generations` | `8` | |
| `temperature` | `1.0` | |
| `epsilon` | `0.2` | |
| `vllm_importance_sampling_correction` | `True` | Default; logged |
| `mask_truncated_completions` | `False` | |
| `log_completions` | `True` | |

Generation backend:
- `use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization≈0.25–0.3`. Server mode needs a second GPU and is therefore excluded.
- Fallback is `use_vllm=False`, chosen once in Gate 4 and identical for every run.
- **Driver check:** current vLLM wheels need a recent CUDA 12.x driver. If `nvidia-smi` on `sit-gpu` reports an older driver, use the HF-generate fallback rather than fighting the stack.

Full fine-tuning, not LoRA:
- 0.5B fits easily (≈ 8 GB for weights, grads and Adam states); 1.5B ≈ 24 GB with gradient checkpointing.
- LoRA changes the update geometry (§7 B8), so it would add a confound.

Optimiser: AdamW. Egashira used Adafactor with lr 5e-6. The LR is chosen by the pilot (§13).

Process: `uv lock` on the login node, producing `uv.lock` (committed). Install with `uv sync --frozen` inside `veriflow-preflight`. Download models once with `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --revision <sha> --cache-dir $HF_HOME` on whichever node has internet, then train offline. Record `nvidia-smi`, `nvcc --version` (if present) and `torch.version.cuda` in the manifest. The driver must support the wheel's CUDA runtime (check in `veriflow-gpucheck`).

---

## 17. Compute budget

Everything here is an **estimate to be replaced by `veriflow-bench` measurements**. The earlier 135 GPU-h figure is unverified and is not used.

Assumptions for Qwen2.5-0.5B, full fine-tune, bf16, L40 48 GB:
- 32 prompts/step × G=8 = 256 completions/step;
- prompt ≈ 60 tokens, completion ≤ 128 tokens.

| Component | Estimate (HF generate) | Estimate (vLLM colocate) |
|---|---|---|
| Model load + vLLM init | 1 min | 2–3 min |
| Rollout per step | 6–15 s | 1.5–4 s |
| Fwd/bwd + optimizer per step | 1–3 s | 1–3 s |
| Step total | 8–18 s | 3–7 s |
| Eval (1,000 greedy) every 50 steps | 40–90 s | 10–20 s |
| **Run, T=500 steps** | **1.3–2.7 h** | **0.5–1.1 h** |

| Phase | Runs | GPU-h (vLLM) | GPU-h (HF) |
|---|---|---|---|
| gpucheck + smoke + bench | — | 3 | 3 |
| Census (6 values of $n$) | — | 1–2 | 2–4 |
| LR pilot (4 runs × 800 steps) | 4 | 3–6 | 8–17 |
| Heterogeneity pilot | 12 | 6–13 | 16–32 |
| Minimum grid | 24 | 12–26 | 31–65 |
| Standard grid (0.5B) | 64 | 32–70 | 83–173 |
| 1.5B replication (≈2.5× cost) | 8 | 10–22 | 26–54 |
| Extended grid | 96 + 24 + 8 | 75–160 | 200–400 |
| **Minimum total** | | **≈ 25–50** | **≈ 60–120** |
| **Standard total** | | **≈ 55–115** | **≈ 140–285** |

Decision rule:
- If the measured cost per run × standard runs exceeds the realistic GPU time available by mid-January (assume ≈ 8 h/night × 4 nights/week until confirmed), drop to the **minimum** design.
- Configurations are cut before seeds (keep 2 seeds). The auxiliary arm is cut to 4 configs before the matched arm is cut.
- Never drop the held-out configs, the homogeneous control, or the de-confounding grid.

---

## 18. Repository structure

```
VERIFLOW/
├── README.md                  project summary, how to reproduce every figure
├── pyproject.toml / uv.lock   pinned deps (Python 3.11)
├── configs/
│   ├── base.yaml              shared training config (frozen after pilot)
│   ├── model/qwen0p5b.yaml, qwen1p5b.yaml   repo id + revision sha
│   ├── task/addition.yaml     n, template, split seeds
│   ├── verifiers/             generated config JSONs (one per config_id) + index.csv
│   ├── pilot_runs.csv, grid_runs.csv        run tables (config_id, seed, model)
│   └── locked/                q_ref.lock, prompt.lock, taxonomy.lock, prereg.lock
├── data/                      generated splits (parquet) + hashes; never edited by hand
├── docs/                      this plan, DECISIONS.md, CLUSTER_POLICY.md, PREREGISTRATION.md, preliminary/
├── literature/                notes per paper, source_docs/, bib/veriflow.bib
├── models/loading.py          pinned model/tokenizer loading, chat template, revision check
├── datasets/
│   ├── arithmetic.py          generator: operands, splits, dedupe, template
│   └── splits.py              split hashing + leakage checks
├── error_modes/
│   ├── classify_errors.py     taxonomy v1 hypothesis matcher (multi-hot + primary)
│   ├── parse_answer.py        `Answer:` extraction
│   └── TAXONOMY_VERSION
├── verifiers/
│   ├── noisy_verifier.py      decision rule + Philox stream + logging
│   ├── matched_verifiers.py   constrained generator, de-confounding grid, permutations
│   └── stats.py               FPR, V, kappa3, CGF, J, trigger … (single source for sims + analysis)
├── training/
│   ├── train_grpo.py          TRL GRPOTrainer wrapper, reward fn → verifier, callbacks
│   ├── callbacks.py           eval schedule, mode census, SIGUSR1 checkpoint, logging
│   └── reward_fn.py
├── simulations/
│   ├── sim_replicator.py      Gate 1
│   ├── sim_grpo_tabular.py    finite-G GRPO (B1–B3, B7)
│   └── sim_shared_policy.py   Gate 2
├── experiments/
│   ├── base_mode_census.py    Gate 3
│   └── throughput_bench.py    Gate 4
├── evaluation/
│   ├── evaluate_true_accuracy.py   held-out greedy + sampled accuracy per checkpoint
│   └── mode_trajectories.py        q_k(t) from eval-mode set
├── statistics/
│   ├── predictor_comparison.py     H1–H3, LOCO-CV, bootstrap, Holm
│   ├── power_sim.py
│   └── mixed_models.py
├── analysis/                  notebooks are thin; logic lives in statistics/
├── figures/make_fig{1..7}.py  each reads only from runs/ + manifests/
├── slurm/_common.sh, preflight.sbatch, gpucheck.sbatch, smoke.sbatch, census.sbatch,
│         bench.sbatch, grpo_pilot.sbatch, grpo_grid.sbatch, eval.sbatch, stats.sbatch
├── scripts/run_spec.py, run_id.py, write_manifest.py, finalize_manifest.py, check_isolation.sh
├── tests/                     test_sim_replicator.py, test_classifier.py, test_verifiers.py,
│                              test_matching.py, test_splits.py, test_manifest.py, test_isolation.py
├── paper/                     LaTeX, figures symlinked from figures/out
├── manifests/                 one JSON per run (copied from runs/ at finalize) + index.parquet
└── runs/                      per-run dirs: manifest, logs, verifier_log, evals, ckpts (git-ignored)
```

Note: a top-level package named `statistics/` shadows Python's stdlib `statistics` module. Name the package `vf_stats/` (and `vf_datasets/` instead of `datasets/`, which shadows Hugging Face `datasets`). Keep the user-facing names in docs, but the actual import names must avoid shadowing.

---

## 19. Exact coding roadmap

Order of implementation; each step has its tests passing before the next starts.

1. `verifiers/stats.py` + `tests/test_stats.py` (CPU)
2. `simulations/sim_replicator.py` + tests → **Gate 1**
3. `verifiers/matched_verifiers.py` + `tests/test_matching.py`
4. `simulations/sim_grpo_tabular.py`, `simulations/sim_shared_policy.py` → **Gate 2**
5. `error_modes/parse_answer.py`, `classify_errors.py` + synthetic recovery tests
6. `datasets/arithmetic.py`, `splits.py` + leakage tests
7. `verifiers/noisy_verifier.py` + tests
8. `models/loading.py`, `scripts/*manifest*`, `slurm/_common.sh`, `preflight.sbatch`, `gpucheck.sbatch`
9. `experiments/base_mode_census.py` → **Gate 3**
10. `training/train_grpo.py`, `reward_fn.py`, `callbacks.py`; `smoke.sbatch`; `throughput_bench.py` → **Gate 4**
11. `evaluation/*`; pilot → **Gate 5**
12. `statistics/power_sim.py` → pre-registration → grid
13. `statistics/predictor_comparison.py`, `figures/*` → paper

### 19.1 First implementation milestone (CPU only, no GPU)

- **First file:** `simulations/sim_replicator.py`, with `verifiers/stats.py` as its dependency.
- **First function:** `closed_form_accuracy(A0, q0, tau, f, t)`, which returns $\sigma(\operatorname{logit}A_0+\tau t-\log\sum_k q_{0,k}e^{f_k t})$. Compute it with `scipy.special.logsumexp` for stability at large $t$.

**Tests** (`tests/test_sim_replicator.py`):
1. Agreement with an RK4 / `solve_ivp` replicator integration to ≤1e-10 on 100 random $(q_0,f,\tau,A_0)$.
2. $V=0$ gives $\operatorname{logit}A$ exactly linear with slope $J$.
3. Finite-difference $dF/dt|_0 = V$ to 1e-6.
4. $f_{\max}>\tau$ gives argmax $A(t)$ equal to the root of $K'(t)=\tau$ to 1e-6, and the limit $A\to0$.
5. $f_{\max}<\tau$ gives $A\to1$.
6. Matched pairs give a logit difference of $-(V_B-V_A)t^2/2+O(t^3)$ (check that the ratio → 1 as $t\to0$).
7. Gibbs optimum (Prop. 6) vs the long-run KL-regularised ODE to 1e-8.
8. Property-based (`hypothesis`) invariance: permuting modes jointly in $q$ and $f$ leaves $A(t)$ unchanged.

**Expected numbers:** with the toy setting in `docs/preliminary/cgf_check.py` (seed 0, $M=5$, $\tau=0.9$, $\overline F=0.3$, $A_0=0.4$):
- $V_B=0.015785$, and the error is <1e-14;
- homogeneous verifier: $A(8)=0.988$;
- heterogeneous verifier: $A(8)=0.982$.

**Expected plots** (`figures/out/fig3a.pdf`, `fig3b.pdf`):
- (a) logit accuracy vs $\tilde t$ for 5 matched verifiers with increasing $V$; curves fan out quadratically, then by $f_{\max}$.
- (b) $F(t)$ vs $t$, with initial slopes equal to $V$.

**Definition of done:**
- all 8 tests pass in CI (`pytest -q`, < 30 s);
- figures regenerate from one command (`python -m figures.make_fig3`) with a `params.json` beside them;
- a one-page `docs/GATE1.md` states which statistic leads in which regime, with numbers, and lists what is proved vs simulated.

---

## 20. Slurm job design

Covered in §16.3–16.4. Additional rules:
- `grpo_grid.sbatch` is the pilot template with `configs/grid_runs.csv`. The array is throttled `%1` per available GPU.
- Order runs **seed-major, then config interleaved across the $(V,f_{\max})$ grid**, so that a time-cut grid is still balanced.
- `stats.sbatch` is CPU-only and never requests `--gres`.

---

## 21. Reproducibility strategy

`runs/<run_id>/manifest.json` (written at start, finalised at end; schema-tested):

```json
{
  "experiment_id": "veriflow-grid-v1", "run_id": "cfg017_s1_q05b",
  "git_commit": "...", "git_dirty": false,
  "model": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "<sha>", "tokenizer_revision": "<sha>"},
  "prompt_template_sha256": "...", "dataset_sha256": {"train": "...", "eval": "...", "eval_mode": "..."},
  "q_ref_sha256": "...", "verifier_config_sha256": "...", "verifier_config_id": "cfg017",
  "taxonomy_version": "v1", "seed": 1, "verifier_seed": 1017,
  "training": {"...every GRPOConfig field, dumped from the object, not from the YAML...": null},
  "env": {"python": "3.11.x", "packages": {"torch": "...", "trl": "...", "transformers": "...", "vllm": "..."},
          "cuda_runtime": "...", "driver": "...", "gpu": "NVIDIA L40", "hostname": "sit-gpu"},
  "slurm": {"job_id": "...", "array_task_id": "...", "job_name": "veriflow-grid"},
  "time": {"start": "...", "end": "...", "status": "completed|diverged|failed"},
  "checkpoints": [{"step": 500, "sha256": "...", "path": "..."}],
  "eval": {"window_mean_logit_acc": null, "final_acc": null, "file": "evals.parquet"}
}
```

Rules:
- `git_dirty: true` aborts confirmatory runs.
- Each figure script records the manifests it read (`figures/out/figN.sources.json`).
- At submission time, release code, configs, manifests, verifier logs, eval outputs and the step-T checkpoints for a subset (licence permitting; Qwen2.5 is Apache-2.0 [UNVERIFIED for the 0.5B card; check]) on Zenodo, which is free.

---

## 22. Figure plan

| Fig | X | Y | Unit | Estimator / CI | Message |
|---|---|---|---|---|---|
| 1 | — | — | — | schematic | Verifier = vector $f$ over error modes; matched $\overline F$ hides heterogeneity; CGF links $f$ to dynamics |
| 2 | mode | $f_k$ (bars), $q^{\text{ref}}_k$ (width) | config | exact | Matched configs: same $\overline F$ and $\tau$, different $V$, $f_{\max}$ |
| 3 | effective time $\tilde t$ | logit accuracy (a), $F(t)$ (b), regime map (c) | config (sim) | exact / sim mean ± 95 % over 20 families | Variance governs the short horizon, $f_{\max}$ the long; geometry (natural PG vs vanilla PG vs Adam) changes this |
| 4 | step | true accuracy and assigned reward | run (thin) + config mean (thick) | mean over seeds, 95 % bootstrap over seeds | Reward rises while accuracy diverges in high-$V$/high-$f_{\max}$ configs |
| 5 | step | $q_k(t)$ stacked | config | mean over seeds | Which modes the policy amplifies; whether the high-$f$ mode wins |
| 6 | predictor | held-out MSE (and Spearman) | config | config-bootstrap 95 % CI | **Does $V$ add beyond $f_{\max}$, $J$, trigger; does the CGF beat both** |
| 7 | predicted $\bar Y$ | observed $\bar Y$ on held-out configs | held-out config | 95 % CI per point (seed SE) + calibration line | Out-of-configuration generalisation |

**Strongest figure: Figure 6.** It answers the research question directly, whichever way the result goes.

---

## 23. Publication strategy

**Manuscript structure:**
1. Introduction: rate vs fate is known; shape is not.
2. Related work (§6 matrix, condensed).
3. Setup: modes, matched verifiers, reference distribution.
4. Theory: CGF law, regimes, geometry caveat; clearly labelled idealized.
5. Simulations: regime map, shared-policy results.
6. Experiments: census, matched grid, held-out prediction.
7. Results: Figs 4–7.
8. Limitations: synthetic verifier, arithmetic, ≤1.5B.
9. Reproducibility statement.

Appendix: proofs, full predictor tables, all runs including diverged ones.

**Venues** (dates checked 2026-10-31; re-check before submitting):

| Venue | Status | Fit |
|---|---|---|
| **ICLR 2027 workshops** | Main conference Apr 26–28 2027, workshops Apr 29–30. Accepted workshops announced **Nov 29 2026**; suggested paper deadline **≈ Feb 1 2027**, notifications by Feb 26 [VERIFIED on iclr.cc; individual workshop deadlines will vary] | **Theory + simulation paper (Gates 1–2 output)** fits this deadline. The GPU results will not be ready by Feb 1, so do not rush them. Pick a workshop on reasoning/RL/reliable ML once the list is public |
| **COLM 2027** | Not yet announced. COLM 2026 deadlines were Mar 26 (abstract) / Mar 31 (paper) [VERIFIED for 2026]; a similar 2027 date is UNVERIFIED | **Primary target for the full paper.** Egashira et al. appeared at COLM 2026, so it is the natural audience. Needs analysis done by mid-March, which is feasible with this calendar |
| **TMLR** | Rolling [VERIFIED] | Strong fallback for a careful null result; reviewers value rigour over novelty claims |
| ICML 2027 | Not announced; a late-January deadline is likely but UNVERIFIED | Too early for confirmatory data; skip |
| NeurIPS 2027 main / workshops | Not announced | Only if the result is strong and replicates at 1.5B; otherwise workshops |

No acceptance is promised. A pre-registered null at matched rates with a correct regime theory is publishable at TMLR or a workshop. Post an arXiv preprint after the confirmatory analysis, not before pre-registration.

## 24. International research fit

Programme details are as of 2026-10-31. **Treat every UNVERIFIED line as unknown and check the official page.**

| Group | Overlapping work | Exact overlap | Route for a 3rd-year Indian B.E. student | Eligibility / funding / deadline | Artifact to attach |
|---|---|---|---|---|---|
| **ETH Zurich SRI Lab** (Martin Vechev) | Egashira et al. 2026 (COLM) and `llm-verifier-noise` | VERIFLOW directly extends their systematic-error study to multi-mode matched verifiers and adds theory | **ETH D-INFK Student Summer Research Fellowship (SSRF)** [VERIFIED exists: inf.ethz.ch/studies/summer-research-fellowship.html; ≈20 students/yr, Jul–Aug] | Open to non-ETH undergraduates worldwide [VERIFIED]. 2027 deadline and stipend not posted [UNVERIFIED; historically Dec–Jan] | Gate-1/2 report: CGF regime theory + simulation figures + clean repo |
| **TU Darmstadt AIML** (Kristian Kersting) | Helff et al. 2026, *LLMs Gaming Verifiers* (authors at TU Darmstadt / hessian.AI / DFKI) [VERIFIED] | Real single-mode verifier gaming; VERIFLOW gives a controlled multi-mode counterpart | **DAAD WISE** (Germany, Indian B.E./B.Tech students) [UNVERIFIED: page unreachable; historically 2nd/3rd-year eligible, stipend, deadline ≈ Oct–Nov — **check today; it may already be closed**] | as left | Same + AGENTRUN benchmark |
| **EPFL** (e.g. Flammarion TML Lab [VERIFIED associate professor]; CLAIRE lab works on RL, head UNVERIFIED) | Optimisation/learning-dynamics theory (TML); RL for LLMs (CLAIRE) | Geometry dependence (Prop. 7) and GRPO dynamics | **Summer@EPFL** [VERIFIED: summer.epfl.ch] | 2nd-year bachelor and above in CS/CE/EE; CHF 1,800/month + travel; 2–3 months between May and Sep; **deadline 29 Nov 2026 (AoE)**; decisions before Jan 25 | Gate-1 note + Before the Recommendation (pre-registered, honest nulls) |
| **MPI for Intelligent Systems, Tübingen** | Co-author affiliation on Egashira et al. [VERIFIED in the paper] | Verifier error in RLVR | Internship pages blocked during the check [UNVERIFIED] | Check is.mpg.de/career manually | Same as ETH |
| **MBZUAI** | RL/LLM reasoning groups (identify specific faculty before contacting; not verified here) | — | **UGRIP** [UNVERIFIED: page returned 403; historically a winter programme with autumn applications] | Check mbzuai.ac.ae/ugrip manually | — |
| Mitacs Globalink | — | — | India is a partner, but the **Summer 2027 call is closed** [VERIFIED] | — | — |

Rules:
- Contact a faculty member only with a concrete, finished artifact: the Gate-1/2 report plus repo, by Week 3.
- Do not claim VERIFLOW results that do not exist yet.
- Describe AGENTRUN's CARE 2026 status and Before the Recommendation's submission status exactly as they stand.

---

## 25. 16-week execution calendar (Mon 2 Nov 2026 – Sun 21 Feb 2027)

Budget: 8–12 h/week. GPU time is scheduled. ErrorEcho has priority on the GPU. ≈2 h/week is reserved for applications from Week 3 on.

| Week (dates) | Objective | Coding | Math | Experiments / HPC | Deliverable | Go/no-go | Hours |
|---|---|---|---|---|---|---|---|
| 1 (Nov 2–8) | Gate 0 + Gate 1 start | repo skeleton, `stats.py`, `sim_replicator.py` | write up Props 1–7 with proofs | **HPC read-only**: §16.2 policy commands → `CLUSTER_POLICY.md` | literature notes for the 3 closest papers; Gate-1 tests green | Gate 0 decision recorded | 10 |
| 2 (Nov 9–15) | Gate 1 done; matched generator | `matched_verifiers.py`, tests; replicator sweeps + Fig 3a/b | Prop 7 derivation, Gibbs (Prop 6) check | create `~/VERIFLOW`, `uv lock`, `veriflow-preflight` (CPU) | Gate-1 report | Gate 1 | 10 |
| 3 (Nov 16–22) | Gate 2 | `sim_grpo_tabular.py`, `sim_shared_policy.py` | regime analysis (horizon × concentration) | CPU only | Gate-2 report + regime map | **Gate 2 (major)** | 11 |
| 4 (Nov 23–29) | Construct tooling + **Summer@EPFL application (deadline Nov 29 AoE)** | `parse_answer`, `classify_errors`, `arithmetic.py`, `noisy_verifier.py`; all tests | — | `veriflow-gpucheck`; model download (pinned) | tested taxonomy v1; EPFL application submitted (Gate-1/2 report attached) | — | 12 |
| 5 (Nov 30–Dec 6) | Gate 3 | `base_mode_census.py`, `models/loading.py`, manifests | power-relevant: how $q$ error propagates to $\overline F$ | `veriflow-census` for n=5..10 | census report; `q_ref.lock` | **Gate 3** | 10 |
| 6 (Dec 7–13) | Gate 4 | `train_grpo.py`, callbacks, resume | — | `veriflow-smoke`, `veriflow-bench` | measured step time; locked compute budget | **Gate 4** | 12 |
| 7 (Dec 14–20) | LR pilot | eval scripts, mode trajectories | — | 4 LR-pilot runs (overnight) | T and LR fixed | — | 9 |
| 8 (Dec 21–27) | Heterogeneity pilot | analysis scaffolding (exploratory) | — | 12 pilot runs | pilot report (exploratory) | **Gate 5** | 8 (holidays) |
| 9 (Dec 28–Jan 3) | Power + prereg | `power_sim.py` | power sims | — | `PREREGISTRATION.md` → OSF; configs generated + split | prereg locked | 8 |
| 10–12 (Jan 4–24) | Confirmatory grid (0.5B) | `predictor_comparison.py`, figures 4–5 on incoming data (blinded to H1–H3) | — | grid array, about 20–25 runs/week if ~1 h/run | runs complete, manifests clean | infra futility look only | 9/wk |
| 13 (Jan 25–31) | Finish grid; 1.5B replication; **optional ICLR-2027-workshop submission of the theory + simulation note (≈ Feb 1)** | — | — | remaining runs + 1.5B | all data frozen; workshop note (no GPU claims) | — | 11 |
| 14 (Feb 1–7) | Confirmatory analysis | Figs 6–7 | interpret vs theory regimes | `veriflow-stats` | results memo (incl. null) | — | 12 |
| 15 (Feb 8–14) | Writing | reproducibility package | appendix proofs | — | full draft | — | 12 |
| 16 (Feb 15–21) | Polish + release | Zenodo bundle | — | — | arXiv-ready draft; workshop target chosen | submit / hold | 10 |

If Gate 2 or Gate 3 fails, the calendar switches to the fallback in §27, and the remaining weeks produce a simulation-plus-theory note instead of a GPU study.

---

## 26. First 72-hour action plan

**First 2 hours (CPU, laptop):**
0. **Applications triage (15 min):** open the official DAAD WISE, ETH SSRF, MBZUAI UGRIP and MPI-IS pages. Record the 2027 deadlines in `docs/APPLICATIONS.md`. Summer@EPFL closes **29 Nov 2026**; DAAD WISE may already be closed.
1. Read §5 against the three closest papers' method sections (bring the PDFs into `literature/`). Write a single-paragraph Gate-0 verdict in `docs/DECISIONS.md`.
2. `git init` in `E:\Veriflow`, create the skeleton directories from §18, `pyproject.toml` (numpy, scipy, pandas, pyarrow, matplotlib, pytest, hypothesis).
3. Move `docs/preliminary/cgf_check.py` logic into `tests/test_sim_replicator.py::test_closed_form_matches_ode`.

**First 24 hours:**
4. Implement `verifiers/stats.py`: `aggregate_fpr`, `weighted_var`, `kappa3`, `cgf`, `cgf_prime`, `youden`, `fmax`, `trigger_freq`, `v_q2`.
5. Implement `simulations/sim_replicator.py::closed_form_accuracy` and `integrate` with tests for Props 1–4 and 6.
6. On the cluster, **read-only**: run the §16.2 commands and save the output to `docs/CLUSTER_POLICY.md`. Submit nothing.

**First 72 hours:**
7. `matched_verifiers.py` with exact-matching tests and the $(V, f_{\max})$ grid.
8. First replicator sweep → Figure 3a/b drafts and the regime map.
9. Gate 1 write-up (one page): which statistic leads in which regime, with numbers.
10. Draft emails/notes for application targets (§24); **send nothing yet**.

---

## 27. Explicit kill criteria

| Gate | Kill / transform condition | Consequence |
|---|---|---|
| 0 Novelty | A paper already runs matched-$\overline F$ and matched-$\tau$ configurations varying error-mode dispersion **and** compares predictors on held-out configurations | Stop the empirical claim. Pivot to replication + the geometry question (Prop 7) only if that is also unaddressed; otherwise drop VERIFLOW |
| 1 Math | Props 1–6 fail a numerical test | Fix or remove them; no theory claims until fixed |
| 2 Simulation | In shared-parameter Adam sims, $V$ adds no held-out information beyond $f_{\max}$ under any realistic coupling **and** the CGF also fails | The GPU study is not worth running as a $V$-test. Reframe as "aggregate and worst-case rates suffice" (a simulation note) or test $V_{q^2}$ if it survives (switch logged before GPU data) |
| 3 Construct | < 3 designed modes with ≥ 5 % mass for every $n$ and both operations; or ambiguity > 10 % that cannot be merged away | Stop the matched-heterogeneity design on this model |
| 4 Feasibility | Measured standard budget > 2× available GPU time **and** the minimum design has < 50 % power for the pilot-estimated effect | Run the minimum design as estimation-only (stated in the prereg), or stop |
| 5 Empirical (pilot) | Modes are not differentially amplified at all (all $q_k(t)$ flat; the homogeneous and high-$V$ configs are indistinguishable in $F(t)$), so the mechanism is not engaged within the horizon | Lengthen the horizon once (≤ 2× T). If it is still not engaged, stop: the manipulation does not reach the policy |

A **null in the confirmatory test is not a kill**. It is a result ("at matched aggregate rates, dispersion does not add predictive information in GRPO at this scale"), and is publishable as such with the simulation regime map.

---

## 28. Final locked-protocol checklist (all must be ✔ before the first confirmatory run)

- [ ] Gate 0–5 decisions written in `docs/DECISIONS.md` with dates
- [ ] `CLUSTER_POLICY.md` complete; sbatch resource numbers derived from it
- [ ] `uv.lock` committed; `veriflow-preflight` and `veriflow-gpucheck` green
- [ ] model + tokenizer revisions pinned (SHA) in `configs/model/*.yaml`
- [ ] prompt template hash, taxonomy v1, `q_ref.lock`, split hashes recorded
- [ ] matched configs generated, split into train/held-out, hashes in `configs/verifiers/index.csv`
- [ ] verifier tests (matching, realised rates, determinism, stream independence) green
- [ ] LR, T, G, batch, max length, β, loss_type, scale_rewards, μ=1, generation backend frozen in `configs/base.yaml`
- [ ] primary outcome, H1–H3, models, α, Holm, missing-data and divergence rules in `PREREGISTRATION.md`, registered on OSF
- [ ] power simulation done; R chosen; design tier (min/standard/extended) declared
- [ ] `git_dirty=false` guard active; manifests schema-tested
- [ ] ErrorEcho isolation test green; no shared paths, caches, job names
- [ ] figure scripts run end-to-end on pilot data (exploratory) before confirmatory data exist

---

PROJECT: VERIFLOW
SCIENTIFIC NOVELTY: REQUIRES MODIFICATION
MATHEMATICAL VALIDITY: ESTABLISHED UNDER SPECIFIED ASSUMPTIONS
HPC FEASIBILITY: MODERATE
FIRST IMPLEMENTATION FILE: simulations/sim_replicator.py (with verifiers/stats.py)
FIRST EXPERIMENT: CPU replicator sweep comparing V, f_max and the CGF predictor across matched verifier families, horizons and update geometries (Gate 1–2), with no GPU use
THE SINGLE NEXT ACTION: Implement closed_form_accuracy() in simulations/sim_replicator.py and its test against an RK4 replicator integration (≤1e-10), then run the read-only Slurm policy commands of §16.2 and save their output to docs/CLUSTER_POLICY.md
