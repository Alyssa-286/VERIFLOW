# VERIFLOW: one-page project summary (draft for applications, 2026-11-03)

> Draft for Stage A1 of COMPLETION_PLAN.md. Edit it into your own voice before
> sending. Every claim below is traceable to a file in this repository. Update
> the "Status" line when Gate 3′ is decided.

**Title:** *Same Error Rate, Different Fate? Error-Mode Structure of Verifier
False Positives and the Dynamics of RLVR*

**Question.** Reinforcement learning with verifiable rewards (RLVR/GRPO)
trains language models against automatic verifiers, and real verifiers make
mistakes. Egashira et al. (2026) showed that the *aggregate* false-acceptance
rate does not determine the outcome. This project asks: when two verifiers
have **exactly the same** aggregate false-positive and true-positive rates,
does the **way their false positives are spread across error modes** predict
how well the trained model ends up?

**Approach.**
- **Theory (done, Gate 1).** In an idealized replicator model of RL
  fine-tuning, accuracy has a closed form governed by the cumulant-generating
  function of per-mode false-acceptance rates:
  - variance V governs short horizons;
  - the worst mode governs long horizons;
  - under vanilla policy gradients the relevant dispersion becomes
    q²-weighted.

  Seven propositions are verified numerically, to 1e-10 where checkable
  (21 tests). These are known replicator/Gibbs results applied to a new
  object, and are stated as such.
- **Simulation (done, Gate 2).** Finite-group GRPO is simulated with
  stochastic verifiers, Adam, shared parameters and heterogeneous prompts.
  Adding V to the worst-mode rate improved out-of-configuration prediction
  in **27/27** setting × horizon cells (bootstrap CIs exclude 0). The
  one-parameter CGF predictor was best at longer horizons.
- **Instrument (done).** Programmatic verifiers matched *exactly* (to 1e-12)
  on aggregate rates while varying the shape of their errors. They use a
  replayable SHA-256 randomness design.
- **Empirical construct test (done, negative result).** Qwen2.5-0.5B-Instruct
  was run on an HPC cluster (Slurm, NVIDIA L40): 24 pre-specified censuses,
  8,000 generations each. They showed that the model's arithmetic errors are
  **not** explained by classic mechanisms (dropped carries, transpositions,
  borrows). 55–97 % of errors match none of them, for addition and for
  subtraction. The pre-registered stop rule fired, so the design was formally
  amended (Amendment 1): error modes are now defined by observable error shape,
  and the amended design is re-tested on fresh data.
- **Next (planned, pre-registered on OSF).** A 64-run GRPO experiment testing
  whether V predicts held-out accuracy among exactly matched verifiers, with
  a Qwen2.5-1.5B replication.

**Methods and engineering.**
- Python, PyTorch, Hugging Face Transformers.
- Slurm on a shared GPU cluster, with job isolation from other projects.
- Reproducibility: pinned model revisions verified by SHA-256, locked
  prompts and taxonomies, per-run manifests (git commit, job id, data
  hashes), cluster-bootstrap CIs.
- A decision log in which every protocol change is written down before the
  data it affects.
- Over 150 automated tests.

**Status (2026-11-03).**
- Theory and simulation gates are passed.
- The first empirical gate stopped as designed; that negative result is
  recorded.
- The amended construct gate (Gate 3′) is running on the cluster.
- Target: an arXiv preprint and code release in early 2027.

**What I learned.** Pre-registration and kill rules matter. The first real
data contradicted a plausible assumption: that small-model arithmetic errors
are mechanistic. The project's value comes from catching that cleanly
instead of tuning around it.

---

*Note for the applicant:* say plainly which parts you designed, implemented
and ran yourself. If asked, mention that you used AI coding assistance, as
most programmes expect that to be disclosed.
