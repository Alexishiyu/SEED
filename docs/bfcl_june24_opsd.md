# BFCL OPD-only with June 24 Skill-SD summaries

This path trains SEED on official multi-turn BFCL interactions while using a
frozen historical Skill-SD summary as training-only privileged context. It is
separate from the self-evolving analyzer recipes and contains no V2R input.

## Input contract

The bank builder reads `skill_summary_call_<task_id>.json` from exactly these
historical directories:

- `a100_skill_sd_50_20260624_055240/skills_openai/`
- `a100_skill_sd_150_50_199_20260624_063820/skills_openai/`

Only `success_analysis`, `mistake_analysis`, and `golden_workflow` are copied.
Every normalized record stores the source path and SHA-256. The loader
recomputes those hashes before rollout and rejects possible-answer, predicted
tool-call, predicted-state, structured-reference, ground-truth-target, and V2R
fields.

The fixed-manifest authority must contain exactly 40 unique rows classified as
`fixed`, with `baseline_valid=False` and `skill_valid=True`. It must also prove
that this cohort is a strict subset of the rejected 75-task all-teacher-success
cohort and explicitly forbid reuse of that cohort.

## Agent and loss contract

- `env.env_name=bfcl` resets the exact official task, parses and executes
  official calls, retains multi-call turns, and returns simulator observations,
  validity, termination, and the official multi-turn verifier result.
- The official Qwen prompt is passed through without applying a second chat
  template. Tool results occur only in a subsequent prompt. Each active SEED
  row contains the exact sampled model response tensor for that step.
- `algorithm.seed.analysis_backend=june24_skill_summary` applies one frozen
  task summary to every active step of its trajectory. No analyzer request,
  hindsight generation, fallback, or repair is allowed.
- `actor_rollout_ref.actor.opd_only=True` uses SEED's unchanged OPD formula and
  excludes policy-gradient, reward, entropy, KL, critic, reference-policy,
  reward-model, skill-generation, and environment-auxiliary loss gradients.
  Teacher log-probabilities are computed by the colocated pre-update actor, not
  a separate teacher pool.

## Build and preflight

```bash
python scripts/build_june24_skill_bank.py \
  --fixed-csv /path/to/fixed_tasks.csv \
  --rejected-75-manifest /path/to/teacher_success_selection.json \
  --fixed-manifest /run/inputs/june24_fixed40_manifest.json \
  --source-dir /drive/a100_skill_sd_50_20260624_055240/skills_openai \
  --source-dir /drive/a100_skill_sd_150_50_199_20260624_063820/skills_openai \
  --output /run/inputs/june24_fixed4_skill_bank.json \
  --task-ids multi_turn_base_0,multi_turn_base_1,multi_turn_base_2,multi_turn_base_3

python examples/seed_trainer/run_bfcl_opsd.py \
  --june24-skill-bank /run/inputs/june24_fixed4_skill_bank.json \
  --fixed-manifest /run/inputs/june24_fixed40_manifest.json \
  --run-root /run \
  --bfcl-root /path/to/gorilla/berkeley-function-call-leaderboard \
  --rlpaper-sha <sha>
```

Omitting `--task-ids` selects the first four ordered fixed-40 IDs. Add
`--execute` only after preflight. Add `--same-prompt-control` for the paired
control arm. The launcher intentionally has no V2R argument.

## Colab smoke

Use `examples/seed_trainer/bfcl_seed_opsd_a100.ipynb`. It requires exactly one
A100, resolves and checks out the pushed `codex/bfcl-opsd` commit, pins BFCL,
materializes the four-task bank directly from the canonical Drive sources, runs
plain-prompt BFCL evaluation before and after training, trains both arms, merges
the privileged LoRA, verifies vLLM reload, and writes a single completion report
under the Drive-backed run root.
