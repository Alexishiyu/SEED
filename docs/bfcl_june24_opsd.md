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

The independent `june24_all200` contract is used only by the five-pass
train-160/validation-40 experiment. It reads the canonical 200-row
`pairwise_tasks.csv`, requires exact outcome counts (`fixed=40`,
`both_wrong=112`, `harmed=13`, `both_correct=35`), and freezes a seed-20260624
stratified split. The normalized runtime bank contains only the 160 training
summaries while its source manifest records and verifies all 200 source-call
paths and hashes. Validation summaries are never loaded into a prompt.
The historical source calls for tasks 56, 154, and 169 contain repair metadata.
For this all-200 experiment only, the builder accepts those exact task IDs via
three explicit `--allow-repaired-task-id` flags. Their source paths, hashes,
repair-metadata field names, and override status are persisted in the bank and
final evidence. Any other repaired record still fails preflight. The fixed-40
contract remains fully repair-intolerant.

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

## Five-pass 160/40 experiment

Build the all-200 cohort, deterministic split, and train-only bank:

```bash
python scripts/build_june24_all200_skill_bank.py \
  --pairwise-tasks-csv /drive/bfcl_qwen_pairwise_analysis/15fRBFq4gbXgJeQ5CHO_bVjeEp0mlH9rB/pairwise_tasks.csv \
  --source-dir /drive/bfcl_qwen_experiment/a100_skill_sd_50_20260624_055240/skills_openai \
  --source-dir /drive/bfcl_qwen_experiment/a100_skill_sd_150_50_199_20260624_063820/skills_openai \
  --cohort-manifest /run/inputs/june24_all200_cohort.json \
  --split-manifest /run/inputs/june24_train160_val40_split.json \
  --output /run/inputs/june24_train160_skill_bank.json \
  --allow-repaired-task-id multi_turn_base_56 \
  --allow-repaired-task-id multi_turn_base_154 \
  --allow-repaired-task-id multi_turn_base_169
```

Preflight or execute the logical five-iteration job:

```bash
python examples/seed_trainer/run_bfcl_opsd.py \
  --june24-skill-bank /run/inputs/june24_train160_skill_bank.json \
  --cohort-manifest /run/inputs/june24_all200_cohort.json \
  --split-manifest /run/inputs/june24_train160_val40_split.json \
  --run-root /content/drive/MyDrive/bfcl_qwen_experiment/seed_opsd_colab/june24_all200_train160_val40_<timestamp> \
  --bfcl-root /path/to/gorilla/berkeley-function-call-leaderboard \
  --iterations 5 --batch-size 4 --checkpoint-updates 10 --optimizer adam \
  --warmup-updates 67 --warmup-target-lr 1e-7 --final-lr 1e-6 \
  --inline-same-prompt-diagnostics --resume auto \
  --rlpaper-sha <sha>
```

This materializes 800 ordered training rows: 160 tasks once in each of five
iterations. It performs 40 updates per iteration, validates the ordinary BFCL
prompt at steps 0/40/80/120/160/200, and keeps resumable recovery checkpoints
every 10 updates, including the five nonzero validation steps. Strict Adam uses betas `(0.9, 0.999)`, epsilon
`1e-8`, and zero weight decay. The first 67 updates rise from zero to `1e-7`;
the remaining updates rise linearly to `1e-6`.

Each four-task batch is one Adam update even when BFCL flattens those task
trajectories into more than four active tool turns. The actor accumulates all
active generated-token losses with full-batch token-mean weighting before the
single optimizer step. Update evidence records exact task order, iteration and
batch coordinates, class-level metrics, token/mask hashes, LR, gradients, and
before/after LoRA hashes. Resume restores the model, Adam, scheduler, actor and
driver RNG state, and the stateful data position.

On a single A100, the launcher intentionally ends the Ray/vLLM process after
each 10-update recovery checkpoint and starts a clean process for the next
segment. This avoids vLLM sleep-pool accumulation and bounds the work lost to a
Colab runtime cutoff while leaving the 800-row order, five 40-update
iterations, validation boundaries, and 200-step learning-rate schedule
unchanged. Every new segment must advance by exactly 10 updates; historical
40-update segments remain valid migration evidence.
For LoRA runs, recovery restores the adapter from the checkpoint's
`lora_adapter/adapter_model.safetensors` and restores Adam, scheduler, RNG, and
dataset position from their ordinary checkpoint files. The frozen base model is
reloaded from the pinned Hugging Face model rather than reread from the roughly
18 GB DriveFS model shard. Because that shard duplicates only frozen parameters
and is not reliably durable on Colab DriveFS, LoRA-only checkpoints deliberately
omit it. The adapter, optimizer, scheduler/RNG, data position, pinned base-model
revision, and update-level checksum continuity are the authoritative resumability
evidence. The final merged Hugging Face export remains a separate required stage.
`metadata/privileged_june24_segment_history.json` records the checkpoint and
SEED SHA used for each segment. If per-update evidence extends beyond the last
atomic checkpoint after a runtime loss, the launcher archives those orphaned
files under `evidence/privileged_june24/recovery_archive/` before resuming.
Resumed segments skip already-recorded boundary validation, so validation
still occurs exactly at steps 0/40/80/120/160/200.

Inline control diagnostics rescore the exact same response tokens under the
ordinary prompt but never contribute that control signal to the gradient.
They are debugging evidence, not a separately trained causal control arm. Use
`examples/seed_trainer/bfcl_seed_opsd_160x40_a100.ipynb` as the thin A100
controller. It writes all inputs, validation curves, checkpoints, update-level
token/hash evidence, the final merged export, and the completion report under
one fresh Drive-backed run root.

## Five-epoch 128/72 large-batch experiment

The `128x72_b64` split profile is a separate deterministic experiment over the
same audited 200-task authority. It keeps the original 40 validation tasks and
adds the next 32 outcome-stratified SHA-ranked tasks to validation. The frozen
training counts are `fixed=26`, `both_wrong=72`, `harmed=8`, and
`both_correct=22`; validation counts are `14/40/5/13` in the same class order.

Each of five epochs contains exactly two 64-task batches. Every batch produces
one strict-Adam, zero-weight-decay update and an atomic resumable LoRA
checkpoint, for 640 training rollouts and 10 updates/checkpoints total. This
profile uses SEED's usual constant `1e-6` learning-rate behavior with no
warmup; all OPD-only, prompt, model, LoRA, source-audit, and inline-control
contracts remain unchanged. Validation covers all 72 held-out tasks at steps
`0, 2, 4, 6, 8, 10` (432 validation rollouts).

Build and preflight it with:

```bash
python scripts/build_june24_all200_skill_bank.py \
  --pairwise-tasks-csv /drive/bfcl_qwen_pairwise_analysis/pairwise_tasks.csv \
  --source-dir /drive/bfcl_qwen_experiment/a100_skill_sd_50_20260624_055240/skills_openai \
  --source-dir /drive/bfcl_qwen_experiment/a100_skill_sd_150_50_199_20260624_063820/skills_openai \
  --cohort-manifest /run/inputs/june24_all200_cohort.json \
  --split-manifest /run/inputs/june24_train128_val72_b64_split.json \
  --output /run/inputs/june24_train128_skill_bank.json \
  --split-profile 128x72_b64 --batch-size 64 \
  --allow-repaired-task-id multi_turn_base_56 \
  --allow-repaired-task-id multi_turn_base_154 \
  --allow-repaired-task-id multi_turn_base_169

python examples/seed_trainer/run_bfcl_opsd.py \
  --june24-skill-bank /run/inputs/june24_train128_skill_bank.json \
  --cohort-manifest /run/inputs/june24_all200_cohort.json \
  --split-manifest /run/inputs/june24_train128_val72_b64_split.json \
  --run-root /content/drive/MyDrive/bfcl_qwen_experiment/seed_opsd_colab/june24_all200_train128_val72_b64_<timestamp> \
  --bfcl-root /path/to/gorilla/berkeley-function-call-leaderboard \
  --iterations 5 --batch-size 64 --checkpoint-updates 1 \
  --optimizer adam --lr-schedule constant --final-lr 1e-6 \
  --inline-same-prompt-diagnostics --resume auto --rlpaper-sha <sha>
```

Use `--execute --stop-after-update 1` for the Drive-backed batch-64 gate. After
checkpoint 1 and its update evidence validate, rerun with `--execute` and no
stop flag to resume the same optimizer, scheduler, RNG, dataset position, and
adapter through checkpoint 10. The paired controller is
`examples/seed_trainer/bfcl_seed_opsd_128x72_b64_a100.ipynb`.
Its first cell fails closed unless the canonical Drive experiment root is
writable. When the established `/content/drive2` DriveFS mount is healthy but
`/content/drive` is not, it bind-mounts that healthy filesystem at the canonical
path; otherwise a fresh mount has a 120-second timeout. A write/read/delete
probe must pass before setup or run-root creation.
