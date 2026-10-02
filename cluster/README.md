# Running on the hessian.AI 42 cluster

Workspace **`IL_rm_bias`** (id 142) · PFSS root **`/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias`**

Our workload is **inference only**: forward passes, last-layer activation extraction, and
linear algebra. No training, no checkpointing, no hyperparameter search — so none of
Determined's trial machinery is needed. A task with a plain `entrypoint` just runs our CLI.

---

## 0. Connect

TU VPN up first. `det` 0.35.0 is installed in the Mac's system Python 3.14
(`/Library/Frameworks/Python.framework/Versions/3.14/bin/det`), so no venv needs activating:

```bash
export DET_MASTER=https://login01.ai.tu-darmstadt.de:8080
det user login ck21zoly
det workspace ls
```

`export` holds for one terminal only: repeat it in every new terminal, or add the line to `~/.zshrc`.
Paste command lines without trailing `# ...` comments into the Mac terminal; zsh without
`interactivecomments` passes them to the command as extra arguments.

The configs are filled in for this workspace and TU-ID; nothing needs editing before a first
launch. `<REGISTRY>` appears only in the optional custom-image route below.

## 1. Get our packages into the environment

The onboarding slides show `py-3.8-pytorch-1.12`, which would have forced a custom image. That
is **stale**: the cluster's own `config_blueprint.yaml` ships
**`determinedai/pytorch-ngc:0.35.0`**, an NGC PyTorch build with a modern torch and the
matching Determined harness. So the base is fine and only our extras (transformers,
scikit-learn, concept-erasure, textstat) need adding. Two routes:

**(b) PFSS install — no Docker, fastest to a first result.** From a staging shell:

```bash
PFSS=/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias

# --retries 1: the image's pip.conf lists pypi.ngc.nvidia.com, which the compute nodes
# cannot resolve, so every package otherwise burns 5 retries before falling back to PyPI.
# A CLI flag cannot unset an extra-index-url, so make the failure fast instead.
python -m pip install --target "$PFSS/pylibs" --retries 1 --timeout 10 \
  "transformers>=4.56,<5" accelerate datasets textstat scikit-learn concept-erasure \
  hf_transfer

# MANDATORY cleanup -- see below.
cd "$PFSS/pylibs" && rm -rf \
  torch torchgen functorch torch-*.dist-info torch.libs \
  numpy numpy.libs numpy-*.dist-info \
  nvidia nvidia_*.dist-info triton triton-*.dist-info \
  cuda_bindings cuda_pathfinder cuda_toolkit cuda_*.dist-info
```

`config.yaml` already puts `$PFSS/pylibs` on `PYTHONPATH`. Nothing is installed on the
compute node itself — the packages sit on shared storage and are simply importable — so this
does not run into the "no installing on compute nodes" rule, which is about `apt`.

**Why the cleanup is not optional.** `pip --target` treats the target directory as isolated:
it cannot see the image's `site-packages`, so when `accelerate` declares a torch dependency
pip installs the *newest* torch (2.14) plus its entire CUDA 13 runtime — several GB of
`nvidia-*` wheels — into `pylibs`. Because `PYTHONPATH` precedes `site-packages`, that would
shadow the NGC-tuned torch `2.3.0a0+…nv24.03` and numpy `1.24.4` the container is built
around, and CUDA breaks in ways that look like a code bug. Deleting them from `pylibs`
restores the image's builds; `--target` never touched `site-packages`, so nothing is lost.

Verify afterwards that `numpy.__file__` and `torch.__file__` both resolve under
`/usr/local/lib/python3.10/dist-packages/`, not `pylibs`.

`hf_transfer` is required, not optional: both configs set `HF_HUB_ENABLE_HF_TRANSFER=1`, and
the Hub raises rather than falling back if the package is missing. It is worth having anyway —
it is substantially faster over the 566 GB of checkpoints.

Pins worth knowing: `transformers>=4.56` because the loader passes `dtype=` (4.56+; Qwen3
support, which the Skywork RMs need, landed in 4.51); `<5` to stay compatible with the image's
torch 2.3. Note the reference numbers in `results/` were produced under transformers 5.10 — the
first thing to suspect if a cluster run comes out close but not equal.

**(a) Custom image — reproducible, do it once things work.** `cluster/Dockerfile` layers our
extras onto the same NGC base and drops `torch` from the requirements for the reason above.
The cluster is amd64 and this Mac is arm64, so build on an amd64 machine or CI:

```bash
docker buildx build --platform linux/amd64 \
  -f cluster/Dockerfile -t <REGISTRY>/rm-bias:latest --push .
```

The registry must be **public**, or the cluster cannot pull without credentials. Then point
both `image.cpu` and `image.cuda` in `config.yaml` at it.

## 2. Stage code, data and checkpoints

Everything lives on PFSS, never on the instance — instance storage is wiped when a task ends,
and overrunning it crashes the whole compute node.

**How the connection works.** There is no direct SSH login. `det shell start` launches a container on a
cluster node (a *shell*, with PFSS mounted) and connects the terminal to it; `exit` only disconnects, the
shell keeps running until `det shell kill <id>` or its 1 h `idle_timeout`. Typing commands on the cluster
needs no further setup. `stage.sh`, however, runs **on the Mac** and copies with `rsync` over SSH, so it
needs an SSH host alias (the `<ssh-target>`) in `~/.ssh/config`. Work with two terminals, one on the Mac
(prompt ends in `%`) and one inside the shell (prompt `ck21zoly@<container>:...$`), and check the prompt
before every command.

**1. Start a staging shell** (Mac terminal; it turns into the cluster terminal). `slots=0` holds no GPU:

```bash
det shell start -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=0
```

**2. Replacing an earlier copy** (cluster terminal; skip on a first staging). `pylibs/` and `hf_cache/` sit
next to the repo folder, so swapping the folder needs no reinstall. Move the old copy aside rather than
deleting it (earlier results live under its `artifacts/`), and reuse its raw corpora so they do not cross
the VPN again (a corpus missing there is simply uploaded by `stage.sh`):

```bash
cd $PFSS
mv OneBiasAfterAnotherFork OneBiasAfterAnotherFork.old
for d in credit cv education; do mkdir -p OneBiasAfterAnotherFork/data/demographic/$d; cp -a OneBiasAfterAnotherFork.old/data/demographic/$d/raw OneBiasAfterAnotherFork/data/demographic/$d/; done
which rsync
```

Do not reuse its `pairs.jsonl` / `cells.jsonl`: step 6 regenerates them from the staged code. `which rsync` must
print a path — rsync is needed on both ends (it is in the image as of 2026-09-25). Delete the `.old`
folder once anything worth keeping is copied out.

**3. Create the SSH alias** (Mac terminal). The shell id is the UUID in `det shell list`, **not** the
container name in the cluster prompt:

```bash
det shell list
det shell show-ssh-command <shell-id>
```

It prints `ssh -o "ProxyCommand=<proxy>" ... -i <key> ck21zoly@<shell-id>`. Copy its parts into
`~/.ssh/config` (`chmod 600` it); drop `-tt`, which forces a terminal and breaks rsync:

```
Host det-stage
    HostName <shell-id>
    User ck21zoly
    ProxyCommand <everything inside the quotes after ProxyCommand=, keeping %h>
    IdentityFile <the path after -i>
    IdentitiesOnly yes
    StrictHostKeyChecking no
```

Every new shell has a new id and key, so `HostName` and `IdentityFile` change each time; the
`ProxyCommand` stays.

**4. Commit, test, then stage** (Mac terminal, repo root):

```bash
git status                      # must be clean: stage.sh refuses uncommitted changes and untracked files
ssh det-stage hostname
./cluster/stage.sh det-stage
```

The test must print the container name without asking for a password (`Permission denied (publickey)`:
wrong `User`/`IdentityFile`; a hang: VPN down or the shell ended). `stage.sh` sends the tracked files of the
committed tree (the copy has no `.git`; it writes `STAGED_COMMIT`, which every result's `meta.code` then names)
and the raw corpora (~105 MB, skipped where step 2 already copied them). It does **not** send generated data
(step 6). If it breaks off, rerun it — rsync skips what has arrived.

**5. Check** (cluster terminal):

```bash
cd $PFSS/OneBiasAfterAnotherFork
python -c "import torch, numpy, transformers; print(torch.__file__); print(numpy.__file__); print(transformers.__version__)"
python -m pytest -q
```

`torch` and `numpy` must resolve under `/usr/local/lib/python3.10/dist-packages/` (see §1); also
`python -c "import concept_erasure, sklearn"` (the erasure runner needs both).

**6. Regenerate the data** (cluster terminal, still slots=0; ~3 min, CPU only):

```bash
bash cluster/prepare_data.sh
```

It runs every generator from the staged raw corpora (credit, hiring at 12,000 bios, education, the grade-level
stage design, both A2 positioned manifests) and ends with `cluster/check_data.py`: every manifest present, generator
0.4.0, generated by the staged commit. `pilot.sh` runs the same check and refuses to start otherwise. Never rerun it
while a lane is running.

Then, still on **slots=0** (downloading while holding an A100 wastes the allocation):

```bash
export HF_HOME=/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias/hf_cache
python cluster/prefetch_models.py --check     # connectivity + free space
python cluster/prefetch_models.py --tier small
```

`--check` answers the one thing we could not determine from outside: **whether compute nodes
have outbound internet.** If they do not, fetch the checkpoints on a machine that does and
rsync the cache across.

**Embedding cache (since 2026-09-24).** Every runner embeds each unique text once per model and
stores the pooled state (`probes/embedding_cache.py`); probes, nulling, the α-sweep and all
offline analyses reuse it. Put it on PFSS, next to the results rather than inside the code tree:

```bash
export ONEJUDGE_EMBED_CACHE=/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias/embedding_cache
```

Size is about `unique texts × hidden size × 2 bytes` per model: roughly 0.1 GB for Qwen3-0.6B
and 0.4 GB for an 8B model over the whole education pool, 0.8 GB for a 70B. Files: one shard per 4,096
new texts (and at least one per call that embedded something new), a few dozen per run, far below the inode
budget; a killed run keeps every finished shard, so a resubmitted one continues. Shards are memory-mapped
when a cache is opened. Concurrent jobs on the same model are safe (each writes its own shards).
`ONEJUDGE_EMBED_CACHE=off` disables it.

Since 2026-09-27 the fingerprint names the environment (device type and GPU name, attention implementation,
torch and transformers versions), so a cache is never shared across hardware or library versions: a Mac
smoke run and a cluster run get separate directories, and a `pip` upgrade on the cluster starts a new cache.
The change invalidated every earlier cache once, including the first pilot's `$PFSS/embedding_cache` (its texts
are stale after the code review and the 0.4.0 data anyway); delete it before the pilot re-run. The local
`artifacts/embedding_cache` is gitignored, so `stage.sh` never copies it.

**Cross-marker decision design** (`runners/run_cross_marker.py`, the harm evidence since 2026-09-24). Per
record, template and encoding: 9 prompts (8 factorial cells + unmarked) x 5 responses = 45 decision texts,
plus the same 8 cells in the direct format for the placement check (usually cache hits after a battery run).
At 300 strong + 300 weak records, 2 templates and 2 encodings that is ~108k decision texts per domain and
model; the pilot fixes n (pilot-then-freeze). Every nulling variant, the cross-fitting, the geometry and the
alpha-sweep reuse the one embedding pass. It reads `cells.jsonl` next to the config's `dataset_source`,
which `prepare_data.sh` writes together with `pairs.jsonl` (they must come from the same generator run).
Education essays are long: expect roughly 3x the credit runtime per record.

**Every model is checked at load** (`verify_score_path`): the pipeline's reward, which is the score
head on the pooled last-token state, must reproduce the model's own score on two texts. A model that
pools differently is refused rather than silently mis-scored. As of 2026-09-24 that is **the
OpenAssistant DeBERTa RM** (first-token `ContextPooler`); it cannot run until it has its own pooling and
projection site.

**QRM-Gemma-2-27B** (since 2026-09-26) is loaded with our own implementation of its architecture
(`scoring/qrm.py`): its remote code imports a transformers constant that no longer exists, so
`trust_remote_code` fails under 4.57. Its score is a gated mix of quantile heads; the gate reads the end of
the user turn. The pipeline projects the last-token state only and holds the gate fixed
(`probes/heads.py`), the gates are cached in `<cache>/gates/`, and the cross-marker report adds a
`gate fixed` column (the part of each disparity that does not run through the gate). `verify_score_path`
checks both the score and the gate at load. Tested on a tiny random QRM only; the first cluster load of the
real checkpoint is its real test. Run it with `--model nicolinho/QRM-Gemma-2-27B` (one A100-80GB, ~54 GB
in bf16).

## 3. Parity smoke test — do this before spending the allocation

Everything so far ran on the MLX (Apple Silicon) backend. The cluster uses the
transformers/CUDA path, which the demographic arms have never exercised. We have reference
numbers, so this is a real regression check rather than a vibe check:

```bash
det shell start -w IL_rm_bias --config-file cluster/config.yaml     # slots: 1
# then, inside:
python runners/run_battery.py --config configs/demographic_credit_sex_qwen06.yaml --axes sex --encodings explicit
# -> artifacts/results/demographic/battery_credit_Skywork-Reward-V2-Qwen3-0.6B__sex__explicit.json
```

(Until 2026-09-28 this read `runners/run_experiment.py`, which exited without running anything from the
repo's initial commit on and was removed; the 2026-09-09 pass below predates that commit. `run_battery.py` is
the direct arm's runner.)

**Expected: auto-influence 1.00 baseline → 0.06 nulled.** A mismatch means the CUDA path
diverges, and every scaled number would inherit the fault. This single run catches padding
side, dtype, and chat-template differences at once.

Note the config's split: the probe is 150 records (`probe_records`, stratified by quality, the same
records for every axis) and the evaluation 200 pairs. Since 2026-09-16 the credit generator emits the
full sex × age × marital-status factorial (~6.4k pairs per single-axis cell), so the split is always
filled; `--n-records` is the only cap, and a small value under-fills it.
The expected numbers above predate the corrected German Credit codebook and the factorial design;
re-establish them after regenerating the credit data.

### Result — PASSED, 2026-09-09 (A100-80GB, torch 2.3.0a0+nv24.03, transformers 4.57.6)

| metric | reference (MLX, Mac) | cluster (CUDA) |
|---|---|---|
| probe accuracy | 91.5% | 93.00% |
| baseline `mean_gap` | +0.391 | +0.3794 |
| baseline `auto_influence` | 1.00 | **1.0000** |
| nulled `abs_mean_gap` | 0.069 | **0.0689** |
| nulled `auto_influence` | 0.06 | 0.1100 |

The decisive line is `abs_mean_gap`: the nulled effect in raw reward units agrees to three
decimals across two frameworks, dtypes and backends. The rewards are right.

The `auto_influence` difference is the effect `CLAUDE.md` already documents from the MLX
parity work: "nulled/debiased headline metrics can differ by ~0.03 (3 comparison-flips/100),
a stable bf16-vs-fp32 precision effect near decision ties, not a framework bug." After nulling,
`mean_gap` is 0.0083 -- effectively zero -- so each A/B comparison is decided by numerical
noise. `pref_a_rate` 0.555 vs the reference 0.53 is **0.7 SE** at n=200. Indistinguishable.

**Implication for the scaling runs.** `auto_influence` is a preference *rate*: it saturates at
1.00 whenever an effect is consistent, and is pure noise once an effect is nulled -- both ends
of its range are where we actually read it. Report `mean_gap` / `abs_mean_gap` as the primary
magnitude alongside it, as the A2 arm already does with `identity_gap`.

## 4. Scale the ladder

**70B pilot (the first `70b` lane, 2026-09-26; a record — the lane now follows amendment (1), see "The pilot"
below).** Llama-3.1-70B RB2 on both GPUs, small n, for speed and a first look at quality. Estimate, extrapolated from the 8B pilot (8.8x the FLOPs; `device_map="auto"` runs
the two GPUs one after the other, so they add memory, not speed): about 6 / 7 / 2 texts/s for credit / hiring /
education. Credit ~45 min (model load ~5, 50 probe records ~5, 60 records x 212 texts ~35), hiring ~40 min,
education ~80 min (32 records, batch 2): **about 3 h in total, uncertain by ±50%**. Credit runs first; its timing
line predicts the rest.

1. Download (unattended, no GPU; ~139 GB of safetensors from the pinned revision, `.bin` skipped):
   ```bash
   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=0 \
     --config idle_timeout=12h --config description=prefetch_70b python cluster/prefetch_models.py --tier 70b
   ```
   `det command logs <id> --tail 5` shows progress; wait until `det command list` shows it terminated.
2. Run (both GPUs, so nothing else can run meanwhile):
   ```bash
   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=2 \
     --config idle_timeout=12h --config description=pilot_70b bash cluster/pilot.sh 70b
   ```
**Result, 2026-09-26:** 2 h 15 min in total (estimate ~3 h). Placement 40/44 modules on the two GPUs, no
offloading; peak 136-139 GiB of 160 (education fits at batch 2 only). Forward passes 7.1 / 8.1 / 2.0 texts/s
(credit / hiring / education), model load ~2.8 min per process, mechanism layer + metrics ~10 s per domain.
Extrapolated to full size (150 probe records; credit 653 records, hiring and education 300 + 300): credit ~5.4 h,
hiring ~4.4 h, education ~17.5 h, **~27 h of both GPUs per 70B model**, two thirds of it education (compute-bound).

3. Read out (any `slots=0` shell): `grep -h "^timing\|OFFLOADED\|placement" $PFSS/pilot_logs/*_70b.log`. A
   `OFFLOADED` warning means accelerate put layers on the CPU (the GPUs were too small) and the timings are
   not representative.

**`.bin`-only checkpoints.** transformers >= 4.50 refuses PyTorch `.bin` weights under torch < 2.6
(CVE-2025-32434), and the image has torch 2.3. Both AllenAI RB2 models (8B, 70B) and the DeBERTa RM publish
only `.bin` on main. The RB2 models load from Hugging Face's own safetensors conversion (SFconvertbot pull
requests), pinned in `scoring/backend.py::PINNED_REVISIONS`; `prefetch_models.py` fetches that revision and skips
`.bin` wherever safetensors exist. Checked on the Hub 2026-09-26: the Nemotron ids exist; the two 32B ones are
sequence classifiers stored in fp32 (~128 GB download, 64 GB in bf16), **Llama-3.3-Nemotron-70B-Reward is a
`LlamaForCausalLM`**, but a Bradley-Terry scalar RM underneath: its model card reads the reward as the raw logit of
vocabulary token 0 after the conversation, i.e. row 0 of `lm_head` applied to the last-token state (linear, no bias).
Loaded since 2026-09-26 through `scoring/logit_reward.py` (that row as the score head; the load-time check
compares it with the model's own logit). Its model card tokenizes with the chat template's own ids, which carry no
BOS, so the pipeline adds no special tokens for it (`scoring/backend.py::TEMPLATE_TOKENIZED`); every other model keeps
the tokenizer's defaults, as the Skywork cards and RewardBench (RB2) score. The loader logs how each model's tokens
relate to its template. `--tier 70b` now downloads both 70B models (~280 GB).

**Throughput trial 1 — 2026-09-25** (A100-80GB, Qwen3-0.6B, `run_cross_marker.py --n-strong 20 --n-weak 20`,
batch 8, fresh embedding cache). Each domain put 12,882 texts through the model: 8,480 decision and
placement texts, which grow with the record count, and 4,402 direct-probe texts for the 150 probe records,
which do not.

| domain | wall | forward passes | peak GPU memory |
|---|---|---|---|
| credit | 3 min 09 s | 42–85 s | 3.6 GB (all runs) |
| hiring | 3 min 44 s | 41–82 s | |
| education | 5 min 25 s | 156–312 s | |

The forward-pass figures are ranges because they were summed from tqdm bars, which sometimes print their
final line twice. Mean GPU utilisation was 25%: batch 8 leaves the card mostly idle. The rest of the wall
time (model load, analysis) could not be split, because the log had no timestamps. Since then the runner
records the seconds per phase, the texts through the model, the texts/s and the peak GPU memory in
`summary["timing"]` and on the report's last line, and `--model` / `--batch-size` override the config.
The trial outputs are named `trial_crossmarker_*`; they are throughput checks, not results.

**Throughput trial 2 — 2026-09-25** (A100-80GB, `summary["timing"]`, one empty cache per run; up to two
runs in parallel, so the CPU phases are noisy by up to ±50%). Seconds per phase:

| run | load | direct dirs | prepare | embed | mechanism | metrics | total | texts/s | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| credit 20+20, 0.6B, b8 | 9 | 121 | 4 | 35 | 140 | 10 | 319 | 228 | 1.4 |
| credit 60+60, 0.6B, b8 | 7 | 95 | 14 | 106 | 322 | 23 | 567 | 228 | 1.4 |
| credit 60+60, 0.6B, b32 | 6 | 90 | 14 | 85 | 307 | 23 | 525 | 285 | 2.2 |
| education 20+20, 0.6B, b8 | 8 | 189 | 11 | 104 | 185 | 11 | 509 | 78 | 2.5 |
| education 60+60, 0.6B, b8 | 5 | 120 | 33 | 299 | 134 | 24 | 617 | 81 | 2.6 |
| education 60+60, 0.6B, b32 | 5 | 124 | 32 | 287 | 158 | 24 | 630 | 85 | 6.9 |
| credit 20+20, Llama-3.1-8B, b8 | 27 | 189 | 3 | 144 | 166 | 10 | 540 | 56 | 15.2 |

Reading: 202 scoring texts per record; the direct directions are a fixed ~1.5–3 min (4,800 texts, mostly
not forward passes); the **mechanism layer runs on the CPU and grows with n** (~2.3 s per record for credit)
and is the largest phase for credit and hiring; batch 32 buys +25% forward speed on credit and +5% on
education, too little to leave batch 8. The 8B model's forward pass is ~4x slower than the 0.6B's, and it
passed `verify_score_path` (Skywork-Reward-V2-Llama-3.1-8B, 2026-09-25).

**The pilot** (pilot-then-freeze; the rules are stated in the working notes, 2026-09-25, amendments (1) and (2) of
2026-09-26, the comparative rule of 2026-09-30, all before any run) **and the test run of every planned experiment**
(re-run on the reviewed code, 2026-10-01). `cluster/pilot.sh <lane>`, one lane per model: `small` = Qwen3-0.6B,
`8b` = Skywork-Reward-V2-Llama-3.1-8B, `70b` = Llama-3.1-70B RB2 (two GPUs), and `smoke` = Qwen3-0.6B with every
step at a tiny size. With 4 A100s (since 2026-10-01) `small`, `8b` and `70b` run in parallel.

- **Part 1, sizing** (what the rules read): the probe-size curve to 500 probe records, the cross-marker design and the
  comparative design, for credit, hiring and education. `small`/`8b`: cross-marker on the full credit and education
  pools and 600 + 600 hiring records, comparative at the configured 150 pairs per pairing (credit: its capacity,
  ~60). `70b`: amendment (1), cross-marker 100 + 100 (education 50 + 50), comparative 50 pairs per pairing
  (education 25), education at batch 2 (comparative: 1).
- **Part 2, test runs** of everything else: the placement matrix, the reasoning arm (flip, probe, erasure per domain,
  the transfer), the direct arm (battery per domain, grade-level stage, A2 on both positioned manifests), A2's main
  effect, the decision floor (both encodings), additivity, the real-field check, the scrub check. `small` at the
  configured sizes (the full-scale dress rehearsal), `8b` and `70b` reduced (sizes at the top of `pilot.sh`).
  Nothing in part 2 sizes anything.

Expected: `smoke` ≲1 h; `small` ~8–10 h; `8b` ~1 day; `70b` 1–2 days (amendment (1) alone was estimated at ~13 h of
both GPUs; education dominates). Part 1 runs first in every lane, so the sizing inputs arrive before the test runs.

1. Stage and regenerate (§2, steps 4–6). Delete the first pilot's outputs and cache:
   `rm -rf $PFSS/embedding_cache $PFSS/pilot_logs artifacts/results/demographic/pilot`.
2. Smoke first (Mac terminal, repo root, `DET_MASTER` exported), and read its last line (`0 failed step(s)`):

   ```bash
   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config idle_timeout=6h --config description=pilot_smoke bash cluster/pilot.sh smoke
   ```
3. Then the three lanes:

   ```bash
   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config idle_timeout=48h --config description=pilot_small bash cluster/pilot.sh small
   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config idle_timeout=48h --config description=pilot_8b bash cluster/pilot.sh 8b
   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=2 --config idle_timeout=48h --config description=pilot_70b bash cluster/pilot.sh 70b
   ```

   Unattended: the laptop can close. `idle_timeout=48h` guards against Determined counting a command without
   a connection as idle; the command ends when its lane does. The 70B checkpoint must be on PFSS first
   (`prefetch_models.py --tier 70b`, ~139 GB for this model).
4. Check: `det command list` (state), `det command logs <id> --tail 20` (one `start` / `done` / `FAILED` line
   per step, with its timing line; the last line lists the failed steps), and the per-step logs in
   `$PFSS/pilot_logs/`.
5. A lane that stopped (a failed step, a killed command) is resubmitted with the same command: steps whose
   JSON exists in `artifacts/results/demographic/pilot/` are skipped, and the embedding cache
   (`$PFSS/embedding_cache`) serves every state already computed.
6. Afterwards, on any shell: `python runners/pilot_sizing.py --inputs artifacts/results/demographic/pilot/*_*.json`
   prints the records needed per group (for δ × m) and the probe-records answer per domain. The pilot JSONs
   carry ids and numbers only, so they can be copied to the Mac (the `_rewards.jsonl` side files too; the
   `_directions.pt` files hold directions only).

**First loads of the other models** (2026-10-02). Before a main run commits GPU hours to a model the pilot does not
cover, three steps:

1. Download every missing checkpoint (no GPU; already-cached files are skipped; ~0.5 TB for the mid tier and the
   second 70B — `--check` prints the free space first):
   ```bash
   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=0 --config idle_timeout=24h --config description=prefetch_all python cluster/prefetch_models.py --tier all
   ```
2. `cluster/load_check.py` on one GPU: each model loaded as the runners load it (pinned revision, attention,
   offload refusal, `verify_score_path`), then peak memory and texts/s at 2,048 and 4,096 tokens for batch 1, 2,
   4, … up to the first out-of-memory. Results in `artifacts/results/load_check/` (numbers only). **While pilot lanes
   run, do not re-stage** (it would change `STAGED_COMMIT` under them): copy the script to `$PFSS/tools/` and run it
   from the staged repo, which it imports unchanged (`meta.script_sha256` names the copy):
   ```bash
   ssh det-stage mkdir -p /pfss/mlde/workspaces/mlde_wsp_IL_rm_bias/tools                      # Mac, a slots=0 shell open
   scp cluster/load_check.py det-stage:/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias/tools/
   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config idle_timeout=12h --config description=load_check bash -c "cd \$PFSS/OneBiasAfterAnotherFork && python \$PFSS/tools/load_check.py"
   ```
3. Once the lanes have finished (re-stage then): the smoke lane on the model, every runner at a tiny size, with
   a batch that `load_check` showed to fit:
   ```bash
   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config idle_timeout=24h --config description=smoke_qrm bash -c "PILOT_EXTRA='--batch-size 2' bash cluster/pilot.sh smoke nicolinho/QRM-Gemma-2-27B"
   ```

| models | `resources.slots` |
|---|---|
| 0.6B, DeBERTa, 3× 8B | 1 |
| 2× 27B, 2× 32B (54–64 GB bf16) | 1 |
| 2× 70B (~140 GB bf16) | **2** |

`device_map="auto"` is already in the loader, so multi-GPU sharding needs no code change —
but it shards across the GPUs visible to **one process** and cannot span nodes.

For **experiments**, therefore, set `resources.is_single_node: true`, or a `slots: 2` request
could be placed one GPU per node and the 70B load would fail or silently see half the memory.
Do **not** set it in `config.yaml`: NTSCs (notebooks, shells, commands) reject it with
`cannot be set for NTSCs`, because an NTSC is one container on one node anyway — the
guarantee is implicit there and only needs stating for multi-node-capable experiments.

Interactive:
```bash
det shell start -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=2
```

Unattended (preferred once the smoke test passes — it releases the GPU when the script exits):
```bash
det experiment create --project_id <ID> cluster/config.yaml    # add an `entrypoint:`
```

Create a project first: `det project create IL_rm_bias scaling`.

---

## Things that will bite

- **At most 4 GPUs at once per workspace** (2 until 2026-10-01). The cluster's "GPU slots limiter" counts every
  running allocation of the workspace (shells, commands, experiments; `slots=0` shells count 0). A task that
  would exceed the limit is **not queued**: it starts, is stopped at once with exit code 1, and `det command
  list` shows it TERMINATED (`Allocations slots limit reached - Limit: …` in `det command logs`). The three pilot
  lanes (1 + 1 + 2) fill it: anything else, a GPU shell included, has to wait.
- **Thread oversubscription.** Without a cap, torch and numpy use every core of the node for each
  operation, and the cross-marker mechanism layer (thousands of tiny per-record torch ops) ran 13x slower:
  credit 0.6B explicit, 1,289 s vs 96 s with `OMP_NUM_THREADS=8` (profiled 2026-09-26; the whole run 1,513 s
  vs 258 s). `config.yaml` now sets `OMP_NUM_THREADS=8` and `MKL_NUM_THREADS=8` for every task. The pilot
  timings in this README predate the cap.
- **Unattended runs:** `det command run -d ... bash -c "..."` runs one command in its own container,
  independent of the SSH connection and VPN, and frees the GPU when it exits. A run started inside a
  `det shell` dies with the connection (laptop closed, VPN down).
- **No default compute pool** on this workspace, so `resource_pool: 42_Compute` must be set on
  every launch. It is in `config.yaml`; do not drop it. `42_Priority` does not exist for us.
- **Idle shells keep burning GPU quota.** `det shell kill <id>` when done, or launch with
  `slots=0`. With a two-month allocation, a forgotten JupyterLab is expensive.
- **Write results to PFSS.** Anything under the instance is deleted on exit, and overfilling
  instance storage crashes the node and any job sharing it.
- **Inodes:** aim under 2M per workspace. We are naturally fine — a few large corpora and
  single-file `pairs.jsonl` — but the HF cache of 11 checkpoints is the one real consumer.
  `tar` up caches you are done with rather than leaving them expanded.
- **Determined.AI is end-of-life.** The only docs are at
  `https://login01.ai.tu-darmstadt.de:8080/docs/index.html`, and its search is broken; navigate
  by the left-hand tree.
- **Two Nemotron model ids were never verified** (`working_notes.tex` flags this). They are in
  `prefetch_models.py` but skipped by default — confirm them against current NVIDIA releases
  before the 32B/70B runs, rather than discovering it mid-ladder.
