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
  "transformers>=4.51,<5" accelerate datasets textstat scikit-learn concept-erasure \
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

Pins worth knowing: `transformers>=4.51` because Qwen3 support (which the Skywork RMs need)
landed there; `<5` to stay compatible with the image's torch 2.3. Note the reference numbers
in `results/` were produced under transformers 5.10 — the first thing to suspect if a cluster
run comes out close but not equal.

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

Do not reuse its `pairs.jsonl` / `cells.jsonl`: they come from an older generator. `which rsync` must
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

**4. Test, then stage** (Mac terminal, repo root):

```bash
ssh det-stage hostname
./cluster/stage.sh det-stage
```

The test must print the container name without asking for a password (`Permission denied (publickey)`:
wrong `User`/`IdentityFile`; a hang: VPN down or the shell ended). `stage.sh` sends code, the raw corpora
(~665 MB, skipped where step 2 already copied them) and the generated pairs and cells (~1.9 GB); if it
breaks off, rerun it — rsync skips what has arrived.

**5. Check** (cluster terminal):

```bash
cd $PFSS/OneBiasAfterAnotherFork
python -c "import torch, numpy, transformers; print(torch.__file__); print(numpy.__file__); print(transformers.__version__)"
python -m pytest -q
```

`torch` and `numpy` must resolve under `/usr/local/lib/python3.10/dist-packages/` (see §1).

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
and 0.4 GB for an 8B model over the whole education pool, 0.8 GB for a 70B. Files: one shard per call
that embedded something new, a few dozen per run, far below the inode budget. Concurrent jobs on the
same model are safe (each writes its own shards). `ONEJUDGE_EMBED_CACHE=off` disables it.

The fingerprint does not include the device. Local (Mac) runs are smoke tests only — publishable numbers
come from the cluster — and the local `artifacts/embedding_cache` is cleared before the cluster phase; it
is gitignored, so `stage.sh` never copies it.

**Cross-marker decision design** (`runners/run_cross_marker.py`, the harm evidence since 2026-09-24). Per
record, template and encoding: 9 prompts (8 factorial cells + unmarked) x 5 responses = 45 decision texts,
plus the same 8 cells in the direct format for the placement check (usually cache hits after a battery run).
At 300 strong + 300 weak records, 2 templates and 2 encodings that is ~108k decision texts per domain and
model; the pilot fixes n (pilot-then-freeze). Every nulling variant, the cross-fitting, the geometry and the
alpha-sweep reuse the one embedding pass. It reads `cells.jsonl` next to the config's `dataset_source`,
which `stage.sh` stages together with `pairs.jsonl` (they must come from the same generator run).
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
python runners/run_experiment.py --config configs/demographic_credit_sex_qwen06.yaml
```

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

**70B pilot (lane `70b` of `cluster/pilot.sh`, 2026-09-26).** Llama-3.1-70B RB2 on both GPUs, small n, for speed
and a first look at quality. Estimate, extrapolated from the 8B pilot (8.8x the FLOPs; `device_map="auto"` runs
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
3. Read out (any `slots=0` shell): `grep -h "^timing\|OFFLOADED\|placement" $PFSS/pilot_logs/*_70b.log`. A
   `OFFLOADED` warning means accelerate put layers on the CPU (the GPUs were too small) and the timings are
   not representative.

**`.bin`-only checkpoints.** transformers >= 4.50 refuses PyTorch `.bin` weights under torch < 2.6
(CVE-2025-32434), and the image has torch 2.3. Both AllenAI RB2 models (8B, 70B) and the DeBERTa RM publish
only `.bin` on main. The RB2 models load from Hugging Face's own safetensors conversion (SFconvertbot pull
requests), pinned in `scoring/backend.py::PINNED_REVISIONS`; `prefetch_models.py` fetches that revision and skips
`.bin` wherever safetensors exist. Checked on the Hub 2026-09-26: the Nemotron ids exist; the two 32B ones are
sequence classifiers stored in fp32 (~128 GB download, 64 GB in bf16), **Llama-3.3-Nemotron-70B-Reward is a
`LlamaForCausalLM`** (a generative reward) and cannot be scored without its own head adapter.

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

**The pilot** (pilot-then-freeze; the rules are stated in the working notes, 2026-09-25, before any pilot run).
`cluster/pilot.sh <lane>` runs one lane per GPU: `small` = Qwen3-0.6B, `8b` = Skywork-Reward-V2-Llama-3.1-8B.
Each lane runs the probe-size curve (`runners/run_probe_curve.py`) for credit, hiring and PERSUADE, then the
cross-marker design on the full credit and education pools and 600 + 600 hiring records. Expected: ~4 h for
`small`, ~9–10 h for `8b`. The two lanes use both GPUs the workspace allows, so nothing else can run meanwhile.

1. Stage first (§2). `stage.sh` copies **tracked** files only, so `cluster/pilot.sh` and the two new runners
   must be committed (or at least `git add`ed) before staging.
2. Submit both lanes from the Mac (repo root, `DET_MASTER` exported):

   ```bash
   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config idle_timeout=24h --config description=pilot_small bash cluster/pilot.sh small
   det command run -d -w IL_rm_bias --config-file cluster/config.yaml --config idle_timeout=24h --config description=pilot_8b bash cluster/pilot.sh 8b
   ```

   Unattended: the laptop can close. `idle_timeout=24h` guards against Determined counting a command without
   a connection as idle; the command ends when its lane does.
3. Check: `det command list` (state), `det command logs <id> --tail 20` (one `start` / `done` / `FAILED` line
   per step, with its timing line), and the per-step logs in `$PFSS/pilot_logs/`.
4. A lane that stopped (a failed step, a killed command) is resubmitted with the same command: steps whose
   JSON exists in `artifacts/results/demographic/pilot/` are skipped, and the embedding cache
   (`$PFSS/embedding_cache`) serves every state already computed.
5. Afterwards, on any shell: `python runners/pilot_sizing.py --inputs artifacts/results/demographic/pilot/*_*.json`
   prints the records needed per group (for δ × m) and the probe-records answer per domain. The pilot JSONs
   carry ids and numbers only, so they can be copied to the Mac.

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

- **At most 2 GPUs at once per workspace.** The cluster's "GPU slots limiter" counts every running
  allocation of the workspace (shells, commands, experiments; `slots=0` shells count 0). A task that
  would exceed 2 is **not queued**: it starts, is stopped at once with exit code 1, and `det command
  list` shows it TERMINATED (`Allocations slots limit reached - Limit: 2` in `det command logs`). Submit
  a third job only after one has finished. A 70B run (`slots: 2`) needs the workspace to itself.
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
