# Running on the hessian.AI 42 cluster

Workspace **`IL_rm_bias`** (id 142) · PFSS root **`/pfss/mlde/workspaces/mlde_wsp_IL_rm_bias`**

Our workload is **inference only**: forward passes, last-layer activation extraction, and
linear algebra. No training, no checkpointing, no hyperparameter search — so none of
Determined's trial machinery is needed. A task with a plain `entrypoint` just runs our CLI.

---

## 0. Connect

```bash
source .venv-mlx/bin/activate                 # `det` 0.35.0 lives here
export DET_MASTER=https://login01.ai.tu-darmstadt.de:8080
det user login ck21zoly                        # TU VPN must be up first
det workspace ls
```

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

```bash
det shell start -w IL_rm_bias --config-file cluster/config.yaml --config resources.slots=0
det shell show-ssh-command <shell-id>
./cluster/stage.sh <ssh-target>               # ~730 MB of code + corpora
```

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
OpenAssistant DeBERTa RM** (first-token `ContextPooler`), and the QRM-Gemma quantile head is untested.
Neither can run until it has its own pooling and projection site.

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
