**English** · [中文说明](README.zh-CN.md)

---

# MiniMax H3 on an 8 GB Laptop — Six Measured Traps

> Hardware: **RTX 5060 Laptop 8 GB (sm_120) + 15.26 GiB RAM**, Windows 11 25H2,
> ComfyUI 0.35.0 (a7b1d39), PyTorch 2.14.0+cu130, Python 3.13.12
> Everything here is measured on that machine. Numbers that are *not* measured are labelled.

This is not a "best settings" post. It's six things that cost me hours, each with a
reproduction you can run, and each one something I got wrong before I got it right.

---

## 1. A 10-second H3 video silently loses its subject at ~7 s — it's the prompt, not the model

**Symptom.** A 10 s / 20-step / 768×1024 I2VA generation looked great for ~6 s, then the
character vanished and the last 3 seconds were empty scenery. Pushing in to a close-up
earlier in the clip produced a **featureless skin patch where the face should be**.

**Root cause.** The prompt described only `[Shot 1]` — one shot, no timeline coverage —
and gave a camera move with **no destination**:

```
❌ 镜头以小幅度、慢速度向前推进            (amplitude + speed, no target)
✅ The camera pushes in with small amplitude at slow speed
   toward her hand clutching the front of her sweater, and comes to rest at a medium shot.
```

The official guide (`h3-prompt-writing/references/base-en.txt`) requires:

- §4.3 — camera motion = **type + amplitude + speed**, and its examples embed a target
  (`toward the folded letter in her hands`). Amplitude/speed alone give the model no stop
  condition, so it keeps pushing until the subject leaves frame.
- §4.2 — later shots need `[Shot N] At 00:0S.SSS, the camera cuts to ...`. With only
  `[Shot 1]` and 10 seconds to fill, the model improvises the tail.
- §3.1 — I2VA structure is **first-frame anchor → action onset → development → result or
  reaction**. The original prompt stopped at "development".

Two more mistakes in the same sentence:

- Writing 「保持…与**整体构图不变**」 while also asking for a push-in is **self-contradictory**.
  The guide's wording is `preserving her appearance, clothing, seat position, and the
  carriage layout` — **not** composition, which necessarily changes under a camera move.
- Asking to push in on a face that the reference image deliberately hides behind hair
  forces the model to invent detail it does not have.

**Fix and result.** Same seed, same 20 steps, same resolution, **only the prompt changed**:

| | before | after |
|---|---|---|
| 7.0–10.0 s | ❌ subject gone | ✅ subject present the whole clip |
| close-up face | ❌ blank skin | ✅ profile silhouette, hair still covering |
| push-in | ❌ overshoots to extreme close-up | ✅ stops at medium shot |
| cut at 6 s | — | ✅ lands where asked |

Evidence: `evidence/old_vs_new.png`, `evidence/dense_grid.png` (before),
`evidence/new_grid.png` (after). Videos: `H3_Quality_00001_.mp4` vs `H3_Quality_00002_.mp4`.

**Takeaway:** if an H3 clip degrades late or loses its subject, suspect prompt timeline
coverage **before** touching steps, samplers or acceleration nodes. A 20-step official
baseline reproduces the failure exactly.

---

## 2. On sm_120, the SageAttention patch is a pessimisation — and it disables backend selection

**Measured on H3's real geometry** (heads=56, head_dim=128, S=8192, bf16):

| backend | ms | rel L2 error |
|---|---|---|
| SDPA bf16 *(what you get with no flags)* | 143.1 | 0.0025 |
| **comfy-kitchen INT8** | **22.8** | **0.0164** |
| kitchen INT8 + `head_chunks=10` | 26.2 | 0.0164 |
| **KJ SageAttention (sm89 kernel)** | **29.0** | **0.0387** |
| sage + `head_chunks=10` | 31.6 | 0.0387 |

comfy-kitchen is **6.3× faster than SDPA and 1.27× faster than sage, at 2.4× lower error.**

Worse: `MiniMaxH3MemoryEfficientSageAttentionPatch` calls `_sageattn_int8_fp8_nhd`
directly (`ltxv_nodes.py:2106`), **bypassing `optimized_attention`/`wrap_attn` entirely**.
While that patch is in the graph, `ModelAttentionBackend` and `--use-ck-attention` have
**no effect on H3's 50 blocks**.

**Do this instead:** drop the sage patch, add
`ModelAttentionBackend = "comfy kitchen attention"`.

Also worth knowing: `MiniMaxLowVRAMAttention` and the sage patch do **not** conflict.
The head-chunk setting lives in `transformer_options["minimax_head_chunks"]`
(`minimax_nodes.py:188`) and `minimax_sageattn_forward` reads it (`ltxv_nodes.py:2102`),
so both orders behave identically. `head_chunks=10` measured **bit-identical error** to
unchunked, at +15% time.

Backend name: the CLI flag is `--use-ck-attention` (not `--use-comfy-kitchen-attention`),
and the six attention flags are in one `add_mutually_exclusive_group()`. comfy-kitchen's
CUDA backend requires cu130+ (`comfy/quant_ops.py:22-28`).

---

## 3. `SaveVideo`'s H.264 re-encode fails on odd dimensions — use `VHS_VideoCombine`

`SaveVideo` exposes a `crf` control, but reaching it through the API is non-obvious, and
once you do, **it fails — whenever the video's width or height is odd**:

```
av.error.ExternalError: [Errno 542398533] Generic error in an external library:
    'avcodec_open2("libx264", {})'          <- fails with NO options at all
    'avcodec_open2("libx264", {'crf': '12.0'})'
```

**It is the dimensions, not a broken encoder.** `video_types.py` hands the frame's
width/height to libx264 unchanged while setting `pix_fmt = "yuv420p"` — and `yuv420p`
subsamples chroma 2x2, so **both dimensions must be even**. libx264 refuses to open with
`EINVAL (22)`, and since PyAV opens the encoder lazily inside `encode()`, it surfaces as a
generic external-library error with no hint about dimensions.

Measured on the same PyAV (18.1.0) with no ComfyUI running: `64x64` and `1080x1920` open
fine; `65x63`, `63x65`, `65x65`, `1171x2532`, `1080x1441` all fail. Our reference image is
**1259 x 1672** — odd width. Reported upstream as
[ComfyUI #16544](https://github.com/Comfy-Org/ComfyUI/issues/16544).

`format=auto` is not a workaround: it *"preserves a compatible source stream"*, so it
cannot change quality at all.

The API spelling for these nested combos is **dotted top-level keys**
(`comfy_api/latest/_io.py: finalize_prefix` joins with `.`):

```json
"format": "mp4",
"format.codec": "h264",
"format.codec.encoding": "re-encode",
"format.codec.encoding.crf": 12.0
```

Three spellings I tried first, and what each did — all silently useless or wrong:

| spelling | result |
|---|---|
| flat `codec` / `encoding` / `crf` | **silently dropped** (`crf=None`); output byte-identical |
| one dict nested inside `format` | validator dropped `format` → `TypeError: missing required argument: 'format'` |
| dotted keys | ✅ parsed, reached `avcodec_open2` → **then libx264 failed to open** |

**Fix:** `VHS_VideoCombine` (VideoHelperSuite) shells out to the ffmpeg executable instead
of PyAV. Its `crf` works — same 48-frame input:

```
crf=12  ->  570,872 bytes
crf=19  ->  287,454 bytes
crf=40  ->   40,296 bytes
```

It also accepts `IMAGE` + `AUDIO` directly, so `CreateVideo` can be deleted.
Note its `pix_fmt` / `crf` / `save_metadata` / `trim_to_audio` inputs are **not listed in
`/object_info`** (they hang off the `format` combo), so a UI→API converter will drop them.

Probe: `scripts/probe_encoder.py` — it runs in **seconds** because it never loads a model
(just `LoadImage → RepeatImageBatch → save`). Highly recommended pattern for poking at
output nodes.

---

## 4. The turbo LoRA only applies 80.3% to the pruned base

LoRA modules counted: **259 total — 208 apply (80.3%), 51 do not (19.7%)**.
All 51 are `adaln_proj.linear`, and the shapes explain it:

```
base (pruned)  adaln_proj input dim = 8
LoRA (unpruned) adaln_proj input dim = 2688
```

The LoRA metadata says so itself:
`incompatible_base: MiniMax-H3 pruned_* (AdaLN input is 8, LoRA input is 2688)`.

AdaLN is the timestep-modulation path, which is exactly what a **step-distilled** LoRA
needs to modify. So the distillation is only partially applied.

The error spam this produces —
`ERROR lora diffusion_model.blocks.N.adaln_proj.linear.weight shape '[96768, 8]' is invalid
for input of size 260112384`, 50× per step — is not a "progress indicator". It is 50
per-layer failures per step. It doesn't abort the run, which is why it's easy to dismiss.

Reproduce: `scripts/lora_applicability.py`.

---

## 5. The binding constraint is 16 GB of RAM, not 8 GB of VRAM

| ComfyUI session | per block | 10 s clip |
|---|---|---|
| **freshly restarted** | **0.9–6 s** | **34–44 min** |
| after ~1.7 h of continuous sessions | **212 s** | **~12 h** |

Same settings. The tell is **GPU power**: healthy sampling draws **64–98 W**; when the
machine is thrashing it sits at a **flat 34–36 W** at 99% "utilisation", because the
utilisation counter counts memory-shuffling kernels too. I used that as a stop signal.

Supporting measurements:

- Output resolution is *not* the lever: one 12 GB-card test shrank the workload 40× for
  **0.5%** less peak VRAM and 1.0% less system RAM.
- Capacity is set by **resolution × duration**, not steps: on a 4060 8 GB, 1024×576/243f
  completes at 20 steps (2762.89 s) while 1344×768/**362f** CUDA-OOMs at 1 step.
- Official limits: **short side 768**, area ≤ 768×1344, multiples of 32. At 3:4 the ceiling
  is 768×1024 = 0.786 MP; at 9:16 it's 768×1344 = 0.984 MP. Note the `ResolutionSelector`
  megapixel dial is a trap — the same MP value gives different short sides per aspect ratio
  (1.0 MP at 3:4 yields 896×1184, short side 896, **outside the trained range**).
- Attempting 9:16 (768×1344, +31% pixels) on this 16 GB machine pushed it straight into
  thrashing (flat 34 W) even though it had run fine at 3:4 minutes earlier. **+31% pixels
  was over the line.**

Related: `--disable-pinned-memory` is **required** on 16 GB machines —
`MAX_PINNED_MEMORY = ram * 0.90` otherwise gets the process killed.

**How to tell you are thrashing, in one number.** `nvidia-smi` power draw. Healthy
sampling on this machine sits at **64–98 W**; thrashing sits at a **flat 34–36 W while
`utilization.gpu` still reads 99%**, because that counter counts memory-shuffling kernels
too. Sample it in a loop and stop the run if it goes flat — that was the signal that ended
the 9:16 attempt five minutes in instead of hours later.

---

## 6. 8 steps vs 20 steps: the composition barely changes, the time halves

Same seed, same prompt, same 768×1024, both following the fixed prompt from §1:

| | 20 steps | 8 steps |
|---|---|---|
| camera plan | push-in → cut at 6 s → shoulder/neck | **identical** |
| subject present 0–10 s | ✅ | ✅ |
| knit-texture detail in close-ups | slightly more | slightly softer |
| wall clock (10 s clip) | **33.8–43.5 min** | **14.3 min** |
| bitrate | 2067 kb/s (SaveVideo auto) | 10393 kb/s (VHS crf=12) |

**8 steps costs 1/2.5 of the time for what is, at normal viewing size, the same shot.**

Caveat, stated plainly: the two runs used *different encoders*, and the 8-step one had
**5× the bitrate** while still looking marginally softer. That is weak evidence that the
extra steps buy a little genuine detail rather than just surviving compression better.

Comparison sheet: `evidence/steps_8_vs_20.png`. Still images are the author's own; the
clips are in `videos/`.

**Practical read:** 8 steps is a sensible daily driver at 10 s / 768×1024 on this hardware,
with 20 steps reserved for final renders.

---

## Timing model (only valid at 5 s / 768×1024 / kitchen / LoRA 1.0)

```
T_wall ≈ 102 + 34.1 × steps        residual ≤ 3.2% over 4/6/8/20 steps
```

Measured: 4 steps 245.3 s · 6 steps 297.4 s · 8 steps 377.9 s · 20 steps 785.3 s.

**Do not extrapolate this to longer clips.** At 10 s the per-step cost is ~95 s, not the
~57 s the 5 s slope predicts — I got this wrong and predicted 28.7 min for a run that
actually took 38–43.5 min. At 10 s / 20 steps, four runs measured:
**2613, 2283, 2560, 2029 s → 33.8–43.5 min, ±11%.**

Useful consequence at 5 s: **20 steps / 4 steps = 3.20×, not 5×** — a ~102 s fixed
overhead is 42% of a 4-step run and only 13% of a 20-step one.

---

## Repo layout

```
scripts/
  probe_encoder.py        # seconds-fast SaveVideo vs VHS crf probe (no model loading)
  lora_applicability.py   # per-key LoRA vs pruned-base shape comparison
  attention_probe.py      # SDPA / kitchen / sage timing + error on H3 geometry
  build_workflow.py       # generate a ComfyUI workflow from a declarative node list
  benchmark_loop.py       # restart -> submit -> sample peaks -> parse phases -> CSV
workflows/
  H3_Quality.json  20 steps, no LoRA, res_multistep   (official baseline)
  H3_Balanced.json  8 steps, turbo LoRA @1.0, euler
  H3_Fast.json      4 steps, turbo LoRA @1.0, euler
evidence/
  results.csv       # 14 runs × 33 fields (peak VRAM/RAM, phase timings, crash flags)
  old_vs_new.png    # the prompt fix, same seed, same timestamps
  dense_grid.png    # subject dropout at 7.0 s
  steps_8_vs_20.png # 8 steps vs 20 steps
  encoder_isolation.png  # same frames re-encoded at two bitrates
videos/
  01-...mp4         # before the prompt fix: subject drops at ~7 s
  02-...mp4         # after the prompt fix, 20 steps
  03-...mp4         # after the prompt fix, 8 steps, VHS crf=12 (with audio)
docs/
  publish-notes.md      # repo description / topics / release notes / share text
  working-report-cn.md  # raw working notes (contains retracted conclusions)
```

## Honest status

**Verified:** everything in sections 1–5 has a reproduction above, and 1, 3, 4 were
confirmed with byte-level or same-seed single-variable comparisons.

**Not verified / open:**

- **RTX 4060 Laptop (sm_89) is now measured** (2026-09): at 4 steps / 5 s / 768x1024,
  pytorch attention 403.8 s, comfy-kitchen 270.1 s, sage 263.0 s.
  **kitchen beats pytorch by 1.5x on the 4060 too**, so the recommendation does not
  need to be split by architecture. Still unmeasured there: 8-step and 20-step timings.
  That machine's commit limit was only 22.14 GB (29% below this one), so A1's absolute
  value may be partly polluted by pagefile growth.
- **No claim that the included workflows are optimal.** They are a reasonable,
  evidence-aligned starting point.
- The 8-vs-20 comparison in §6 is confounded by the encoder difference and rests on a
  single seed. A clean test needs both at the same encoder and several seeds.
- `ref_image_size` has a `max` setting that the node's own tooltip says gives "best identity
  fidelity" at "several times slower". **Untested here.**
- The 9:16 route (768×1344) was abandoned for memory reasons, not because it is wrong —
  with a larger pagefile it may well be the better resolution.
- Long clips: everything measured was 5 s or 10 s. The official trained range is ~124–362
  frames and nothing here probes beyond 243.
- This session contained several of my own retracted conclusions (Sol-Attn "lossless",
  a mis-attributed TE-Speed quote, three wrong CRF "verifications", one wrong timing
  extrapolation, one wrong crash diagnosis). Treat anything not explicitly labelled
  *measured* as suspect.

## License

MIT for the scripts. The workflow JSONs are generated artifacts; no model files are
included or redistributed.
