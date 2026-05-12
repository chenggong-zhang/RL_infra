# RL Infra Learning Thesis

> An interactive learning map for reinforcement-learning post-training infrastructure — 9 frameworks, 15 repos read end-to-end, code excerpts annotated against the engineering principles they demonstrate.

**Live site:** https://chenggong-zhang.github.io/RL_infra/

A single-page HTML deep dive on the design of modern RL infrastructure. The thesis: every RL post-training framework is the same skeleton — a training engine + an inference engine + an orchestrator, stitched around a single cycle (generate → score → train → sync weights). Once you see that skeleton, every repo becomes legible.

## What the page covers

### Universal architecture
The 3-pillar diagram every framework instantiates, the rollout cycle written out as code, and the **5 axes of choice** that distinguish frameworks (training backend · inference backend · orchestration · placement · domain).

### Framework matrix
Nine frameworks compared across the five axes with each one's distinctive bet:

| Framework | Distinctive bet |
|---|---|
| [Miles](https://github.com/radixark/miles) | Bit-identical FP8 / INT4 across train↔infer; **R3 routing replay** |
| [SLIME](https://github.com/THUDM/slime) | On-policy distillation; rich math/science graders |
| [SGLang](https://github.com/sgl-project/sglang) | The picks-and-shovels: `update_weights_from_*`, RadixAttention, spec decoding |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | `skyrl-gym` + `skyrl-agent` + Tinker API |
| [cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) | Physical AI: WFM RL (DDRL); FP8/MXFP4; 6D parallelism |
| [RLinf](https://github.com/RLinf/RLinf) | Macro→Micro flow transformation: 2.43× throughput |
| [verl](https://github.com/verl-project/verl) | 3D-HybridEngine + handle-tuple zero-copy weight sync |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | The OG — now structurally limited by sync design |
| [AReaL](https://github.com/inclusionAI/AReaL) | Fully-async via `_PendingWeightUpdateBucket` |

### The elegance of rollout design
Eight design pillars distilled from [Chenyang Zhao's tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial): the opposing-preferences problem, hybrid engine pattern (训推一体), the four `update_weights_from_*` paths, NCCL static vs RDMA dynamic, RadixAttention × GRPO multiplicative composition, Engine vs Server duality, partial rollout + staleness, correctness by construction.

### Connected analysis — why this stack scales
An end-to-end walk through one RL step (18 numbered steps across 5 phases), six design decisions interrogated against their alternatives, and a **scaling chain** (~8 GPUs → 10000+ GPUs) showing what breaks first at each tier and which primitive saves you.

### Where the alternatives die
8 paths the community tried and abandoned, each with concrete cost numbers — monolithic RL (2-3× slower), strict on-policy (5 GPU-hours/GPU wasted per 1000 iters), CPU rollout, hash-based prefix cache, shared memory for weight sync, inference-engine logprobs for loss.

### Pitfalls — 踩坑录
Six concrete production failures from Chenyang's tutorial: BERT-era numerical drift, handle-tuple segfaults, NCCL hangs, memory choreography under colocate, off-policy ratio drift, mixed-precision routing divergence.

### Code patterns made concrete
Eight annotated real code excerpts (all linked to source on GitHub):
1. verl's `fit()` body — the `marked_timer` phase blocks
2. SGLang's four `update_weights_from_*` methods
3. `release_memory_occupation` — tag-based GPU memory choreography
4. verl's `DataProto` — the universal data envelope
5. verl's dispatch decorator — SPMD made invisible
6. AReaL's `_PendingWeightUpdateBucket` — `async_op=True`
7. OpenRLHF's blocking `broadcast_to_vllm` — the structural limit
8. RadixCache method signatures — reference-counted prefix sharing

### Rollout targets
What RL infra actually trains on: FLUX/FLUX.2 (image diffusion), Wan2.2 (video MoE), VLA models (π₀, OpenVLA, GR00T), and the [TPU detour](https://cambrian-mllm.github.io/blog/tpu-training-experiments.html) — why no TPU-native RL framework exists yet.

### Choosing a framework
A decision table — pick the first row that fits your use case.

## Source repos covered

15 repos read end-to-end:

1. [verl-project/verl](https://github.com/verl-project/verl)
2. [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2)
3. [black-forest-labs/flux](https://github.com/black-forest-labs/flux)
4. [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
5. [RLinf/RLinf](https://github.com/RLinf/RLinf)
6. [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)
7. [radixark/miles](https://github.com/radixark/miles)
8. [THUDM/slime](https://github.com/THUDM/slime)
9. [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)
10. [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
11. [sgl-project/sglang](https://github.com/sgl-project/sglang)
12. [vllm-project/vllm](https://github.com/vllm-project/vllm)
13. [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)
14. [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
15. [Cambrian-MLLM TPU blog](https://cambrian-mllm.github.io/blog/tpu-training-experiments.html)

## Page structure

- **Static HTML** — single `index.html`, no build step, no dependencies
- **Visual template** styled after [just-the-docs](https://github.com/just-the-docs/just-the-docs) — light theme, purple `#7253ed` accent, sidebar nav, sticky right-rail TOC
- **Source badges** above every code block linking to the exact file on GitHub
- **Study-resource boxes** at the end of each framework deep-dive with curated reading lists
- **Interactive** — clickable concept nodes, filterable by framework, live search

## Build & deploy

This site has no build step. The single `index.html` is the whole site.

Deployment is via GitHub Pages with a static-file workflow ([`.github/workflows/pages.yml`](.github/workflows/pages.yml)). The `.nojekyll` file at the root tells Pages to skip the Jekyll build pipeline.

### Local preview

Just open the file:

```bash
open index.html
# or serve over HTTP for relative-link testing:
python3 -m http.server -d . 8000
```

### Deploy

Push to `main`. The Pages workflow uploads the repo root as the deployment artifact. Then in **Settings → Pages → Build and deployment → Source**, choose **GitHub Actions**.

## Acknowledgements

- **Conceptual source**: [Chenyang Zhao's Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial) — the Chinese-language deep dives on SGLang, verl, and slime internals. Most of the "Connected analysis" and "Elegance of rollout" sections are synthesized from his writing.
- **Visual template**: [just-the-docs](https://github.com/just-the-docs/just-the-docs) — the Jekyll documentation theme whose look this page approximates.
- **Source repos**: all 15 above. The framework authors did the actual engineering; this page just connects the dots.

## License

MIT — see [LICENSE](LICENSE).
