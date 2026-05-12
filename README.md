# RL Infrastructure for the Mathematically Inclined

> A field survey of reinforcement-learning post-training systems for mathematicians and RL theorists who want engineering literacy. Six engineering primitives, three kernel-level DSLs, Megatron-LM, quantization, and the nine production frameworks compared on five axes.

**🌐 Live site:** https://chenggong-zhang.github.io/RL_infra/

The premise: if you're a theorist reading the DeepSeek-V3 report or studying GRPO, the engineering choices in production RL systems are quietly encoding mathematical decisions about your algorithm — which approximations are bounded, which biases are corrected, which invariants hold by construction. This blog gives you the vocabulary to read the source code of [verl](https://github.com/verl-project/verl), [SGLang](https://github.com/sgl-project/sglang), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), and [Triton](https://github.com/triton-lang/triton), and recognize what's mathematically interesting about it.

It is a survey, not a tutorial. It is read in one sitting (~30 minutes), not section by section over weeks.

---

## What's in the survey

The article is structured as a single long-form blog post. Section headings:

- **Why this matters to a theorist** — the case for caring about systems
- **The skeleton** — the five-stage cycle and the three pillars every framework instantiates
- **The central tension** — why training and inference fight on the same hardware
- **Six engineering primitives** — the engineered solutions:
  - ① The hybrid engine (训推一体)
  - ② Memory choreography
  - ③ Zero-copy weight synchronization
  - ④ Four `update_weights_from_*` paths, one verb
  - ⑤ RadixAttention × GRPO — algebraic composition
  - ⑥ Async training and the staleness tradeoff
- **The layer beneath: CUDA, Triton, TileLang** — kernel-level DSLs
- **The training backbone: Megatron-LM** — 5D parallelism (TP × PP × DP × EP × CP)
- **Quantization and the numerical-alignment problem** — FP8, INT4, MXFP4; the MoE routing divergence
- **Reading real code** — verl's `fit()` loop and AReaL's async pattern
- **The framework landscape** — 9 frameworks on 5 axes
- **Beyond chat** — multi-turn agents, embodied AI, world models, the TPU detour
- **The scaling chain** — what breaks at each cluster tier
- **Pitfalls (踩坑录)** — six production failure modes from Chenyang's tutorial
- **A reading list for the theoretically inclined**

---

## Six SVG diagrams

The survey ships six SVG diagrams that visualize the design principles. They live in [`diagrams/`](diagrams/).

| # | File | Section | What it visualizes |
|---|------|---------|--------------------|
| 1 | [`01-rollout-cycle.svg`](diagrams/01-rollout-cycle.svg) | The skeleton | The five-stage loop |
| 2 | [`02-three-pillars.svg`](diagrams/02-three-pillars.svg) | The skeleton | Training + Inference + Orchestrator |
| 3 | [`03-hybrid-engine-timeline.svg`](diagrams/03-hybrid-engine-timeline.svg) | Memory choreography | One GPU, two roles, over time |
| 4 | [`04-weight-update-paths.svg`](diagrams/04-weight-update-paths.svg) | Four update paths | Four transport topologies |
| 5 | [`05-radix-tree-grpo.svg`](diagrams/05-radix-tree-grpo.svg) | RadixAttention × GRPO | Hash cache vs radix tree for group rollouts |
| 6 | [`06-scaling-chain.svg`](diagrams/06-scaling-chain.svg) | The scaling chain | What breaks at 8 → 10K GPUs |

---

## Why this might be useful to a mathematician

You're reading a paper and wondering:

- *"Why does GRPO actually work better than PPO at scale?"* — Because group sampling enables prefix-cache reuse multiplicatively, not because the variance reduction math is fundamentally cleaner. The system advantage is doing the heavy lifting.
- *"Is on-policy in the empirical evaluation really on-policy?"* — In every framework with partial rollout, no. The off-policy ratio is a hyperparameter you should be checking.
- *"How can an MoE model train stably in FP8?"* — Only with explicit routing replay (R3). Without it, two precision regimes diverge on top-k routing decisions and your gradient becomes noise. Miles' contribution is precisely this invariant.
- *"What's actually different about cosmos-rl's diffusion RL?"* — Standard PPO doesn't apply to a diffusion model; cosmos-rl invented DDRL (replaces KL with reward + standard diffusion loss) and ships custom 6D parallelism for 100K-token video sequences.
- *"Why does AReaL claim 2.77×?"* — One keyword: `async_op=True` on every NCCL broadcast, paired with memory-bounded bucketing of the queued operations.

The survey answers each of these in its body. The point isn't the specific answer; it's that engineering choices are answering questions you'd otherwise ask of the math.

---

## Source repos surveyed

19 repos read end-to-end. Frameworks first, then the layer beneath, then the rollout targets:

**RL frameworks (9):**
1. [verl-project/verl](https://github.com/verl-project/verl) — the hybrid-engine reference implementation
2. [radixark/miles](https://github.com/radixark/miles) — bit-identical FP8/INT4 + R3 routing replay
3. [THUDM/slime](https://github.com/THUDM/slime) — on-policy distillation, GLM-5.1
4. [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL) — multi-turn agents, SA-SWE-32B
5. [nvidia-cosmos/cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) — physical AI, diffusion world models
6. [RLinf/RLinf](https://github.com/RLinf/RLinf) — M2Flow scheduler, embodied + agentic
7. [sgl-project/sglang](https://github.com/sgl-project/sglang) — the inference substrate
8. [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) — fully-async RL
9. [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — the OG framework

**The layer beneath (4):**
10. [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — 5D parallelism training backbone
11. [triton-lang/triton](https://github.com/triton-lang/triton) — Python DSL for GPU kernels
12. [tile-ai/tilelang](https://github.com/tile-ai/tilelang) — newer TVM-based DSL with Z3 verification
13. [NVIDIA/cuda-python](https://github.com/nvidia/cuda-python) — Python bindings to CUDA proper

**Adjacent infrastructure (3):**
14. [vllm-project/vllm](https://github.com/vllm-project/vllm) — SGLang's primary competitor
15. [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) — the compilation framework
16. [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — optimized inference (not RL-friendly)

**Rollout targets (4):**
17. [black-forest-labs/flux](https://github.com/black-forest-labs/flux) · [flux2](https://github.com/black-forest-labs/flux2) — image diffusion
18. [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) — MoE video model
19. [Cambrian-MLLM TPU blog](https://cambrian-mllm.github.io/blog/tpu-training-experiments.html) — why TPU-native RL is unsolved

---

## Repository layout

```
RL_infra/
├── README.md                  # this file
├── index.html                 # the survey (single self-contained HTML, ~30 min read)
├── diagrams/                  # 6 SVG diagrams embedded in the survey
│   ├── 01-rollout-cycle.svg
│   ├── 02-three-pillars.svg
│   ├── 03-hybrid-engine-timeline.svg
│   ├── 04-weight-update-paths.svg
│   ├── 05-radix-tree-grpo.svg
│   └── 06-scaling-chain.svg
├── .github/workflows/pages.yml   # static deploy to GitHub Pages
├── .nojekyll                  # bypass Jekyll on Pages
├── .claude/settings.json      # project-shared Claude Code permissions
├── .gitignore
└── LICENSE
```

The site has no build step. `index.html` plus the SVGs in `diagrams/` are the entire deliverable.

### Local preview

```bash
open index.html

# Or serve over HTTP (SVG <img> links work properly):
python3 -m http.server -d . 8000
# then visit http://localhost:8000
```

### Deploy

Push to `main`. The workflow at [`.github/workflows/pages.yml`](.github/workflows/pages.yml) uploads the repo root as the GitHub Pages artifact.

---

## Acknowledgements

The connective analysis — especially "what's mathematically interesting" — synthesizes [Chenyang Zhao's Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial) with primary-source reading of the repos above. Read his tutorial directly if you can read Chinese; it's the canonical reference for the field.

Visual styling after [just-the-docs](https://github.com/just-the-docs/just-the-docs).

The actual engineering was done by the framework teams at ByteDance Seed (verl), THUDM (slime), Radix Ark (Miles), Berkeley NovaSky (SkyRL), NVIDIA Cosmos (cosmos-rl), the RLinf team, the SGLang and vLLM communities, Inclusion AI (AReaL), OpenRLHF, NVIDIA (Megatron-LM, cuda-python), the Triton and TileLang teams. This survey just connects the dots and points you at the source.

---

## License

MIT — see [LICENSE](LICENSE).
