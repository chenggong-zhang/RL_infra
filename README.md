# RL Infra · A Course

> A pedagogical course on reinforcement-learning post-training infrastructure. **7 acts, 24 lessons**, six SVG diagrams, eight annotated code excerpts pulled live from production frameworks.

**🌐 Live site:** https://chenggong-zhang.github.io/RL_infra/

Most resources on RL infrastructure are either too high-level (marketing pages) or too low-level (one repo's source). This site sits in between — a structured course that builds your mental model lesson by lesson, with diagrams and real code at every step.

---

## The course at a glance

The course is structured as a story in **7 acts**. Each act builds on the previous. Each lesson asks a question, answers it, and points you to the next.

| Act | What you'll learn | Lessons |
|-----|-------------------|---------|
| **I. The Question** | Why RL post-training infra exists; the central tension | L1–3 |
| **II. The Universal Pattern** | The rollout cycle and the three pillars every framework instantiates | L4–6 |
| **III. Engineered Solutions** | The six primitives that solve the central tension | L7–12 |
| **IV. Reading Real Code** | Walk through the actual `fit()` loops | L13–15 |
| **V. Framework Landscape** | How the 9 frameworks differ on 5 axes | L16–19 |
| **VI. Beyond Chat** | Multi-turn agents, embodied AI, world models | L20–22 |
| **VII. Synthesis** | Scaling chain + lessons paid in pain | L23–24 |
| **Capstone** | Decision tree + 5-week reading plan + canonical literature | — |

---

## Visual diagrams

The course includes six SVG diagrams that visualize the design principles. They live in [`diagrams/`](diagrams/) and are embedded in the relevant lessons.

| # | Diagram | Lesson | What it shows |
|---|---------|--------|---------------|
| 1 | [`01-rollout-cycle.svg`](diagrams/01-rollout-cycle.svg) | L4 | The five-stage loop every framework instantiates |
| 2 | [`02-three-pillars.svg`](diagrams/02-three-pillars.svg) | L5 | Training engine + Inference engine + Orchestrator |
| 3 | [`03-hybrid-engine-timeline.svg`](diagrams/03-hybrid-engine-timeline.svg) | L8 | Memory choreography over time (训推一体) |
| 4 | [`04-weight-update-paths.svg`](diagrams/04-weight-update-paths.svg) | L10 | The four `update_weights_from_*` topologies |
| 5 | [`05-radix-tree-grpo.svg`](diagrams/05-radix-tree-grpo.svg) | L11 | RadixAttention × GRPO multiplicative composition |
| 6 | [`06-scaling-chain.svg`](diagrams/06-scaling-chain.svg) | L23 | What breaks at each cluster tier (8 → 10K GPUs) |

---

## The 24 lessons in detail

### Act I — The Question (Lessons 1–3)
- **L1. What does RL infrastructure actually solve?** — Why pretraining and inference primitives aren't enough.
- **L2. Two halves that fight** — The opposing-preferences problem: training wants TP/PP, rollout wants DP.
- **L3. Why this is harder than it looks** — The three pain points colocation creates.

### Act II — The Universal Pattern (Lessons 4–6)
- **L4. The rollout cycle** — Generate → Score → Filter → Train → Sync weights. (Figure 1)
- **L5. The three pillars** — Training engine + inference engine + orchestrator. (Figure 2)
- **L6. One step end-to-end** — The 18-step call graph for one PPO iteration.

### Act III — The Engineered Solutions (Lessons 7–12)
- **L7. The hybrid engine (训推一体)** — Colocate and serialize.
- **L8. Memory choreography** — `release_memory_occupation` / `resume_memory_occupation`. (Figure 3)
- **L9. The weight-sync handoff** — Handle tuples and CUDA IPC.
- **L10. Four `update_weights_from_*` paths** — One verb, four transports. (Figure 4)
- **L11. RadixAttention × GRPO** — Multiplicative composition. (Figure 5)
- **L12. Async & staleness** — Partial rollout, TIS/MIS, AReaL's bucket pattern.

### Act IV — Reading Real Code (Lessons 13–15)
- **L13. verl's `fit()` loop** — The canonical PPO orchestration.
- **L14. DataProto + SPMD dispatch** — One driver call → 100 GPUs running.
- **L15. AReaL vs OpenRLHF** — Async vs sync in one keyword.

### Act V — The Framework Landscape (Lessons 16–19)
- **L16. The 5 axes of choice** — What actually distinguishes frameworks.
- **L17. 9 frameworks compared** — verl, Miles, SLIME, SGLang, SkyRL, cosmos-rl, RLinf, OpenRLHF, AReaL.
- **L18. Family clusters** — LLM purists vs Agent platforms vs Domain extenders vs Substrate.
- **L19. Tour of the 9** — One paragraph each + the file to read first.

### Act VI — Beyond Chat (Lessons 20–22)
- **L20. Multi-turn agents and tool use** — SkyRL, SLIME, verl multi-turn.
- **L21. Embodied AI, VLA, world models** — cosmos-rl, RLinf, FLUX, Wan2.2.
- **L22. The TPU detour** — Why a TPU-native RL framework doesn't exist yet.

### Act VII — Synthesis (Lessons 23–24)
- **L23. The scaling chain** — What breaks at each tier from 8 GPUs to 10K+. (Figure 6)
- **L24. Pitfalls (踩坑录)** — The six lessons paid in pain.

### Capstone
- Decision table — choose your framework
- 5-week reading plan
- Canonical literature

---

## Source repos covered

15 repos read end-to-end:

1. [verl-project/verl](https://github.com/verl-project/verl) — the hybrid-engine reference implementation
2. [radixark/miles](https://github.com/radixark/miles) — bit-identical FP8/INT4 + R3 routing replay
3. [THUDM/slime](https://github.com/THUDM/slime) — on-policy distillation, GLM-5.1
4. [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL) — multi-turn agents, SA-SWE-32B
5. [nvidia-cosmos/cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) — physical AI, diffusion world models
6. [RLinf/RLinf](https://github.com/RLinf/RLinf) — M2Flow scheduler, embodied + agentic
7. [sgl-project/sglang](https://github.com/sgl-project/sglang) — the substrate
8. [vllm-project/vllm](https://github.com/vllm-project/vllm) — PagedAttention origin
9. [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) — fully-async RL
10. [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — the OG framework
11. [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) — the underlying compilation framework
12. [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — optimized inference (not RL-friendly)
13. [black-forest-labs/flux](https://github.com/black-forest-labs/flux) — image diffusion rollout target
14. [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2) — FLUX.2 (klein for fast inference)
15. [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) — video MoE rollout target
16. [Cambrian-MLLM TPU blog](https://cambrian-mllm.github.io/blog/tpu-training-experiments.html) — why TPU-RL is unsolved

---

## How to use this course

- **If you have 90 minutes:** Read Acts II and V (Lessons 4–6 and 16–19). Universal pattern + framework matrix.
- **If you have a weekend:** Read Acts I → III in order. By the end you'll understand the central tension and the six engineered primitives that resolve it.
- **If you have a week:** Read everything. End with the 5-week reading plan in the Capstone and pick one framework to go deeper into.

Each lesson is short (5–10 minutes), self-contained, and ends with a "Next →" link to the next lesson. No need to navigate manually.

---

## Repository layout

```
RL_infra/
├── README.md                  # this file
├── index.html                 # the 24-lesson course (single self-contained HTML)
├── diagrams/                  # 6 SVG diagrams referenced by the course
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
# Just open the HTML directly
open index.html

# Or serve over HTTP (recommended — SVG <img> links work properly)
python3 -m http.server -d . 8000
# then visit http://localhost:8000
```

### Deploy

Push to `main`. The workflow at [`.github/workflows/pages.yml`](.github/workflows/pages.yml) uploads the repo root as the GitHub Pages artifact. In **Settings → Pages → Build and deployment → Source**, choose **GitHub Actions**.

---

## Acknowledgements

- **Conceptual source**: [Chenyang Zhao's Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial) — the Chinese-language deep dive on SGLang, verl, and slime internals. The course's connective analysis, the "elegance of rollout" thesis, and the 踩坑录 lesson are all synthesized from Chenyang's writing. If you can read Chinese, read his tutorial directly.
- **Visual template**: [just-the-docs](https://github.com/just-the-docs/just-the-docs) — the Jekyll documentation theme whose look this course approximates.
- **Source authors**: the framework teams at ByteDance Seed (verl), THUDM (slime), Radix Ark (Miles), Berkeley NovaSky (SkyRL), NVIDIA Cosmos (cosmos-rl), the RLinf team, the SGLang/vLLM communities, Inclusion AI (AReaL), and OpenRLHF.

---

## License

MIT — see [LICENSE](LICENSE).
