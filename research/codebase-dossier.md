# RL Infra Codebase Dossier

This file is a reusable evidence map for revising the blog. It records what has
actually been checked locally, at which commit, and which source files are good
anchors for later section-by-section edits.

## Working Rule

When the blog makes a systems claim, prefer a concrete software-engineering
statement over a decorative abstraction:

- Name the interface or invariant.
- Point to the file that enforces it.
- Say what can go wrong when the invariant is violated.
- Only use mathematical language when it adds precision to the engineering fact.

## Local Clone State

Sparse/partial clones live under `_external_repos/` and should not be committed.

| Repo | Local dir | Commit | First-pass status |
| --- | --- | --- | --- |
| verl-project/verl | `_external_repos/verl` | `7dc39fe` | core loop, DataProto, worker, checkpoint engine read |
| sgl-project/sglang | `_external_repos/sglang` | `cadfa2d` | weight update API, cache flush, memory release, prefix cache read |
| THUDM/slime | `_external_repos/slime` | `6961f59` | train loop, rollout manager, sample contract, R3/OPD read |
| radixark/miles | `_external_repos/miles` | `ae083c5` | train loop, async loop, rollout package, monitoring, R3/TIS read |
| NVIDIA/Megatron-LM | `_external_repos/Megatron-LM` | `3fb34c6` | native RL stack, GRPO loss, sequence packing read |
| vllm-project/vllm | `_external_repos/vllm` | `8de5cab` | block pool, KV manager, scheduler pause/sleep/wake read |
| inclusionAI/AReaL | `_external_repos/AReaL` | `4aef98d` | workflow/controller/staleness/weight-update paths partially read |
| OpenRLHF/OpenRLHF | `_external_repos/OpenRLHF` | `c3188af` | async PPO actors, vLLM lock, weight broadcast read |
| NovaSky-AI/SkyRL | `_external_repos/SkyRL` | `ccc181e` | README, SkyRL-Train, generator contract, masking read |
| nvidia-cosmos/cosmos-rl | `_external_repos/cosmos-rl` | `9438dc5` | README, reward service, configs, WFM/VLA paths sampled |
| RLinf/RLinf | `_external_repos/RLinf` | `231e772` | README, placement/distribution docs, scheduler/backend paths sampled |
| triton-lang/triton | `_external_repos/triton` | `e283e01` | README and compiler/runtime docs sampled |
| tile-ai/tilelang | `_external_repos/tilelang` | `b61e19e` | README and tensor-check compiler docs sampled |
| NVIDIA/cuda-python | `_external_repos/cuda-python` | `79809a3` | README sampled |
| NVIDIA/TensorRT | `_external_repos/TensorRT` | `5302b28` | README, Python bindings, samples sampled |
| NVIDIA/TensorRT-LLM | `_external_repos/TensorRT-LLM` | `827d0c1` | README, docs/source pointers, custom op paths sampled |
| black-forest-labs/flux | `_external_repos/flux` | `802fb47` | README/model docs sampled |
| black-forest-labs/flux2 | `_external_repos/flux2` | `50fe516` | README/model docs sampled |
| Wan-Video/Wan2.2 | `_external_repos/Wan2.2` | `42bf4cf` | README/inference overview sampled |
| zhaochenyang20/Awesome-ML-SYS-Tutorial | `_external_repos/Awesome-ML-SYS-Tutorial` | `14e1c9e` | RLHF/slime/verl/SGLang tutorial sources sampled |

All repos listed in the blog reading plan have now been cloned sparsely. Depth
is not uniform: the first eight RL-infra repos are read at code-anchor level;
the remaining repos are first-pass anchors and should be deepened only when a
section actually depends on them.

## High-Value Technical Themes

### 1. Weight Sync Is a Consistency Protocol

Useful blog replacement for abstract language:

"Updating rollout weights is not just copying tensors. It is a protocol: stop or
drain generation, transfer weights through one of several physical paths, then
invalidate any prefix/KV state that was computed under the old policy."

Anchors:

- SGLang exposes four update paths:
  - `_external_repos/sglang/python/sglang/srt/entrypoints/engine.py`
  - `_external_repos/sglang/python/sglang/srt/entrypoints/http_server.py`
  - `_external_repos/sglang/python/sglang/srt/managers/scheduler_components/weight_updater.py`
- `WeightUpdater.flush_cache_after_weight_update` asserts cache flush success
  after a requested update.
- slime/Miles wrap the SGLang endpoints with higher-level train/rollout
  orchestration:
  - `_external_repos/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
  - `_external_repos/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
  - `_external_repos/miles/miles/backends/sglang_utils/sglang_engine.py`
- verl abstracts the same problem as `CheckpointEngine`:
  - `_external_repos/verl/verl/checkpoint_engine/base.py`
  - `_external_repos/verl/verl/workers/engine_workers.py`

Editorial implication:

The "functor" framing in the current blog is much weaker than simply explaining
the API contract and failure modes: stale prefix cache, weight/version mismatch,
wrong rank mapping, and request-in-flight mutation.

### 2. Rollout Samples Need a Schema, Not Just a List of Tokens

Useful blog replacement:

"A sample is an interoperability object between the inference engine, reward
logic, and training engine. The hard part is preserving enough metadata for the
training loss to be correct."

Anchors:

- verl `DataProto`:
  - `_external_repos/verl/verl/protocol.py`
  - `DataProto` has tensor batch, non-tensor batch, and meta info.
  - `union` is a field-merge operation used by the controller loop.
- slime sample contract:
  - `_external_repos/slime/slime/utils/types.py`
  - `_external_repos/slime/slime/ray/rollout.py`
- Miles sample contract:
  - `_external_repos/miles/miles/utils/types.py`
  - `_external_repos/miles/miles/ray/rollout.py`
  - important fields include `rollout_log_probs` and `rollout_routed_experts`.
- Megatron native RL data prep:
  - `_external_repos/Megatron-LM/megatron/rl/rl_utils.py`
  - `_external_repos/Megatron-LM/megatron/rl/sequence_packing_utils.py`

Editorial implication:

This is a good place to teach mathematicians real engineering: schemas,
semantic interoperability, and contracts. It is also where the mathematical loss
meets messy systems state.

### 3. Prefix/KV Cache Is a Versioned Resource

Useful blog replacement:

"A cache entry is only valid relative to the model weights, request state, and
cache lifetime discipline under which it was created."

Anchors:

- SGLang radix/unified cache:
  - `_external_repos/sglang/python/sglang/srt/mem_cache/base_prefix_cache.py`
  - `_external_repos/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py`
  - `_external_repos/sglang/python/sglang/srt/managers/schedule_policy.py`
- SGLang update path flushes cache after weight updates:
  - `_external_repos/sglang/python/sglang/srt/managers/scheduler_components/weight_updater.py`
- vLLM comparison points:
  - `_external_repos/vllm/vllm/v1/core/block_pool.py`
  - `_external_repos/vllm/vllm/v1/core/kv_cache_manager.py`
  - `_external_repos/vllm/vllm/v1/core/sched/scheduler.py`
  - `_external_repos/vllm/vllm/v1/engine/core.py`

Editorial implication:

This section can be made concrete by explaining `inc_lock_ref`, `dec_lock_ref`,
`reset_prefix_cache`, and scheduler pause semantics.

### 4. Async RL Introduces Off-Policy Accounting

Useful blog replacement:

"Async rollout is a throughput optimization that changes the statistical object
being optimized unless the system records policy versions and corrects or bounds
staleness."

Anchors:

- Miles async loop:
  - `_external_repos/miles/train_async.py`
  - next rollout is launched before current training completes.
  - `update_weights_interval` controls how often rollout weights are refreshed.
- Miles monitoring:
  - `_external_repos/miles/docs/user-guide/monitoring.md`
  - `train_rollout_logprob_abs_diff` is a train/inference drift metric.
- Miles loss/correction:
  - `_external_repos/miles/miles/backends/training_utils/loss_hub/losses.py`
  - `_external_repos/miles/miles/backends/training_utils/loss_hub/corrections.py`
- Megatron native RL:
  - `_external_repos/Megatron-LM/megatron/rl/rl_utils.py`
  - GRPO loss has old logprobs, inference logprobs, truncation, and IS-style correction.
- AReaL:
  - `_external_repos/AReaL/areal/trainer/ppo/actor.py`
  - logs version staleness metrics.
  - example configs use `max_head_offpolicyness`.

Editorial implication:

The right explanation is not "async is a categorical buffer/fold pattern." The
real technical lesson is: throughput creates policy lag, policy lag must be
measured, and the loss/metrics must know which policy generated the data.

### 5. Multi-Turn RL Is an Environment/Workflow Abstraction

Useful blog replacement:

"Multi-turn RL turns generation into a workflow: model action, tool/environment
step, observation encoding, loss mask update, termination check, repeat."

Anchors:

- slime/Miles custom rollout hooks:
  - `_external_repos/slime/slime/ray/rollout.py`
  - `_external_repos/miles/miles/ray/rollout.py`
  - `_external_repos/miles/examples/geo3k_vlm_multi_turn/rollout.py`
- Miles turn/sample merging:
  - `_external_repos/miles/miles/rollout/generate_utils/sample_utils.py`
  - `_external_repos/miles/miles/rollout/generate_utils/tool_call_utils.py`
- AReaL workflow abstraction:
  - `_external_repos/AReaL/docs/en/reference/rollout_workflow.md`
  - `_external_repos/AReaL/areal/api/workflow_api.py`
  - `_external_repos/AReaL/examples/multi_turn_math/gsm8k_rl_mt.py`
- SkyRL generator abstraction:
  - `_external_repos/SkyRL/skyrl/train/generators/base.py`
  - `_external_repos/SkyRL/skyrl/train/generators/utils.py`
  - `_external_repos/SkyRL/skyrl-train/README.md`

Editorial implication:

This is a strong example for software engineers: define a narrow workflow
interface, then let environments vary behind it.

### 6. Low Precision and MoE Need Replay/Drift Checks

Useful blog replacement:

"Low-precision rollout is not merely an optimization. It can change routing,
logprobs, and therefore the training target unless the framework records and
replays the relevant state or monitors the mismatch."

Anchors:

- slime R3:
  - `_external_repos/slime/slime/utils/routing_replay.py`
  - `_external_repos/slime/slime/utils/types.py`
- Miles R3:
  - `_external_repos/miles/miles/utils/replay_base.py`
  - `_external_repos/miles/docs/advanced/fp8-low-precision.md`
  - `_external_repos/miles/docs/advanced/int4-qat.md`
  - `_external_repos/miles/docs/advanced/miles-router.md`
- Miles train/infer mismatch metric:
  - `_external_repos/miles/docs/user-guide/monitoring.md`
  - `_external_repos/miles/miles/backends/training_utils/loss_hub/losses.py`

Editorial implication:

This is one of the best places to preserve mathematical substance: routing
stability and logprob drift are real quantities, not decorative language.

## Repo Notes

### verl

Best framing: programming model / control-flow separation.

Anchors:

- `_external_repos/verl/docs/hybrid_flow.rst`
- `_external_repos/verl/verl/protocol.py`
- `_external_repos/verl/verl/trainer/ppo/ray_trainer.py`
- `_external_repos/verl/verl/workers/engine_workers.py`
- `_external_repos/verl/verl/checkpoint_engine/base.py`

Use for:

- control flow vs computation flow
- `DataProto` as a data envelope
- worker method dispatch as a contract
- checkpoint engine as weight-transfer abstraction
- hybrid engine as an enforced architectural decision

Be careful:

- Do not oversell `DataProto.union` as a deep mathematical structure. It is
  mostly a disciplined merge operation over a batch envelope.

### SGLang

Best framing: inference substrate with explicit control-plane endpoints.

Anchors:

- `_external_repos/sglang/python/sglang/srt/entrypoints/http_server.py`
- `_external_repos/sglang/python/sglang/srt/entrypoints/engine.py`
- `_external_repos/sglang/python/sglang/srt/managers/tokenizer_control_mixin.py`
- `_external_repos/sglang/python/sglang/srt/managers/scheduler_components/weight_updater.py`
- `_external_repos/sglang/python/sglang/srt/mem_cache/base_prefix_cache.py`
- `_external_repos/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py`

Use for:

- four weight update APIs
- cache flush as consistency obligation
- release/resume memory occupation
- prefix-cache lock/reference discipline

Be careful:

- Current files differ from some older blog paths. Fact-check any reference to
  `scheduler_update_weights_mixin.py` against current SGLang.

### slime

Best framing: clean Megatron + SGLang + Ray RL loop and extension points.

Anchors:

- `_external_repos/slime/train.py`
- `_external_repos/slime/slime/ray/rollout.py`
- `_external_repos/slime/slime/ray/placement_group.py`
- `_external_repos/slime/slime/utils/types.py`
- `_external_repos/slime/slime/utils/routing_replay.py`
- `_external_repos/slime/slime/utils/arguments.py`
- `_external_repos/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`

Use for:

- five-phase synchronous loop
- user-provided rollout function
- sample fields and train-data conversion
- OPD mode checks
- R3 stage machine

### Miles

Best framing: production hardening of the slime-style stack.

Anchors:

- `_external_repos/miles/train.py`
- `_external_repos/miles/train_async.py`
- `_external_repos/miles/miles/ray/rollout.py`
- `_external_repos/miles/miles/backends/sglang_utils/sglang_engine.py`
- `_external_repos/miles/miles/backends/megatron_utils/actor.py`
- `_external_repos/miles/miles/backends/training_utils/loss_hub/losses.py`
- `_external_repos/miles/miles/backends/training_utils/loss_hub/corrections.py`
- `_external_repos/miles/docs/user-guide/monitoring.md`
- `_external_repos/miles/docs/advanced/fault-tolerance.md`
- `_external_repos/miles/docs/advanced/p2p-weight-transfer.md`

Use for:

- async rollout overlap
- update-weight intervals
- health/fault tolerance
- P2P/disaggregated weight transfer
- train/rollout logprob mismatch monitoring
- TIS/ICE corrections

Be careful:

- The current repo has its own `miles/` package. Blog text that treats Miles as
  a thin `slime/` directory fork should be softened or updated.

### Megatron-LM

Best framing: the strongest math-meets-systems anchor.

Anchors:

- `_external_repos/Megatron-LM/train_rl.py`
- `_external_repos/Megatron-LM/megatron/rl/rl_utils.py`
- `_external_repos/Megatron-LM/megatron/rl/sequence_packing_utils.py`

Use for:

- native GRPO loss
- old/current/inference logprob roles
- importance/truncation corrections
- sequence packing with attention leakage prevention
- policy/KV epoch and staleness statistics

### vLLM

Best framing: block paging and scheduler semantics as a contrast to SGLang.

Anchors:

- `_external_repos/vllm/vllm/v1/core/block_pool.py`
- `_external_repos/vllm/vllm/v1/core/kv_cache_manager.py`
- `_external_repos/vllm/vllm/v1/core/sched/scheduler.py`
- `_external_repos/vllm/vllm/v1/engine/core.py`
- `_external_repos/vllm/vllm/v1/worker/gpu_worker.py`

Use for:

- block pool and KV cache allocation
- reset prefix cache
- scheduler pause
- sleep/wake memory semantics

### AReaL

Best framing: workflow/controller architecture and explicit staleness control.

Anchors:

- `_external_repos/AReaL/docs/en/reference/rollout_workflow.md`
- `_external_repos/AReaL/docs/en/reference/agent_workflow.md`
- `_external_repos/AReaL/areal/api/workflow_api.py`
- `_external_repos/AReaL/areal/infra/controller/rollout_controller.py`
- `_external_repos/AReaL/areal/infra/controller/train_controller.py`
- `_external_repos/AReaL/areal/trainer/ppo/actor.py`
- `_external_repos/AReaL/areal/experimental/engine/archon_weight_sync.py`
- `_external_repos/AReaL/areal/experimental/weight_update/awex/sglang_adapter.py`

Use for:

- `RolloutWorkflow.arun_episode`
- grouped/concurrent rollout
- `max_head_offpolicyness`
- version staleness metrics
- async/NCCL weight update paths

Be careful:

- The blog's `_PendingWeightUpdateBucket` claim needs more fact-checking against
  the current repo. Current code shows async broadcasts and weight-update
  adapters, but not that exact class name in the first pass.

### OpenRLHF

Best framing: baseline async PPO/Ray framework with explicit vLLM synchronization.

Anchors:

- `_external_repos/OpenRLHF/openrlhf/trainer/ray/vllm_worker_wrap.py`
- `_external_repos/OpenRLHF/openrlhf/trainer/ppo_trainer_async.py`
- `_external_repos/OpenRLHF/examples/scripts/train_ppo_ray_hybrid_engine.sh`
- `_external_repos/OpenRLHF/examples/scripts/train_dapo_ray_hybrid_engine.sh`

Use for:

- `VLLMLock`
- partial rollout pause/resume around broadcast
- CUDA IPC / NCCL weight updates
- sleep-enabled hybrid engine scripts

### SkyRL

Best framing: modular RL stack for agentic and multi-turn tasks.

Anchors:

- `_external_repos/SkyRL/README.md`
- `_external_repos/SkyRL/skyrl-train/README.md`
- `_external_repos/SkyRL/skyrl/train/generators/base.py`
- `_external_repos/SkyRL/skyrl/train/generators/utils.py`
- `_external_repos/SkyRL/examples/train_integrations/harbor/harbor_generator.py`

Use for:

- backend abstraction: FSDP/Megatron training, vLLM/SGLang/OpenAI-compatible inference
- custom `Generator.generate()` interface for environments and agent harnesses
- multi-turn masking: user/tool messages get loss mask 0, assistant tokens get loss mask 1
- step-wise training for fully on-policy multi-turn RL
- colocated vs disaggregated execution-plan knobs

Be careful:

- The repo is actively reorganizing into `skyrl/`; cite current paths and avoid
  presenting older `skyrl-train` paths as the permanent API.

### cosmos-rl

Best framing: Physical AI RL framework where the modality changes the systems problem.

Anchors:

- `_external_repos/cosmos-rl/README.md`
- `_external_repos/cosmos-rl/reward_service/README.md`
- `_external_repos/cosmos-rl/cosmos_rl/rollout/schema.py`
- `_external_repos/cosmos-rl/cosmos_rl/policy/config/__init__.py`
- `_external_repos/cosmos-rl/cosmos_rl/policy/model/wfm/models/t2v_model.py`
- `_external_repos/cosmos-rl/configs/cosmos-predict2-5/`

Use for:

- policy/rollout replica specialization
- controller messages such as `weight-sync`, `rollout`, and `evaluate`
- reward service as a separate FastAPI + Redis-backed system
- diffusion/video rollouts where outputs are not just token strings
- context/sequence/FSDP/pipeline/tensor parallelism in WFM/VLA settings
- async staleness knobs such as allowed outdated rollout steps

Be careful:

- The blog currently says "DDRL" very strongly. Current config paths show
  `ddrl` examples, but a section rewrite should explain the concrete trainer,
  rollout schema, and reward-service mechanics instead of only naming the
  algorithm.

### RLinf

Best framing: broad scheduler/framework for agentic and embodied RL.

Anchors:

- `_external_repos/RLinf/README.md`
- `_external_repos/RLinf/docs/source-en/index.rst`
- `_external_repos/RLinf/docs/source-zh/index.rst`
- `_external_repos/RLinf/docs/source-zh/rst_source/start/distribute.rst`
- `_external_repos/RLinf/rlinf/scheduler/`
- `_external_repos/RLinf/rlinf/workers/rollout/`
- `_external_repos/RLinf/rlinf/envs/world_model/`

Use for:

- M2Flow as logical workflow vs physical scheduling separation
- component placement: shared `actor,rollout: all` vs split GPU ranges
- support matrix for VLA, WAM, VLM, agentic tasks, PPO/GRPO/SAC/Async PPO
- Megatron + SGLang/vLLM stack for large-scale training
- embodied/world-model examples such as Wan/OpenSora-based environments

Be careful:

- RLinf is wide. For the blog, use it as a scheduler/placement and domain-extension example, not as a place to prove every RLHF primitive.

### Triton

Best framing: language/compiler for writing custom deep-learning kernels with
higher productivity than CUDA.

Anchors:

- `_external_repos/triton/README.md`
- `_external_repos/triton/python/triton/`
- `_external_repos/triton/lib/`
- `_external_repos/triton/docs/`

Use for:

- custom kernel DSL
- compiler/debugging knobs
- MLIR/LLVM compilation pipeline
- attention/matmul-style primitive optimization

Be careful:

- Triton is a kernel tool, not an RL framework. Keep it in a lower layer of the stack.

### TileLang

Best framing: TVM-backed tile-level kernel DSL with richer host-side contracts.

Anchors:

- `_external_repos/tilelang/README.md`
- `_external_repos/tilelang/docs/compiler_internals/tensor_checks.md`
- `_external_repos/tilelang/examples/`
- `_external_repos/tilelang/src/`

Use for:

- GEMM/FlashAttention/LinearAttention examples
- shape/dtype/device checks inserted into generated host stubs
- symbolic shape constraints and ABI stability
- Z3/SMT-backed verification as a compiler-internals point

Be careful:

- The technical value is compiler/runtime validation, not a broad claim that it
  magically proves GPU kernels correct.

### cuda-python

Best framing: Python access layer to CUDA APIs.

Anchors:

- `_external_repos/cuda-python/README.md`
- `_external_repos/cuda-python/cuda_core/`
- `_external_repos/cuda-python/cuda_bindings/`

Use for:

- distinction between Pythonic CUDA runtime access and low-level CUDA C bindings
- why frameworks should avoid homegrown CUDA wrappers when official bindings exist

Be careful:

- It belongs in the "layer beneath" section, not the RL-infra section.

### TensorRT

Best framing: optimized inference engine/toolchain, plus plugins and samples.

Anchors:

- `_external_repos/TensorRT/README.md`
- `_external_repos/TensorRT/python/`
- `_external_repos/TensorRT/plugin/`
- `_external_repos/TensorRT/samples/`

Use for:

- engine build/runtime distinction
- plugin API as an extension mechanism
- explicit quantization and strong typing as API direction
- why an optimized inference engine can be less convenient for live RL weight updates

Be careful:

- TensorRT OSS is a subset of the full GA product. Avoid treating the GitHub repo
  as the entire TensorRT implementation.

### TensorRT-LLM

Best framing: high-performance LLM/visual-gen inference stack.

Anchors:

- `_external_repos/TensorRT-LLM/README.md`
- `_external_repos/TensorRT-LLM/docs/`
- `_external_repos/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/custom_ops/attention/triton_paged_attention.py`
- `_external_repos/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/`

Use for:

- specialized kernels, runtime optimizations, and parallelism for inference
- prefill-decode disaggregation, KV cache, prefix caching, CUDA graphs
- visual generation support and quantized models
- contrast with SGLang/vLLM: strong inference performance, less naturally used as a mutable rollout engine in open RL stacks

Be careful:

- Do not claim it is "not RL-friendly" as an absolute. Say the common open RL
  stacks in this blog expose weight-update/control-plane hooks through SGLang or
  vLLM more directly.

### FLUX / FLUX.2

Best framing: visual generative model examples, not RL infrastructure.

Anchors:

- `_external_repos/flux/README.md`
- `_external_repos/flux2/README.md`
- `_external_repos/flux2/docs/flux2_klein_kv_cache.md`

Use for:

- diffusion/flow-matching model family context
- open-weight image generation and editing
- FLUX.2 [klein] model family and KV-cache/latency tradeoffs

Be careful:

- Do not spend many words here unless the blog section is about visual RL or
  diffusion rollouts. These repos are motivating targets, not core infra.

### Wan2.2

Best framing: open video diffusion/MoE model target for world-model or video RL.

Anchors:

- `_external_repos/Wan2.2/README.md`
- `_external_repos/Wan2.2/generate.py`
- `_external_repos/Wan2.2/wan/`

Use for:

- MoE video diffusion model
- 720P/24fps TI2V 5B model and multi-GPU inference
- why video rollouts stress context, memory, and reward-service design

Be careful:

- Wan2.2 itself is an inference/model repo. For RL fine-tuning claims, use RLinf
  or cosmos-rl as the infra source and Wan as the model/world-model target.

### Awesome-ML-SYS-Tutorial

Best framing: secondary synthesis source and useful Chinese commentary.

Anchors:

- `_external_repos/Awesome-ML-SYS-Tutorial/rlhf/slime/code-walk-through/readme.md`
- `_external_repos/Awesome-ML-SYS-Tutorial/rlhf/slime/vlm-multi-turn/readme.md`
- `_external_repos/Awesome-ML-SYS-Tutorial/rlhf/slime/mismatch/blog-cn.md`
- `_external_repos/Awesome-ML-SYS-Tutorial/rlhf/partial-rollout/readme.md`
- `_external_repos/Awesome-ML-SYS-Tutorial/rlhf/verl/server-based/veRL-server-based-rollout.md`
- `_external_repos/Awesome-ML-SYS-Tutorial/sglang/scheduler-evolution/SGLang Scheduler 技术变迁.md`

Use for:

- background and intuition
- discovering code paths and failure modes
- cross-checking Chinese terminology

Be careful:

- Treat it as commentary, not primary source. Blog claims should ultimately point
  to actual framework code or official docs when possible.

## Blog Claims To Revisit

- Replace "transport functor" with "weight-transfer interface plus cache
  invalidation contract."
- Replace "categorical product" with "composition of two independent tactics:
  colocation handles resource utilization; Megatron parallelism handles model
  scale."
- Replace "buffer-then-fold" with concrete performance reasoning: batch small
  transfers, avoid doing an O(n) operation inside an O(n) loop, and bound memory.
- Re-check AReaL `_PendingWeightUpdateBucket`; do not keep that name unless it
  exists in the current target revision or we explicitly cite an older commit.
- Re-check current SGLang file paths before linking.
- Re-check Miles current package names before presenting it as a direct slime fork.
- Rephrase SkyRL around `Generator` and `skyrl-gym`, not just "multi-turn agents."
- Rephrase cosmos-rl around policy/rollout replicas, reward service, and diffusion/video rollout schemas.
- Use TensorRT/TensorRT-LLM as inference-stack context, not as direct RL framework evidence.
- Use FLUX/Wan as model-domain motivation; use cosmos-rl/RLinf for actual RL infra claims.
