# ngpt-nvfp4

This repository provides the code for the paper *Normalized Architectures are Natively 4-Bit*.

## Summary

The paper shows that nGPT, which constrains weights and hidden representations to the unit hypersphere, is inherently robust to NVFP4 4-bit training. This robustness comes from stronger constructive signal accumulation in dot products rather than lower local quantization noise, yielding higher SNR, a flatter loss landscape, and better learning-rate transfer from BF16 to NVFP4. Experiments on a 1.2B dense model and hybrid Mamba-Transformer MoE models up to 3B/30B show stable NVFP4 training with lower relative error while removing Randomized Hadamard Transforms and dynamic per-tensor scaling.

The `Megatron-LM` codebase was cloned from `https://github.com/NVIDIA/Megatron-LM` at commit `40d590d7f148c34cfe8cf6d473dd55812a61648a`.  
The `TransformerEngine` codebase was cloned from `https://github.com/NVIDIA/TransformerEngine` at commit `c188b533cc3721ca9c6bbfd26148f5cf60108c25`.  

## Build and install Transformer Engine wheels

Build the Transformer Engine wheel from the local `TransformerEngine` checkout:

```bash
cd TransformerEngine
NVTE_FRAMEWORK=pytorch pip wheel --no-build-isolation --no-deps -w dist .
```

This writes the built wheel to `TransformerEngine/dist/`, for example
`dist/transformer_engine-2.11.0+...whl`.

Install the built wheel into the active environment:

```bash
pip install --no-deps --force-reinstall dist/transformer_engine*.whl
```

## NVFP4 skip-amax toggles

`skip_amax` is an NVFP4 performance optimization that skips per-tensor amax computation during quantization. When enabled, TE does not launch the amax kernel and instead initializes amax to `1.0`, which gives a fixed FP4 E2M1 scale of `6.0 / 1.0 = 6.0`.

Correctness constraint: when `skip_amax=True`, the input data for that cast type must already be in the `[-1, 1]` range. If the data is outside that range, skipping amax can produce incorrect numerical results because the scale is no longer derived from the actual tensor maximum.

Global toggle for all NVFP4 cast types:

```bash
export NVTE_NVFP4_SKIP_AMAX=1
```

Equivalent programmatic toggle:

```python
from transformer_engine.common.recipe import NVFP4BlockScaling

recipe = NVFP4BlockScaling(skip_amax=True)
```

Per-cast-type toggles:

| Constructor field | Environment variable | Controls |
| --- | --- | --- |
| `skip_amax_fwd_inp` | `NVTE_NVFP4_SKIP_AMAX_FWD_INP` | Forward input activations |
| `skip_amax_fwd_weight` | `NVTE_NVFP4_SKIP_AMAX_FWD_WEIGHT` | Forward weights |
| `skip_amax_bwd_grad` | `NVTE_NVFP4_SKIP_AMAX_BWD_GRAD` | Backward gradients |

Resolution priority, from highest to lowest:

1. Constructor argument, for example `NVFP4BlockScaling(skip_amax_bwd_grad=False)`.
2. Per-type environment variable, for example `NVTE_NVFP4_SKIP_AMAX_BWD_GRAD=0`.
3. Global `skip_amax` value, either from `NVFP4BlockScaling(skip_amax=True)` or `NVTE_NVFP4_SKIP_AMAX=1`.

Example: skip amax globally but keep amax computation enabled for backward gradients:

```bash
export NVTE_NVFP4_SKIP_AMAX=1
export NVTE_NVFP4_SKIP_AMAX_BWD_GRAD=0
```

```python
recipe = NVFP4BlockScaling(skip_amax=True, skip_amax_bwd_grad=False)
```

Internally, each resolved `skip_amax` value is propagated into the corresponding NVFP4 `QParams`, then into `NVFP4Quantizer`. In C++, `NVFP4Quantizer::quantize()` passes `!skip_amax` as `compute_amax`, so `skip_amax=True` means the amax kernel is not launched and quantization uses the pre-initialized amax value of `1.0`.

## NGPT+FNGPT toggles

NGPT and FNGPT are controlled with environment variables, not Megatron-LM CLI flags.

Enable the full NGPT+FNGPT path:

```bash
export NGPT=true
export FNGPT=true
```

Optional NGPT/FNGPT-related environment variables:

| Environment variable | Default | Controls |
| --- | --- | --- |
| `NGPT` | `false` | Enables nGPT behavior: layernorm bypass, `sqk`/`suv`/`sz`, SLERP residuals, and nGPT weight normalization. |
| `FNGPT` | `false` | Enables fine-grained activation normalization in attention, MLP, MoE/shared-expert, and Mamba mixer paths. |
| `NGPT_ALPHA_INIT` | `0.05` | Sets the initial residual interpolation alpha for NGPT transformer and Mamba SLERP paths. |
| `NVTE_JUSTNORM_MODE` | `default` | Selects the L2 normalization implementation used by NGPT/FNGPT helpers. Use `triton` for fused Triton kernels; other supported paths include `default`, `compile`, and `fused`. |

Example with fused Triton L2 normalization and a custom residual alpha:

```bash
export NGPT=true
export FNGPT=true
export NVTE_JUSTNORM_MODE=triton
export NGPT_ALPHA_INIT=0.05
```

## Training script examples

### 1.2B dense nGPT example

This example configures the 1.2B dense Transformer run with `NGPT=true` and `FNGPT=true`.

```bash
#!/bin/bash
#SBATCH --job-name=<job-name>
#SBATCH --output=<output-path>
#SBATCH --error=<error-path>
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=<partition>
#SBATCH --time=<time-limit>
#SBATCH --account=<account>

DATA_PATH="<data-path>"
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT="<master-port>"

srun \
--container-image=<container-image> \
--container-mounts=<container-mounts> \
bash -lc "
pip install tiktoken

EXPERIMENT=\"\${SLURM_JOB_NAME}\"
OUTPUT_BASE=\"<output-base>\"
EXPERIMENT_DIR=\"\${OUTPUT_BASE}/\${EXPERIMENT}\"
CHECKPOINT_PATH=\"\${EXPERIMENT_DIR}/checkpoint\"
TENSORBOARD_DIR=\"\${EXPERIMENT_DIR}/tensorboard\"
TOKENIZER_MODEL=\"<tokenizer-model>\"
BLEND_PATH=\"<blend-path>\"
DATA_CACHE_PATH=\"\${OUTPUT_BASE}/data_cache\"

mkdir -p \"\${EXPERIMENT_DIR}\"
mkdir -p \"\${CHECKPOINT_PATH}\"
mkdir -p \"\${TENSORBOARD_DIR}\"
mkdir -p \"\${DATA_CACHE_PATH}\"

DISTRIBUTED_ARGS=(
    --nproc_per_node \${SLURM_GPUS_PER_NODE}
    --nnodes \${SLURM_NNODES}
    --node_rank \${SLURM_NODEID}
    --master_addr ${MASTER_NODE}
    --master_port ${MASTER_PORT}
)

export NGPT=\"true\"
export FNGPT=\"true\"
export NVTE_JUSTNORM_MODE=triton

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 20
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --seq-length 8192
    --max-position-embeddings 8192
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --squared-relu
    --init-method-std 0.0221
    --attention-backend flash
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --normalization RMSNorm
)

TRAINING_ARGS=(
    --micro-batch-size 3
    --global-batch-size 768
    --train-samples 122070313
    --lr-decay-samples 121046313
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-samples 18310547
    --lr-warmup-samples 0
    --lr 0.3e-3
    --min-lr 1.2e-5
    --lr-decay-style WSD
    --clip-grad 1.0
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --manual-gc
)

# Optional NVFP4 path:
# export NVTE_NVFP4_SKIP_AMAX_FWD_INP=1
# export NVTE_NVFP4_SKIP_AMAX_FWD_WEIGHT=1
# export NVTE_NVFP4_SKIP_AMAX_BWD_GRAD=0
# export NVTE_NVFP4_DISABLE_RHT=1
# DTYPE_ARGS=(
#     --fp4-format e2m1
#     --fp4-recipe nvfp4
# )

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)

DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

DATA_ARGS_LIST=(
    --per-split-data-args-path \${BLEND_PATH}
    --tiktoken-pattern v2
    --data-cache-path \${DATA_CACHE_PATH}
    --no-mmap-bin-files
    --num-workers 4
    --no-create-attention-mask-in-dataloader
    --tokenizer-type TikTokenizer
    --tokenizer-model \${TOKENIZER_MODEL}
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-progress
    --eval-iters 14
    --eval-interval 2000
    --save-interval 2000
    --log-throughput
    --ckpt-format torch
    --distributed-timeout-minutes 60
    --save \${CHECKPOINT_PATH}
    --tensorboard-dir \${TENSORBOARD_DIR}
    --timing-log-option minmax
    --log-params-norm
    --log-num-zeros-in-grad
    --log-straggler
    --disable-straggler-on-startup
    --straggler-minmax-count 16
    --check-weight-hash-across-dp-replicas-interval 20000
)

if [ -f \"\${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt\" ]; then
    echo \"Found existing checkpoint, resuming training...\"
    EVAL_AND_LOGGING_ARGS+=(--load \${CHECKPOINT_PATH})
else
    echo \"No existing checkpoint found, starting fresh training...\"
fi

torchrun \${DISTRIBUTED_ARGS[@]} \\
    \"<megatron-root>/pretrain_gpt.py\" \\
    \${MODEL_ARGS[@]} \\
    \${TRAINING_ARGS[@]} \\
    \${DTYPE_ARGS[@]} \\
    \${MODEL_PARALLEL_ARGS[@]} \\
    \${DDP_ARGS[@]} \\
    \${DATA_ARGS_LIST[@]} \\
    \${EVAL_AND_LOGGING_ARGS[@]}
"
```

Submit the 1.2B dense template after saving it locally:

```bash
sbatch <path-to-1p2b-template>
```

### 3B/30B hybrid Mamba-Transformer MoE example

This example configures the 3B/30B hybrid Mamba-Transformer MoE run. In the hybrid pattern, `M` is a Mamba block, `E` is an MoE block, and `*` is an attention block.

```bash
#!/bin/bash
#SBATCH --job-name=<job-name>
#SBATCH --output=<output-path>
#SBATCH --error=<error-path>
#SBATCH --nodes=48
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=<partition>
#SBATCH --time=<time-limit>
#SBATCH --account=<account>

export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_WORKER_START=fork
export NCCL_IB_TIMEOUT=23
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="<data-path>"
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT="<master-port>"

srun \
--container-image=<container-image> \
--container-mounts=<container-mounts> \
bash -lc "
pip install tiktoken
pip install --no-cache-dir causal-conv1d --no-build-isolation --no-deps
pip install --no-cache-dir mamba-ssm --no-build-isolation --no-deps
pip install transformers

export TRITON_CACHE_DIR=/tmp/.triton_cache

EXPERIMENT=\"\${SLURM_JOB_NAME}\"
OUTPUT_BASE=\"<output-base>\"
EXPERIMENT_DIR=\"\${OUTPUT_BASE}/\${EXPERIMENT}\"
CHECKPOINT_PATH=\"\${EXPERIMENT_DIR}/checkpoint\"
TENSORBOARD_DIR=\"\${EXPERIMENT_DIR}/tensorboard\"
WANDB_DIR=\"\${EXPERIMENT_DIR}/wandb\"
TOKENIZER_MODEL=\"<tokenizer-model>\"
BLEND_PATH=\"<blend-path>\"
DATA_CACHE_PATH=\"\${OUTPUT_BASE}/data_cache\"

mkdir -p \"\${EXPERIMENT_DIR}\"
mkdir -p \"\${CHECKPOINT_PATH}\"
mkdir -p \"\${TENSORBOARD_DIR}\"
mkdir -p \"\${DATA_CACHE_PATH}\"

DISTRIBUTED_ARGS=(
    --nproc_per_node \${SLURM_GPUS_PER_NODE}
    --nnodes \${SLURM_NNODES}
    --node_rank \${SLURM_NODEID}
    --master_addr ${MASTER_NODE}
    --master_port ${MASTER_PORT}
)

export NGPT=\"true\"
export FNGPT=\"true\"
export NVTE_JUSTNORM_MODE=triton

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 52
    --hidden-size 2688
    --ffn-hidden-size 1856
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 2
    --kv-channels 128
    --seq-length 8192
    --max-position-embeddings 8192
    --position-embedding-type none
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --squared-relu
    --init-method-std 0.0192
    --attention-backend flash
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --normalization RMSNorm
    --is-hybrid-model
    --mamba-num-heads 64
    --mamba-head-dim 64
    --hybrid-override-pattern \"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME\"
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
)

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 6
    --moe-router-score-function sigmoid
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-4
    --moe-router-topk-scaling-factor 2.5
    --moe-router-enable-expert-bias
    --moe-router-dtype fp32
    --moe-router-load-balancing-type seq_aux_loss
    --moe-shared-expert-intermediate-size 3712
    --moe-shared-expert-overlap
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 3072
    --train-samples 3051757813
    --lr-wsd-decay-style minus_sqrt
    --lr-warmup-samples 0
    --lr-decay-samples 121046312
    --lr-wsd-decay-samples 24209262
    --lr 0.5e-3
    --min-lr 0.5e-5
    --lr-decay-style WSD
    --clip-grad 1.0
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --bf16
    --manual-gc
    --exit-interval 40000
    --override-opt-param-scheduler
)

# Optional NVFP4 path:
# export NVTE_NVFP4_SKIP_AMAX_FWD_INP=1
# export NVTE_NVFP4_SKIP_AMAX_FWD_WEIGHT=1
# export NVTE_NVFP4_SKIP_AMAX_BWD_GRAD=0
# export NVTE_NVFP4_DISABLE_RHT=1
# DTYPE_ARGS=(
#     --fp4-format e2m1
#     --fp4-recipe nvfp4
# )

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --sequence-parallel
)

DDP_ARGS=(
    --use-distributed-optimizer
    --ddp-num-buckets 8
    --ddp-pad-buckets-for-high-nccl-busbw
    --overlap-grad-reduce
    --overlap-param-gather
)

EXPERIMENTAL_ARGS=(
    --enable-experimental
    --use-fused-weighted-squared-relu
)

DATA_ARGS_LIST=(
    --per-split-data-args-path \${BLEND_PATH}
    --tiktoken-pattern v2
    --data-cache-path \${DATA_CACHE_PATH}
    --no-mmap-bin-files
    --num-workers 8
    --no-create-attention-mask-in-dataloader
    --tokenizer-type TikTokenizer
    --tokenizer-model \${TOKENIZER_MODEL}
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-progress
    --eval-iters 14
    --eval-interval 2000
    --save-interval 250
    --save-retain-interval 2000
    --log-throughput
    --log-timers-to-tensorboard
    --log-energy
    --ckpt-format torch
    --distributed-timeout-minutes 180
    --save \${CHECKPOINT_PATH}
    --tensorboard-dir \${TENSORBOARD_DIR}
    --timing-log-option minmax
    --log-params-norm
    --log-num-zeros-in-grad
    --log-straggler
    --disable-straggler-on-startup
    --straggler-minmax-count 16
    --check-weight-hash-across-dp-replicas-interval 20000
)

if [ -f \"\${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt\" ]; then
    echo \"Found existing checkpoint, resuming training...\"
    EVAL_AND_LOGGING_ARGS+=(--load \${CHECKPOINT_PATH})
else
    echo \"No existing checkpoint found, starting fresh training...\"
fi

torchrun \${DISTRIBUTED_ARGS[@]} \\
    \"<megatron-root>/pretrain_mamba.py\" \\
    \${MODEL_ARGS[@]} \\
    \${MOE_ARGS[@]} \\
    \${TRAINING_ARGS[@]} \\
    \${DTYPE_ARGS[@]} \\
    \${MODEL_PARALLEL_ARGS[@]} \\
    \${DDP_ARGS[@]} \\
    \${EXPERIMENTAL_ARGS[@]} \\
    \${DATA_ARGS_LIST[@]} \\
    \${EVAL_AND_LOGGING_ARGS[@]}
"
```

Submit the 3B/30B hybrid MoE template after saving it locally:

```bash
sbatch <path-to-3b-30b-template>
```
