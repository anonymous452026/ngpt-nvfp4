# ngpt-nvfp4

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
