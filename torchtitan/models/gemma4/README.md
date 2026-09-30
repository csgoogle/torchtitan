# Gemma 4 in TorchTitan

This directory contains the TorchTitan implementation of the **Gemma 4** text model family (`31B` Dense and `26B-A4B` Mixture-of-Experts).

## Architecture Overview

- **Hybrid Sliding / Global Attention**:
  - 5-to-1 ratio of local sliding-window attention (`sliding_window=1024`, `head_dim=256`, RoPE $\theta = 10{,}000$) to global full causal attention (`head_dim=512`, proportional RoPE with `partial_rotary_factor=0.25` and $\theta = 1{,}000{,}000$, and $K = V$ projection sharing via `use_k_eq_v=True`).
  - Q/K RMSNorm (with learned scale) and V RMSNorm (scale-free, `elementwise_affine=False`), with attention softmax scale `1.0`.
- **Pre/Post LayerNorms & Layer Scalar**:
  - 4 RMSNorms per dense block (`attention_norm`, `post_attention_layernorm`, `ffn_norm`, `post_feedforward_layernorm`) plus `layer_scalar` buffer, and 3 additional RMSNorms on MoE blocks (`pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`, `post_feedforward_layernorm_2`).
- **Parallel Dense + Sparse MoE (`26B-A4B`)**:
  - Every layer runs a GeGLU dense MLP branch in parallel with a GeGLU routed-experts MoE branch (`128` experts, `top_k=8`, `Gemma4Router` with internal RMSNorm, learned input `scale`, and `per_expert_scale`).
- **Embeddings & Output Softcapping**:
  - Tied input/output embeddings (`vocab_size=262144`) scaled by $\sqrt{\text{dim}}$ on input and `final_logit_softcapping=30.0` on output logits.

## Supported Model Configs

| Config Function | Model Flavor | Type | Layers | Dim | Heads (Q / Local KV / Global KV) | Experts (Total / Top-K) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma4_debugmodel` | `debugmodel` | Dense | 6 | 256 | 4 / 4 / 2 | - |
| `gemma4_moe_debug` | `debugmodel_moe` | MoE | 6 | 256 | 4 / 4 / 2 | 8 / 2 |
| `gemma4_31b` | `31B` | Dense | 60 | 5376 | 32 / 16 / 4 | - |
| `gemma4_26b_a4b` | `26B-A4B` | MoE | 30 | 2816 | 16 / 8 / 2 | 128 / 8 |

## Usage

```bash
NGPU=8 ./run_train.sh --module gemma4 --config gemma4_31b
NGPU=8 ./run_train.sh --module gemma4 --config gemma4_26b_a4b
```
