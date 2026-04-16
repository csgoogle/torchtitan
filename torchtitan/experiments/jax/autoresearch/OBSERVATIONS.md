# TorchTitan JAX Llama3 8B — performance-autoresearch observations

Branch: `perfautoresearch/<tag>`
Session start: `<YYYY-MM-DD>`
Hardware: TPU v6e-4

---

## reference: MaxText llama3.1-8b

*Filled in once at session start from the MaxText profile. This is the "optimized ceiling" target.*

- **Command used**: (paste full command here)
- **Profile path**: `<dump_folder>/<run_name>/`

**Metrics**:
- TPS: _
- MFU: _
- Step time (median steps 11–13): _
- Peak HBM: _
- Compile time: _

**Profile signals**:
- Bottleneck category (from `get_overview`): _
- Top 10 ops by self-time (from `get_top_hlo_ops`): _
- Attention kernel: _ (e.g. flash / splash / pallas kernel name in HLO)
- Collective placement (all-gather / reduce-scatter / all-reduce): _
- Remat fingerprint: _
- Notable HLO patterns: _

**XLA flags in use** (from MaxText config + env): _

**Key takeaways for TorchTitan optimization**: _

---

## baseline: TorchTitan JAX Llama3 8B

*Filled in after the first run on the fresh branch.*

- **Commit**: _
- **Command used**: (paste full command here)
- **Profile path**: `<dump_folder>/<run_name>/`

**Metrics**:
- TPS: _
- MFU: _
- Step time (median steps 6–15): _
- Peak HBM: _
- Compile time: _

**Gap vs MaxText reference** (TPS ratio, MFU delta, what's slower): _

**Profile signals**:
- Bottleneck category: _
- Top 10 ops: _
- Attention kernel: _
- Collective placement: _
- Remat fingerprint: _
- Notable HLO patterns: _

**Initial hypotheses** (ordered by expected TPS impact): _

---

## experiments

<!-- Append one detailed block per experiment, using the template in program.md. -->

---

## approach evolution

<!-- Record program.md updates here, one block per approach change. -->
