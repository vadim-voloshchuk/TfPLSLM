# TfPLSLM

Research POC: a Russian autoregressive language model with Mamba-2-style
selective state-space blocks and a fixed 16×256 latent memory. No self-attention
or token-history KV cache appears in the experimental model. The separate
Transformer baseline is explicitly labelled.

## First completed experiment — 2026-10-07

**Partially successful.** All three models trained from scratch on the same
350,025,631 supervised tokens (300,145,627 natural / 49,880,004 synthetic),
with one seed per model. The sampler's final RNG and document cursors match.

| Model | Parameters | Natural validation PPL | Median useful tokens/s |
|---|---:|---:|---:|
| SSM + explicit memory | 77,087,008 | 26.36 | 76,322 |
| SSM-only | 72,609,024 | 26.25 | 73,946 |
| Transformer, 512-token window | 75,883,520 | 28.99 | 154,603 |

Validation uses the same 399,150 target tokens. Context availability differs;
these single runs do not establish general architectural superiority.

The experimental model's mean whole-answer accuracy across seven synthetic task
families is 77.23% at 512 tokens and 55.80% at 2k, but 0% at both 8k and 32k.
Disabling its explicit memory lowers accuracy at 512 but not at 2k. A separate,
post-hoc component-swap diagnostic finds that answers mainly follow SSD/conv
state, not the explicit slots, at the final chunk boundary. Earlier effects of
explicit memory on processing and learning remain possible.

Persistent state remains 4,323,328 bytes per example through 128k processed tokens.
Generation improves over random initialization but remains repetitive and
unreliable. See the [full report](reports/FINAL_REPORT.md), including negative
results, costs, generation samples, and the next controlled experiment.

## Architecture

The default model has 77,087,008 parameters: 16 residual blocks, width 512,
SSM expansion 2, state dimension 64, head dimension 64, SwiGLU width 1536,
and tied 16,384-token embeddings. Each block carries its SSD state and the
last three convolution inputs between chunks. Upstream Mamba's unmodified
Triton SSD kernels implement the training scan; the Python token loop is a
test reference only.

Explicit memory is read through a learned projection and an elementwise gate
at the input of each chunk. The mean of final hidden states updates it after
each 512-token chunk via learned retain/write gates and bounded values.
The update is causal: it cannot influence logits earlier in the same chunk.
Streaming generation accumulates the same summary and writes only at the
same fixed boundaries. This is an intentionally simple, order-sensitive POC;
chunk pooling may lose exact identifiers and independent facts.

Training unrolls two adjacent chunks with gradients, then detaches recurrent
state. Each batch row stays on one document until its end; padding losses are
masked and every new document resets that row's complete state. Thus the
explicit writer receives learning signal from the following chunk, while the
gradient horizon remains 1024 tokens. State can persist beyond this horizon.

## Reproduce

Use Linux, a CUDA GPU, Python 3.12 and the CUDA 12.4 PyTorch 2.6 wheel. The
validated hardware is H100 PCIe 80 GB. Do not change the host driver.

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
bash scripts/setup_server.sh
python -m pytest -q
python -m tfplslm.train --mode smoke --config configs/smoke.json --out artifacts/smoke
python -m tfplslm.train --mode benchmark --config configs/poc_70m.json --out artifacts/benchmark
python -m tfplslm.data --out data
# Only after reviewing smoke.json, benchmark.json and the remaining budget:
python -m tfplslm.train --mode train --config configs/poc_70m.json --out artifacts/poc
python -m tfplslm.evaluate --checkpoint artifacts/poc/last.pt --out artifacts/poc/eval
```

The independently trained recurrent control and optional Transformer control use
the same tokenizer, document sampler seed and 350M-token budget:

```bash
python -m tfplslm.train --mode train --config configs/baseline_ssm.json --out artifacts/ssm
python -m tfplslm.evaluate --checkpoint artifacts/ssm/last.pt \
  --config configs/eval_ssm.json --out artifacts/ssm/eval
python -m tfplslm.train --mode train --config configs/baseline_transformer.json --out artifacts/transformer
python -m tfplslm.evaluate --checkpoint artifacts/transformer/last.pt --out artifacts/transformer/eval
```

Run one GPU job at a time. Start each new experiment in a new output directory;
use `--resume` when continuing an existing run. The experiment controller scripts
add rental-specific gates and expect the saved smoke/pilot/budget artifacts;
the individual commands above do not require Vast.ai.

Resume includes weights, optimizer, RNGs, sampler cursors, recurrent state,
LR schedule position, token counts and config:

```bash
python -m tfplslm.train --mode train --config configs/poc_70m.json \
  --out artifacts/poc --resume artifacts/poc/last.pt
```

Data uses a pinned streaming subset of
[Cultura-Ru-Edu](https://huggingface.co/datasets/deepvk/cultura_ru_edu), a custom
SentencePiece BPE tokenizer, and independently generated Russian memory tasks.
Corpus revision, document hashes, binary/index checksums and tokenizer sample
manifest are written to `data/manifest.json`. Cyrillic and `ё` are preserved;
byte fallback handles unseen characters. Validation is the dataset's separate
split with exact hash overlap removed. No claim of near-duplicate removal is made.

Long-memory tests use disjoint seeds and held-out names. Archive-related framing
is added around the same underlying task templates; this is not a separate
held-out template family.
Distances are token gaps from the end of the factual prefix to the question.
Accuracy is exact generated-answer match, with a terminal full stop ignored.
Normal, zero, prematurely reset, rotated-document, and disabled-explicit-memory
conditions share the same continuation. Wilson intervals accompany small samples.
Zero and shuffled states are intervened on before the final chunk; reset retains
only the last complete history chunk. In contrast, `no_memory` disables explicit
memory reads and writes throughout the prefix and answer. The additional
`eval_state_components.json` diagnostic swaps only the slots or only SSD/conv
state at the final boundary, on a fresh seed, after the primary results.
The memory-scaling test measures persistent state bytes separately from allocator
and peak VRAM; it does not claim constant total training memory.

## Configuration and artifacts

`configs/smoke.json` is the tiny overfit/resume test. `poc_70m.json` defines the
77.1M-parameter experimental model; `baseline_ssm.json` removes explicit memory
(72.6M parameters), and `baseline_transformer.json` defines a 75.9M-parameter
512-token context model. `eval_memory.json` specifies all five paired state
conditions; `eval_ssm.json` specifies normal and zero-state controls.

Each run saves `config.json`, `metrics.jsonl`, `last.pt`, validation results,
and generation samples. Recurrent models additionally save memory accuracy and
individual answers, causal swaps, language-model state ablations, state-size
scaling, and slot diagnostics. Large weights and prepared datasets are excluded
from Git. Small result summaries and the reviewed report belong in `reports/`.

The current memory writer mean-pools a chunk and receives gradients through at
most two chunks. Long-history tests exceed this gradient horizon and, at 8k/32k,
the synthetic training distances. Template-based tasks and one seed per model
support a first diagnostic result, not a general claim about language ability or
architectural superiority.

## Attribution and scope

SSD kernels: [state-spaces/mamba](https://github.com/state-spaces/mamba), commit
`95d8aba8a8c75aedcaa6143713b11e745e7cd0d9` (v2.2.4, Apache-2.0).
Fetched into ignored `vendor/mamba`; no upstream kernel modifications.
The namespace loader avoids importing the unused Mamba-1 CUDA extension.
PyTorch depthwise convolution carries the convolution state explicitly.

Actual outcomes, negative results and costs are documented in
[`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md). Rebuild its tables and plots
with `python scripts/build_report.py` after restoring the local artifacts;
`reports/interpretation.json` contains the reviewed qualitative conclusions.
