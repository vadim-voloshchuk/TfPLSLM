# TfPLSLM

Research POC: a Russian autoregressive language model with Mamba-2-style
selective state-space blocks and a fixed 16×256 latent memory. No self-attention
or token-history KV cache appears in the experimental model. The separate
Transformer baseline is explicitly labelled.

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

Long-memory tests use disjoint seeds, held-out names and modified templates.
Distances are token gaps from the end of the factual prefix to the question.
Accuracy is exact generated-answer match, with a terminal full stop ignored.
Normal, zero, prematurely reset, rotated-document, and disabled-explicit-memory
conditions share the same continuation. Wilson intervals accompany small samples.
The memory-scaling test measures persistent state bytes separately from allocator
and peak VRAM; it does not claim constant total training memory.

## Attribution and scope

SSD kernels: [state-spaces/mamba](https://github.com/state-spaces/mamba), commit
`95d8aba8a8c75aedcaa6143713b11e745e7cd0d9` (v2.2.4, Apache-2.0).
Fetched into ignored `vendor/mamba`; no upstream kernel modifications.
The namespace loader avoids importing the unused Mamba-1 CUDA extension.
PyTorch depthwise convolution carries the convolution state explicitly.

No trained weights or positive scientific result are implied by the code.
Actual outcomes, deviations and costs belong in `reports/FINAL_REPORT.md`.
