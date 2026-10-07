# pretrain/: Artemis I

A small GPT, built up from an empty file in very small steps. Each step is explained,
reviewed, and committed with a Flight Log entry before the next one starts.

## Hardware notes (local GTX 1080, 8 GB)

- Pascal, compute capability 6.1: train in **fp32**. No bf16. Measured matmul throughput
  (`check_env.py`): fp32 8.1 TFLOPS, fp16 8.5 TFLOPS, so fp16 saves memory but not time.
- `torch.compile` needs Triton on compute 7.0+, so skip it locally.
- Fits comfortably: models up to ~10 to 30M parameters with context 256 to 512. That is
  plenty for debugging, overfitting tests, and loss-curve sanity checks.
- The final Artemis I run happens on rented GPUs, using the same code.

## Countdown: Phase 1 steps

**T-minus: environment**
- [x] Install PyTorch with CUDA into `.venv` and confirm `torch.cuda.is_available()`.
- [x] Note the measured matmul throughput of the 1080 in the Flight Log (you will use it to
      estimate run times).

**Stage 1: from the lecture (Zero to Hero)**
- [x] `bigram.py`: character-level bigram model on a small text (a NASA transcript works
      well instead of Shakespeare). Understand logits, cross-entropy, and why initial loss
      ≈ ln(vocab_size).
- [x] Single-head self-attention, then multi-head, then a full block (attention + MLP +
      residuals + LayerNorm).
- [x] `gpt.py`: a GPT-2 style model, scaled up, with dropout, checkpoints, and sampling.
- [x] Check that it can **overfit one batch** to near-zero loss.

**Stage 2: modern deltas**
- [x] Swap LayerNorm → RMSNorm.
- [x] Remove biases.
- [x] Swap learned positional embeddings → RoPE.
- [x] Compute all attention heads in one batch instead of a loop.
- [x] Swap the MLP → SwiGLU (ours was ReLU, not GELU).
- [x] Each swap: one commit, before/after numbers in the Flight Log. Loss curves:
      `pretrain/plot_losses.py`, charts in `docs/curves/`.

**Stage 3: tokenizer**
- [x] `bpe.py`: byte-level BPE, trained on NASA text. (Adding FineWeb-Edu general text
      comes with the Stage 4 data.)
- [x] Compare token counts on aerospace terms (e.g. "LOX/LH2", "ΔV", "Isp", "TLI",
      "N₂O₄") against the GPT-2 tokenizer: `compare_tokenizers.py`, results in the Flight Log.
- [ ] Train the GPT on BPE tokens instead of characters.

**Stage 4: real training**
- [ ] Warmup + cosine or WSD schedule, gradient clipping, checkpoints, eval loss.
- [ ] Reproduce a small nanochat run, swap in your model, match its loss curve.
- [ ] Final mixed general + NASA run in the cloud. Publish curves. Tag `artemis-i`.
