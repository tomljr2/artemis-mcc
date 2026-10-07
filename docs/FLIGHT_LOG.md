# Flight Log

A running learning journal. One entry per session: what you tried, what you learned, what
broke. These entries become the raw material for the phase writeups.

Entry template:

```
## YYYY-MM-DD · Phase N · <short title>
**Objective:**
**What I did:**
**What I learned:** (explain it as if to someone else)
**Anomalies:** (bugs, surprises, things that didn't match expectations)
**Next:**
```

---

## 2026-10-06 · Phase 0 · Repository stood up
**Objective:** Set up the repo, plan, and structure.
**What I did:** Created the project scaffold, the plan (docs/PLAN.md), and this log.
**Next:** Phase 1, step 1: environment check, then the bigram model.

## 2026-10-06 · Phase 0 · Pre-flight review
**Objective:** Review the scaffold before writing any code.
**What I did:** Fixed `pyproject.toml` so the project installs, stopped tracking IDE files,
normalized line endings, and added setup instructions to the README.
**What I learned:** setuptools auto-discovers packages from top-level folders and refuses
to guess when there are several. Pip on Windows installs CPU-only PyTorch unless you point
it at a CUDA wheel index, and the GTX 1080 needs the `cu126` builds.
**Next:** Phase 1, step 1: install PyTorch.

## 2026-10-06 · Phase 1 · Steps 1–2: PyTorch and GPU throughput
**Objective:** Get PyTorch running on the GTX 1080 and measure its speed.
**What I did:** Installed torch 2.14.1+cu126 and wrote `pretrain/check_env.py`, which times
4096×4096 matmuls.
**What I learned:** LLMs are mostly matrix multiplications, so matmul throughput sets
training speed. GPU calls are asynchronous; you must `synchronize()` before timing. One
n×n matmul is 2n³ operations. Training costs about 6 × params × tokens operations.
**Measurements:** fp32 8.07 TFLOPS, fp16 8.46 TFLOPS (no tensor cores on Pascal, so fp16
only saves memory). A 10M-param model on 100M tokens ≈ 30 min locally at ~40% utilization.
**Next:** Step 3: get a small text dataset and look at its characters.

## 2026-10-06 · Phase 1 · Step 3: Apollo 11 transcript
**Objective:** Get a first training text and inspect its characters.
**What I did:** `pretrain/prepare_apollo11.py` downloads NASA's Apollo 11 technical
air-to-ground transcript, strips the HTML and page headers, and saves plain text.
**What I learned:** For a character-level model, the set of unique characters *is* the
vocabulary. Rare characters (`é` ×2, `%` ×1) get almost no training signal: frequency in
the data decides what a small model can learn.
**Measurements:** 826,031 characters, 81 unique (Karpathy's Shakespeare: ~1.1M, 65).
**Anomalies:** (1) Python's TLS failed with "certificate has expired": the Windows store
holds an expired cross-signed ISRG Root X2 (Sep 2025), which OpenSSL chose over the valid
path. Fixed by verifying against `certifi`, not by disabling verification. (2) One OCR byte
(`0xA2` for "o") was not valid UTF-8; fixed explicitly and decoding made strict.
**Next:** Step 4: encode text to integers and split train/validation.

## 2026-10-06 · Phase 1 · Step 4: Character tokenizer
**Objective:** Turn text into integers and back.
**What I did:** `pretrain/char_tokenizer.py` (encode/decode over the sorted unique
characters), written test-first with `tests/test_char_tokenizer.py`.
**What I learned:** A network is only arithmetic, so text must become numbers. A small
vocabulary keeps the output layer small (one score per token). Token ids are *not*
quantities: "R"=45 is not "close" to "S"=46. They are row indices into a learned embedding
table, which is where meaning comes from.
**Measurements:** vocab 81; the full transcript round-trips exactly (826,031 ids).
**Next:** Step 5: train/validation split.

## 2026-10-06 · Phase 1 · Step 5: Train/validation split
**Objective:** Hold out data so we can tell learning from memorizing.
**What I did:** `pretrain/dataset.py` `train_val_split`: first 90% train, last 10% val.
**What I learned:** Validation loss is the only honest signal of generalization; training
loss falling while validation loss rises means overfitting. Split by position, not at
random, since a model predicting from context would otherwise be graded on text whose
neighbours it already trained on (the same idea as holding out whole documents later).
**Measurements:** train 743,427 ids (launch → day 6), val 82,604 ids (day 6 → splashdown).
Expect val loss slightly above train loss: different mission phase, different vocabulary.
**Next:** Step 6: batches of context windows.

## 2026-10-06 · Phase 1 · Step 6: Batches
**Objective:** Turn the token stream into training examples.
**What I did:** `get_batch` in `pretrain/dataset.py` cuts random windows; targets are the
inputs shifted left by one. Also corrected the release mapping to NASA's February 2026
Artemis restructure (III = Earth-orbit docking test, IV = first landing).
**What I learned:** Training is self-supervised: the "label" for each position is just the
next character, which the text already contains. One window of 8 holds 8 examples, with
contexts from 1 to 8 characters. Tensors have shapes, `(batch, block)`, and batches move
to the GPU with `.to(device)` while the dataset stays on the CPU.
**Anomalies:** Timestamps are a noticeable share of the text and their digits are nearly
unpredictable, which puts a floor under the achievable loss.
**Next:** Step 7: bigram model and its loss before training.

## 2026-10-06 · Phase 1 · Step 7: Bigram model
**Objective:** Build the first model and measure it before training.
**What I did:** `pretrain/bigram.py`: one 81×81 `nn.Embedding` table; forward returns
logits and cross-entropy loss. Tests check shapes, that only the current token matters,
and that an all-zero table gives exactly ln(81).
**What I learned:** Logits are raw scores; softmax makes them probabilities; cross-entropy
is −ln(probability of the correct token), averaged over B×T examples. ln(vocab) is the
loss of a uniform guess, the sanity check for any untrained model.
**Measurements:** 6,561 parameters. Untrained loss 4.887 vs ln(81) = 4.394: random initial
scores make the model confidently wrong, which costs ~0.5 over a uniform guess.
**Next:** Step 8: training loop.

## 2026-10-06 · Phase 1 · Step 8: Training loop
**Objective:** Make the loss go down.
**What I did:** `pretrain/train.py`: `train_step` (forward, zero_grad, backward, step) and
`estimate_loss` (averaged train/val loss); AdamW, lr 1e-2, 3,000 steps of 32×8.
**What I learned:** Backprop walks the forward computation in reverse (chain rule) and
returns every parameter's gradient in one pass. For softmax + cross-entropy the gradient on
each score is (predicted probability − correct answer). Gradient descent: value −= lr ×
gradient. Batch gradients average competing evidence, so probabilities settle at the true
frequencies.
**Measurements:** loss 4.886 → 2.398 train / 2.464 val, flat after ~1,000 steps: the bigram
ceiling (≈9% average probability on the right character). Learned R→o 39%, L→M 54% (LMP).
**Next:** Step 9: generate text from the bigram model.

## 2026-10-06 · Phase 1 · Step 9: Generation
**Objective:** Make the model write.
**What I did:** `BigramModel.generate`: forward, take the last position's scores, softmax,
sample, append, repeat.
**What I learned:** Generation is the same loop chat models use. Sampling (not argmax)
avoids loops: argmax after "\n" would print newlines forever.
**Anomalies:** Output is locally plausible pairs ("th", "ou") with broken structure:
timestamps like "47016" because counting digits needs more than one character of memory.
This is the motivation for attention.
**Next:** Step 10: averaging the past as a matrix multiply.

## 2026-10-06 · Phase 1 · Step 10: Averaging the past
**Objective:** Let a position use its context, in the simplest way.
**What I did:** `pretrain/attention.py`: `causal_average_weights` (zero scores → mask the
future with −inf → softmax) and `causal_average` (weights @ x).
**What I learned:** Positions carry vectors (channels). A weighted sum over positions is one
matrix multiply. The lower-triangular mask is what keeps the future from leaking. Real
attention only changes the scores: query · key instead of zeros.
**Next:** Step 11: a single self-attention head.

## 2026-10-06 · Phase 1 · Step 11: Self-attention head
**Objective:** Replace equal weights with learned, content-based ones.
**What I did:** `AttentionHead` in `pretrain/attention.py`: query/key/value projections,
scores = q·k / √head_size, mask the future, softmax, weights @ values. Shared
`mask_future` helper.
**What I learned:** Score (t, s) is the dot product of t's query with s's key. With zero
queries the head reduces exactly to the causal average (tested). Scaling matters: at
head_size 512, unscaled softmax put 0.95 on one random position before any training.
**Anomalies:** Attention is order-blind; it needs position embeddings.
**Next:** Step 12: put the head in a model and train it.

## 2026-10-06 · Phase 1 · Step 12: First attention model
**Objective:** Use the head in a model and beat the bigram.
**What I did:** `pretrain/gpt.py`: token + position embeddings → one attention head →
`lm_head`; `generate` crops to block_size. `train.py` takes `--model bigram|gpt`.
**What I learned:** Position embeddings give the order-blind attention a sense of place.
The context window exists because the position table has block_size rows.
**Measurements:** 9,361 params, block 32, n_embd 32, 5,000 steps (33 s): val 2.262 vs
bigram 2.453. Initial loss 4.430 ≈ ln(81). Train/val gap grew to ~0.12.
Generated timestamps now have the right four-groups-of-two-digits shape.
**Next:** Step 13: multi-head attention.

## 2026-10-06 · Phase 1 · Step 13: Multi-head attention
**Objective:** Let several attention patterns run side by side.
**What I did:** `MultiHeadAttention`: n_head heads of n_embd/n_head channels, concatenate,
then a learned output projection. GPT uses 4 heads × 8 channels.
**What I learned:** Heads specialize because each starts from different random weights and
redundancy doesn't lower the loss. Specialization is encouraged, not guaranteed. A larger
context only helps if the model can use it; attention cost grows with the square of T.
**Measurements (single seed, 5,000 steps):** 1 head 2.262 → 4 heads, same 9,361 params,
2.146 → plus projection (10,417 params) 2.079. Context sweep with 4 heads + proj:
block 8 → 2.180, 32 → 2.079, 128 → 2.140 (too small a model to use 128).
**Next:** Step 14: feed-forward (MLP).

## 2026-10-06 · Phase 1 · Step 14: Feed-forward MLP
**Objective:** Let each position process what attention gathered.
**What I did:** `pretrain/mlp.py` `FeedForward`: Linear n→4n, ReLU, Linear 4n→n, applied
per position. GPT now runs attention then MLP.
**What I learned:** Attention communicates across positions; the MLP computes within each
position (tested: other positions can't affect it). ReLU is what stops the two linear
layers from collapsing into one, and it gives the network if-then behavior.
**Measurements:** 18,769 params, val 1.965 (from 2.079). Gain mixes structure and size.
**Next:** Step 15: residual connections.

## 2026-10-06 · Phase 1 · Step 15: Residual connections
**Objective:** Make layers add to x instead of replacing it.
**What I did:** `x = x + attention(x)`, `x = x + mlp(x)` in `pretrain/gpt.py`. Test: layers
that output zero pass the embeddings straight to `lm_head`.
**What I learned:** The residual stream is a shared record each layer reads and adds to.
It preserves information, makes an untrained layer harmless, and gives gradients a direct
path back to early layers (the ResNet idea that makes depth trainable).
**Measurements:** same 18,769 params, val 1.965 → 1.831: purely structural gain.
**Anomalies:** Initial loss rose to 4.785 (> ln 81): large raw embeddings now reach the
output directly. Normalization is next.
**Next:** Step 16: layer normalization.

## 2026-10-06 · Phase 1 · Step 16: Layer normalization
**Objective:** Keep the residual stream at a readable scale.
**What I did:** Hand-written `LayerNorm` in `pretrain/norm.py` (mean, variance, normalize,
learned scale and shift), tested against `nn.LayerNorm`. Pre-norm placement: ln1 before
attention, ln2 before the MLP, ln_f before `lm_head`.
**What I learned:** Pre-norm normalizes what each layer reads but leaves the residual path
untouched, so the gradient express lane survives. GPT-2 moved norms before layers for
stable deep training.
**Measurements:** val 1.831 → 1.803 (small, possibly noise for one layer); initial loss
4.785 → 4.599. The remaining gap to ln(81) is PyTorch's default `lm_head` init.
**Next:** Step 17: transformer blocks, stacked.

## 2026-10-06 · Phase 1 · Step 17: Stacked transformer blocks
**Objective:** Package attention + MLP + norms + residuals as a Block and stack them.
**What I did:** `pretrain/block.py` `Block`; GPT now runs n_layer blocks (4).
**What I learned:** Same structure, separate weights per block. Depth only helps if it is
trainable: residuals and norms are what make it so.
**Measurements:** 56,785 params, val 1.661 (from 1.803). Ablation with residuals and norms
removed: val 3.125, stuck near 3.1 and worse than the bigram.
**Anomalies:** Train/val gap 0.21 and growing: memorization starting.
**Next:** Step 18: dropout.

## 2026-10-06 · Phase 1 · Step 18: Dropout
**Objective:** Fight memorization.
**What I did:** `dropout` option on attention weights, attention output, and MLP output
(GPT-2 placement). `train.py` now calls `model.eval()` before generating.
**What I learned:** Dropout is only active in train mode. The true overfitting signal is
val loss *rising*, not a train/val gap: part of our gap is distribution shift (even the
bigram had 0.06), and val was still falling at 5,000 steps.
**Measurements:** dropout 0 → 1.661, 0.1 → 1.724, 0.2 → 1.780 val. Default set to 0.
**Anomalies:** I misread the step 17 gap as memorization; corrected here.
**Next:** Step 19: scale up the model and train longer.

## 2026-10-06 · Phase 1 · Step 19: Scale up
**Objective:** Bigger model, longer training.
**What I did:** n_embd 128, block 64, batch 64, 10,000 steps (820,817 params, 41M tokens =
55 passes over the training text). Progress lines now show elapsed time.
**What I learned:** Real overfitting: val loss bottoms out and then rises while train loss
keeps falling. Chinchilla's ~20 tokens/param suggests ~16M unique tokens for this size; we
have 0.74M. The model has outgrown the data, which is why Phase 2's corpus matters.
**Measurements:** best val 1.317 at step 4,000 → 1.424 at 10,000 (train 0.793). 468 s,
~5% of measured matmul throughput (small matrices, per-head Python loop, eval overhead).
Generated sentences are new combinations, not copies (checked against the transcript).
**Next:** Step 20: re-test dropout at this size.

## 2026-10-06 · Phase 1 · Step 20: Dropout, re-tested at scale
**Objective:** Give dropout a fair test now that the model overfits.
**What I did:** Same 821k-param setup and seed with dropout 0.1 and 0.2; default set to 0.1.
**What I learned:** The right regularization depends on whether the model is memorizing.
Dropout hurt the 57k model (not memorizing) and fixed the 821k model (memorizing).
**Measurements (best / final val):** 0.0 → 1.317 / 1.424, 0.1 → 1.267 / 1.267,
0.2 → 1.277 / 1.277 (still falling). ~8 min per run.
**Next:** Step 21: save the best model and sample from it.

## 2026-10-06 · Phase 1 · Step 21: Checkpointing
**Objective:** Keep the trained model so we never retrain just to use it.
**What I did:** `pretrain/checkpoint.py` with `save_checkpoint` / `load_checkpoint` (weights,
model settings, vocab, step, val loss). `train.py` saves to `checkpoints/gpt_apollo11.pt`
(git-ignored) whenever val loss reaches a new best.
**What I learned:** A model is just named weight tensors plus the settings needed to rebuild
an empty one of the same shape. Save on best val, not last step: it guards against
overfitting late in the run. `weights_only=True` makes loading safe.
**Measurements:** Full run saved step 10,000, val 1.267 (3.3 MB). Reloaded model matches the
saved one exactly in tests.
**Next:** Step 22: sample from the checkpoint with a custom prompt.

## 2026-10-06 · Phase 1 · Step 22: Sampling from the checkpoint
**Objective:** Use the trained model without retraining it.
**What I did:** `pretrain/sample.py`: `continue_text` plus a CLI with `--prompt`, `--chars`,
`--seed`, `--checkpoint`. The tokenizer is rebuilt from the saved vocab.
**What I learned:** Inference is just the forward pass in a loop: ~7 s versus 8 min to
train. The prompt sets the scene ("Tranquility Base" → "Neil, this is Houston"), but it falls
out of view after 64 characters (block_size). Same seed means the same dice rolls, so two
prompts ending in "." produced nearly identical continuations.
**Measurements:** Checkpoint step 10,000, val 1.267. 44 tests pass.
**Next:** Step 23: temperature.

## 2026-10-06 · Phase 1 · Step 23: Temperature
**Objective:** Control how boldly the model samples.
**What I did:** `GPT.generate(..., temperature=1.0)` divides the scores before softmax;
`sample.py --temperature`.
**What I learned:** Temperature changes the odds, not the knowledge. Low → greedy, clean but
bland and prone to repeating ("to to"); high → long shots win, and errors compound until the
format collapses.
**Measurements:** Same seed and prompt at 0.3 / 1.0 / 1.8: readable / as before / gibberish
numbers and invented words. 46 tests pass.
**Next:** Step 24: RMSNorm (Stage 2, modern deltas).

## 2026-10-06 · Phase 1 · Step 24: RMSNorm (Stage 2 begins)
**Objective:** First modern delta: replace LayerNorm with RMSNorm.
**What I did:** `RMSNorm` in `pretrain/norm.py` (divide by root mean square, learned scale,
no shift); all 9 norms in the model switched.
**What I learned:** The useful part of normalization is keeping numbers a steady size;
re-centering and the shift can go. Simpler, fewer parameters, same quality.
**Measurements:** 819,665 params (-1,152). Best val 1.260 at step 10,000 vs 1.267 (one seed,
within likely noise), 456 s vs 468 s. Val still falling at the end. 51 tests pass.
**Anomalies:** Old LayerNorm checkpoints no longer load (they carry shift weights).
**Next:** Step 25: RoPE.

## 2026-10-07 · Phase 1 · Step 25: RoPE
**Objective:** Second modern delta: rotary position embeddings instead of a position table.
**What I did:** `pretrain/rope.py` (`rope_angles`, `apply_rope`); each head rotates its
queries and keys by position before scoring. Learned position table removed.
**What I learned:** Rotation makes a query·key score depend only on the distance between
positions. Position decides where to look, not what gets passed on, so values stay
unrotated. A run of identical tokens ([5, 5, 5]) can't be told apart any more; that old test
was replaced by "word order changes the prediction".
**Measurements:** 811,473 params (-8,192). Faster early (val 1.517 vs 1.692 at step 500) but
best 1.272 vs 1.260. 768 s vs 456 s, likely from recomputing angles in all 16 heads. 57 tests.
**Anomalies:** Val loss now wobbles ±0.01-0.02 between evals, so step 24's 0.007 "gain" and
this 0.012 "loss" are both noise. Single-seed comparisons this small can't be called.
**Next:** Step 26: batched heads (and compute the RoPE angles once).
