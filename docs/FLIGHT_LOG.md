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
