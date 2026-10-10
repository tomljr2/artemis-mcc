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

## 2026-10-07 · Phase 1 · Step 26: Batched attention heads
**Objective:** Third modern delta: compute all heads at once instead of a loop.
**What I did:** `CausalSelfAttention`: one query/key/value layer for all heads, heads as an
extra tensor dimension, RoPE angles computed once per block. The loop version stays as the
readable reference.
**What I learned:** GPUs are fast at a few big jobs and slow at many small ones. A test that
copies the loop's weights into the batched version proves it is the same maths.
**Measurements:** 70.5 → 27.4 ms per training step; full run 768 → 343 s. Best val 1.270
(was 1.272, same within noise). 60 tests pass.
**Next:** Step 27: SwiGLU.

## 2026-10-07 · Phase 1 · Step 27: SwiGLU
**Objective:** Fourth modern delta: a gated MLP.
**What I did:** `FeedForward` is now SwiGLU: `down(silu(gate(x)) * up(x))`, hidden size
2/3 of 4 × n_embd to keep the size the same, no biases.
**What I learned:** A learned gate (a volume knob per number) replaces ReLU's fixed rule.
It learns faster, but four upgrades in a row have all landed at ~1.27: the bottleneck is
now the data (one transcript), not the model.
**Measurements:** 808,401 params. Val at step 500: 1.452 vs 1.513. Best val 1.268 vs 1.270.
Final train loss 0.851 vs 0.905. 332 s. 62 tests pass.
**Anomalies:** I flagged the data limit in step 19 but did not raise it until asked.
Checklists in PLAN.md and pretrain/README.md were never ticked.
**Next:** Update README status and checklists, then add more Apollo transcripts.

## 2026-10-07 · Phase 1 · Housekeeping: status and checklists
**Objective:** Make the README and checklists match reality.
**What I did:** README "Mission status" table; ticked done items in `pretrain/README.md`;
open items listed honestly (overfit one batch, remove biases, loss curves).
**What I learned:** Releases (Artemis I, II…), phases (1–6) and stages (1–4, inside
Phase 1) are three different numbering schemes. Always say "Phase 1, Stage 3", never
"Stage 3" alone. No milestone tags: tags are only for real releases.
**Next:** Add more Apollo transcripts to the training data.

## 2026-10-07 · Phase 1 · Step: License log
**Objective:** Find more training text, and record a license decision for every source.
**What I did:** `data/license_log.csv` with three rows: Apollo 11 transcript (include),
Apollo Journals site (exclude, all rights reserved), NASA's scanned Apollo 7-17 PDFs
(pending, garbled text). `tests/test_license_log.py` checks every row and that the source
we train on is marked include.
**What I learned:** NASA hosts the other Apollo transcripts only as 1960s scans with poor
text ("Houston" -> "Baueton", columns split apart). The clean typed versions belong to an
independent, copyrighted site. Our Apollo 11 file came from a nasa.gov copy of that site,
but it is NASA's document word for word.
**Measurements:** 65 tests pass.
**Next:** Choose NASA history books as additional training text.

## 2026-10-07 · Phase 1 · Step: NASA history books
**Objective:** More training text with a clear license.
**What I did:** `data/prepare_nasa_books.py` downloads four NASA History Series books from
NTRS, keeps pages that read as prose, and cleans PDF text (split words, ligatures, curly
quotes, non-ASCII). License log: SP-350, SP-4204, SP-4205, SP-4214 include ("Public Use
Permitted"); SP-4206 *Stages to Saturn* exclude ("Use by or on behalf of the US Gov.
Permitted", university author). The script refuses any book without an include row.
**What I learned:** NTRS states each document's usage terms, which makes license decisions
much clearer than for mirror sites. Parameters (learned numbers) and training data
(tokens read) are separate quantities; Chinchilla suggests ~20 tokens per parameter.
**Measurements:** 1,657 of 1,964 pages kept; 4.6M characters (5.6x the Apollo 11
transcript). 71 tests pass.
**Anomalies:** Remaining noise: page headers ("160 APOLLO"), junk from partly-photo pages,
footnote numbers ("director.65"), occasional missing spaces. Not yet measured.
**Next:** Train on transcript + books, with validation split by document.

## 2026-10-07 · Phase 1 · Step: Train on transcript + books
**Objective:** See what 6x more text does.
**What I did:** `make_splits` holds out one whole book (SP-350) plus the last 10% of the
transcript (the old validation text). `estimate_loss` reports named splits. Checkpoint
renamed `checkpoints/gpt_nasa.pt`.
**What I learned:** In the same 10,000 steps, the model sees each character ~8.5 times
instead of ~55, and mostly book prose. It stopped memorizing (train and val close
together, all falling), so the limit moved from data to training time.
**Measurements:** train 4.79M chars; val transcript 1.333 (was 1.268), val book 1.413,
train 1.202. 374 s. 73 tests pass.
**Anomalies:** I predicted the transcript score would improve; it got worse in this run.
**Next:** Train longer (40,000 steps).

## 2026-10-07 · Phase 1 · Step: Train 4x longer
**Objective:** Give the model time to learn from the larger text.
**What I did:** `max_steps` 10,000 -> 40,000.
**What I learned:** More practice helps, with diminishing returns: the first 10,000 steps
took book val from 4.71 to 1.42, the next 30,000 only to 1.35. The train/val gap is
starting to widen, so the 0.8M-param model is nearing its limit. The model now defaults
to book style, since ~85% of its training text is books.
**Measurements:** Best at step 36,000: val transcript 1.273, val book 1.350 (avg 1.311),
train 1.119. 25 min.
**Next:** Remove the remaining biases (finishes Phase 1, Stage 2).

## 2026-10-07 · Phase 1 · Step: Remove the remaining biases (Stage 2 done)
**Objective:** Last modern delta: no "+ b" offsets, as in Llama.
**What I did:** `bias=False` on the attention output layer (batched and loop versions)
and on `lm_head`. Test: the model has no bias parameters.
**What I learned:** With norms re-centering the numbers, biases add little. Removing them
is about simplicity, not quality. Phase 1, Stage 2 is complete: RMSNorm, RoPE, batched
heads, SwiGLU, no biases.
**Measurements:** 811,904 params (-609). Best (step 36,500): val transcript 1.270, val book
1.347, vs 1.273 / 1.350 with biases: a tie. 24 min. 74 tests pass.
**Next:** Overfit-one-batch sanity check.

## 2026-10-07 · Phase 1 · Step: Overfit one batch
**Objective:** Classic sanity check: can the model memorize one tiny batch?
**What I did:** `test_gpt_can_memorize_a_single_batch` (small GPT, 300 steps). One-off demo:
the same check with a bug (no `loss.backward()`), and at real size on one real batch.
**What I learned:** A model that can't memorize one batch is broken, whatever the data.
The bug leaves the loss perfectly flat. A real batch levels off just above 0, likely
because the first positions of a window have too little context to tell targets apart.
**Measurements:** Small: 3.195 -> 0.022 (bug: 3.195 -> 3.195). Real size: 4.734 -> 0.023
by step 250. 75 tests pass.
**Next:** Plot loss curves.

## 2026-10-07 · Phase 1 · Step: Loss curves
**Objective:** See training as charts, not number tables.
**What I did:** `pretrain/plot_losses.py` parses training printouts (old and new formats)
and draws a PNG; charts in `docs/curves/` for transcript-only and transcript + books runs.
**What I learned:** Running out of data looks like a flat val line with train still
sliding down (transcript only). More data keeps all lines falling together, bending
toward flat. The smaller transcript val set is visibly noisier.
**Measurements:** 77 tests pass. Phase 1, Stages 1 and 2 complete.
**Next:** Phase 1, Stage 3: BPE tokenizer, starting with a toy example.

## 2026-10-07 · Phase 1 · Stage 3 · Step 1: BPE basics
**Objective:** Learn a vocabulary of chunks instead of single characters.
**What I did:** `pretrain/bpe.py`: `pair_counts`, `merge`, `train_bpe` (starts from UTF-8
bytes, ids 0-255; each merge adds one token).
**What I learned:** Counting neighbour pairs alone discovers words and structure:
"Houston" became one token, " you " by merge 250, speaker labels like " CC\n".
**Measurements:** First 50,000 transcript characters, 300 merges: 2.52 characters per
token, ~6 s. 82 tests pass.
**Anomalies:** Tokens cross word and punctuation boundaries (". Th", " CMP\nRoger"), and
rescanning the whole text for every merge is too slow for the full data.
**Next:** Split text into words before merging (pre-tokenizer).

## 2026-10-07 · Phase 1 · Stage 3 · Step 2: Pre-tokenizer
**Objective:** Keep merges inside words, and make training fast enough for real data.
**What I did:** `split_words` (GPT-2-style rule with the built-in `re`: contractions,
letters, digits, punctuation, whitespace); `train_bpe` merges inside pieces only and
counts each distinct word once, weighted by frequency.
**What I learned:** Tokens became real word pieces (" the", " Houston", " Apollo") instead
of junk like ".\n\n00 0". Compression dips slightly (2.52 -> 2.20 chars/token on 50k chars)
because the junk merges were common.
**Measurements:** Full transcript: 234,470 word pieces, 7,850 distinct; 300 merges in
11.7 s. 86 tests pass.
**Anomalies:** Encoding text by trying every merge on every word occurrence is slow.
**Next:** BPETokenizer with encode/decode.

## 2026-10-07 · Phase 1 · Stage 3 · Step 3: BPETokenizer
**Objective:** A usable tokenizer: train, encode, decode.
**What I did:** `BPETokenizer` in `pretrain/bpe.py`: vocab built from merges, encode
applies the earliest-learned merge first within each word, with a per-word cache.
**What I learned:** Starting from bytes means nothing is unknown: unseen characters
("ΔV", emoji) are spelled out byte by byte, and decode(encode(text)) is exact.
**Measurements:** Transcript, vocab 512: train 9.9 s, encode 826k chars in 0.26 s,
2.02 chars/token, exact round trip. Longest tokens: " Houston", "COLUMBIA", " Apollo".
89 tests pass.
**Next:** Compare with GPT-2's tokenizer on aerospace terms.

## 2026-10-07 · Phase 1 · Stage 3 · Step 4: Compare with GPT-2's tokenizer
**Objective:** How does a tokenizer trained on NASA text cut aerospace terms?
**What I did:** `pretrain/compare_tokenizers.py`: ours (vocab 2,048, training split only)
vs GPT-2 (vocab 50,257, via tiktoken).
**What I learned:** A tokenizer reflects its training text: ours gives "Grumman" one token
(GPT-2: 3) despite a 25x smaller vocab. Both shred acronyms and formulas (S-IVB, LOX/LH2,
N2O4: 4-6 tokens). Ours never saw "Δ" or "₂": the book cleaning dropped non-ASCII.
**Measurements:** chars/token on held-out text: transcript 2.66 vs 3.44, book 3.07 vs 4.59
(ours vs GPT-2). Training ours: 271 s.
**Anomalies:** Revisit the non-ASCII stripping in Phase 2 now that BPE is byte-level.
**Next:** Save/load the tokenizer and train the GPT on BPE tokens (vocab 1,024).

## 2026-10-07 · Phase 1 · Stage 3 · Step 5: Save and load the tokenizer
**Objective:** Train the tokenizer once; reuse it in every run.
**What I did:** `BPETokenizer.save/load` (the ordered merge list as JSON);
`pretrain/train_tokenizer.py` trains a 1,024-token vocab on the training split only.
**What I learned:** A BPE tokenizer is fully described by its ordered merges. A bigger
vocab costs model parameters (input table + output layer), and per-token losses are not
comparable to per-character ones.
**Measurements:** 141 s to train; 4.79M chars -> 1.91M tokens (2.51 chars/token);
13 KB file. 90 tests pass.
**Next:** Train the GPT on BPE tokens; report loss per character too.

## 2026-10-07 · Phase 1 · Stage 3 · Step 6: Train the GPT on BPE tokens (code)
**Objective:** Switch the model from characters to 1,024 BPE tokens.
**What I did:** `train.py` loads `checkpoints/tokenizer_1024.json` and reports every loss
per character (`loss_per_char`), so results compare with the character model.
Checkpoints store the tokenizer's merges; `sample.py` uses them (`--tokens`).
**What I learned:** Per-token and per-character losses measure different things; divide by
characters per token to compare tokenizers fairly.
**Measurements:** 92 tests pass. The 40,000-step run's results go in the next entry.
**Anomalies:** Piping training output through `tee` buffers it until the end; use `-u`.

## 2026-10-07 · Phase 1 · Stage 3 · Step 6 results: BPE vs characters (Stage 3 done)
**Measurements:** Best (step 36,500), loss per character: val transcript 1.184, val book
1.279, train 0.908 (characters: 1.270 / 1.347 / 1.119). 1,049,216 params (characters:
811,904). 25 min. Chart: `docs/curves/chars_vs_bpe.png`.
**What I learned:** A clear win, well above the noise, but three things changed together:
a 30% bigger model (vocab rows), ~160 characters of context instead of 64, and ~2.5x
more text per step. The train/val gap widened (each training token seen ~86 times).
**Anomalies:** The model reproduces PDF scan stamps ("ORBIGINAL PAGE"): leftover noise
from the book cleaning.
**Next:** Phase 1, Stage 4: learning-rate schedule (warmup + cosine decay).

## 2026-10-07 · Phase 1 · Stage 4 · Step 1: Learning-rate schedule
**Objective:** Warmup + cosine decay instead of a constant step size.
**What I did:** `learning_rate()` in `train.py` (1,000-step linear warmup, cosine to 1/10 of
the 1e-3 peak); set on the optimizer before every step. Three tests.
**What I learned:** Warmup starts slower and catches up by ~5,000 steps. The late small
steps improved training fit more than validation: with 1.9M training tokens the limit is
the data again, not the optimizer.
**Measurements:** Best (step 31,000): val transcript 1.174, val book 1.276 vs 1.184 /
1.279 constant (a tie). Final train 0.884 vs 0.900. Chart: `docs/curves/lr_schedule.png`.
95 tests pass.
**Next:** Gradient clipping.

## 2026-10-07 · Phase 1 · Stage 4 · Step 2: Gradient clipping
**Objective:** Cap the size of any single update.
**What I did:** Hand-written `clip_gradients` (global norm; scales all gradients by the same
factor if above the limit), matched against `nn.utils.clip_grad_norm_`; `train_step` takes
`max_grad_norm`, set to 1.0 in the config. Four tests.
**What I learned:** Clipping caps a step's length without changing its direction. Our
small model never needs it: norms stay well under 1.0. It is insurance for larger runs.
**Measurements:** Grad norm over the first 3,000 steps: median 0.31 -> 0.55, max 0.61,
0% of steps above 1.0. Full 40,000-step rerun skipped (it would reproduce the last one).
99 tests pass.
**Next:** Clean the remaining PDF noise in the books.

## 2026-10-07 · Phase 1 · Stage 4 · Step 3: Clean book back matter, stamps, footnotes
**Objective:** Remove the non-prose the model was imitating.
**What I did:** `is_back_matter` (labels such as NOTES TO PAGES / SOURCE NOTES / INDEX,
>12% digits, or 8+ numbered entries), `remove_scan_stamps`, `remove_footnote_markers`.
Rules were checked against what they dropped and pages near each threshold.
**What I learned:** About a quarter of the book text was never prose: endnotes, indexes,
tables. Some prose appendices are lost as a side effect.
**Measurements:** Books 4.60M -> 3.54M chars; stamps 31 -> 0, glued footnotes 658 -> 0,
notes pages ~108 -> 0. 107 tests pass. Model not retrained yet.
**Next:** Running headers and page numbers, then retrain tokenizer and model.

## 2026-10-07 · Phase 1 · Stage 4 · Step 4: Strip running headers and page numbers
**Objective:** Remove book and chapter titles repeated at page edges.
**What I did:** `find_running_heads` (a phrase at the start or end of 5+ pages that
continues the same way on 80%+ of them) and `strip_running_heads` (removes it and nearby
page numbers). Headers are discovered per book, not hard-coded.
**What I learned:** A frequency rule alone made "MOONPORT The" a header and would have
deleted real words; the "continues the same way" check fixed it.
**Measurements:** "CHARIOTS FOR APOLLO" ~164 -> 5, "Where No Man Has Gone Before" ~175 -> 6,
pages starting/ending with a page number -> 0. Books 3.54M -> 3.52M chars. 111 tests pass.
**Anomalies:** Left for Phase 2: unnumbered page-bottom footnotes, words split across
pages, photo-caption junk, partial chapter titles.
**Next:** Retrain the tokenizer and model on the cleaned text.

## 2026-10-07 · Phase 1 · Stage 4 · Step 5: Retrain on the cleaned books
**Objective:** See what the cleaning does to the model.
**What I did:** Retrained the tokenizer (vocab 1,024) and the model (40,000 steps) on the
cleaned text. Kept the noisy tokenizer and model for comparison.
**What I learned:** The clean tokenizer swapped citation words (" interview", " Report",
" 1966") for everyday English (" should", " might", " these"). But cleaner is not
automatically better on our metric.
**Measurements:** Train tokens 1.91M -> 1.43M; 2.61 chars/token (was 2.51). Best (step
23,000): val transcript 1.252 (noisy: 1.174, same text, worse by 0.08), val book 1.277
(noisy: 1.276, but the book text changed). Train/val gap 0.40 vs 0.29.
**Anomalies:** Unexplained transcript regression. Hypotheses: less data, or the removed
pages were useful practice for numbers (timestamps).
**Next:** Score both models on transcript timestamps vs spoken words.

## 2026-10-07 · Phase 1 · Stage 4 · Step 6: Diagnose the transcript regression
**Objective:** Find where the cleaned-data model loses on the transcript.
**What I did:** Scored both models on the transcript validation text, split into header
lines (timestamp + speaker, 14% of characters) and spoken lines; compared how each
tokenizer cuts a header line.
**What I learned:** Spoken lines are unchanged (1.117 vs 1.118); header lines got much
worse (1.275 -> 1.972). Removing number-heavy back matter left the tokenizer with only
some two-digit numbers as tokens (67 -> 45 number tokens), so "29" became "2"+"9" while
"03" stayed whole: inconsistent pieces for the same kind of field.
**Next:** Split numbers into single digits (as Llama does), retrain, rerun this check.

## 2026-10-07 · Phase 1 · Stage 4 · Step 7: Split numbers into single digits
**Objective:** Fix the transcript regression by cutting every number the same way.
**What I did:** Changed the pre-tokenizer so each digit is its own piece (as in Llama).
Retrained the tokenizer (2.57 chars/token) and the model (40,000 steps).
**What I learned:** Consistent number pieces let the model learn that timestamps count
upward. Also: the splitting rule lives in code, not in the tokenizer file, so changing it
silently changed how the older saved models read text.
**Measurements:** Best (step 25,500): val transcript 1.120 (cleaned 1.252, noisy 1.174),
val book 1.272. Header lines 0.965 (cleaned 1.972, noisy 1.275), spoken lines 1.111.
112 tests pass. Curve: `docs/curves/clean_digits.png`.
**Anomalies:** A rerun of the diagnostic scored the older models with the new splitting
rule and gave invalid numbers (2.869, 3.566); the figures above are from before the change.
**Next:** Save the splitting rule inside the tokenizer file and checkpoints.

## 2026-10-07 · Phase 1 · Stage 4 · Step 8: Save the splitting rule with the tokenizer
**Objective:** Stop old tokenizers being silently read with a newer splitting rule.
**What I did:** The tokenizer keeps its `pattern` and saves it next to the merges, in the
tokenizer file and in checkpoints. Loading uses the saved rule; a file without one is
refused. Stamped the existing local files with the rule each was trained with.
**What I learned:** A tokenizer is the rule *and* the merges. Real tokenizer files store
the pre-tokenizer for the same reason.
**Measurements:** The diagnostic now reproduces the valid figures for the older models
(header lines 1.275 and 1.972) with no special handling. 114 tests pass.
**Next:** Add general English text (FineWeb-Edu).

## 2026-10-08 · Phase 1 · Stage 4 · Step 9: Mix in general English (FineWeb-Edu)
**Objective:** Give the model far more text to learn English from, without losing NASA.
**What I did:** `data/prepare_fineweb_edu.py` takes one FineWeb-Edu file (license: ODC-By,
logged), keeps ASCII-only documents (78%), and holds out every 100th document: 327M
training characters (~87x our NASA text). Training samples each window from NASA or web
text with chosen weights (`get_mixed_batch`); tokens are stored in 2 bytes each. Tokenizer
and model unchanged, so only the data differs.
**What I learned:** The mixture matters more than the amount. Plain concatenation gives
NASA 1.1% of training and makes NASA text worse. Giving NASA half the windows makes both
NASA validation sets better than NASA-only: general English helps, and the web text
stops the model memorizing the small NASA set.
**Measurements:** 40,000 steps, val transcript / val book (NASA-only: 1.120 / 1.272):
natural 1.365 / 1.302; 20% 1.165 / 1.206; 50% 1.095 / 1.188; 80% 1.087 / 1.204.
50% for 300,000 steps (3.6 h): best 1.054 / 1.154 at step 287,000; web val 1.236 with
train web 1.231. Curves: `docs/curves/web_mixtures.png`, `docs/curves/web_nasa50_300k.png`.
**Anomalies:** 7.5x more steps bought only ~0.04, and train and val web loss are equal:
the 1M-parameter model, not the data, is now the limit. Documents are joined with blank
lines, no end-of-document marker. The tokenizer was trained on NASA text only.
**Next:** Scale the model up now that there is data to support it.

## 2026-10-08 · Phase 1 · Stage 4 · Step 10: A bigger model (11M parameters)
**Objective:** Test whether model size is now the limit.
**What I did:** Added the `gpt-11m` config: 384 channels, 6 blocks, 6 heads of 64; all
other settings as `gpt`. Trained 40,000 steps on the 50/50 NASA/web mixture.
**What I learned:** Size was the limit: by step 10,000 (22 min) the 11M model matched
the 1M model's 3.6-hour run. But a bigger model also memorizes faster. With half of every
batch drawn from only 1.45M NASA tokens (56 passes over them by step 40,000), NASA train
loss kept falling while NASA validation rose: overfitting. The web text, ~100x larger,
shows no gap at all.
**Measurements:** Best (step 10,000): val transcript 1.056, val book 1.167 (1M at 40,000:
1.095 / 1.188; 1M at 300,000: 1.054 / 1.154). Step 40,000: NASA train 0.354, val
transcript 1.208, val book 1.321; web train 1.159, val 1.161. 11M is 2.7x slower per step
(91 vs 34 ms). Curve: `docs/curves/gpt11m_vs_1m.png`.
**Anomalies:** The best-checkpoint rule saved the step-10,000 model, so nothing was lost,
but most of the run was wasted.
**Next:** Fit the NASA weight to the model: fewer repeats of the small NASA text.

## 2026-10-08 · Phase 1 · Stage 4 · Step 11: Fit the NASA share to the bigger model
**Objective:** Find a NASA share that does not make the 11M model memorize NASA text.
**What I did:** Three 10,000-step runs of `gpt-11m`, each with its own full schedule:
10%, 25% and 50% NASA (about 2.8, 4.9 and 9.8 passes over the NASA text). No code changes.
**What I learned:** 25% is best overall. The two NASA sets pull different ways: the
transcript (an unusual format found nowhere else) wants more NASA, the book (ordinary
English prose) wants more web text. 25% is the balance, at about 5 passes, close to the
"about 4 passes" rule of thumb.
**Measurements:** val transcript / val book / val web: 10% 1.113 / 1.147 / 1.129;
25% 1.048 / 1.121 / 1.148; 50% 1.042 / 1.145 / 1.196. 25% beats the 1M model's 3.6-hour
run (1.054 / 1.154) in 23 minutes. All three were still improving at step 10,000.
**Anomalies:** 25% vs 50% on the transcript (0.006) is within run-to-run noise; the book
difference (0.024) is not.
**Next:** Train the tokenizer on NASA + web text instead of NASA alone.

## 2026-10-08 · Phase 1 · Stage 4 · Step 12: A tokenizer learned from NASA + web text
**Objective:** See whether a vocabulary that also fits general English helps.
**What I did:** `train_tokenizer.py --web` learns from all NASA training text plus web
text, 25% NASA by characters (15M characters; the Python trainer is too slow for all of
it). Vocabulary still 1,024. `train.py --tokenizer` picks the tokenizer for a run.
Compared both on held-out text, then trained `gpt-11m` 10,000 steps at 25% NASA with each.
**What I learned:** With only 768 learned tokens, NASA and web words compete for slots:
206 changed. Gained " information", " because", " people"; lost " spacecraft",
" astronauts", " mission", " CDR", "Houston". The model follows the vocabulary: better on
web text, worse on the transcript. For our goal (NASA text) the NASA-only tokenizer stays.
**Measurements:** chars per token (NASA-only -> mixed): transcript 2.19 -> 2.14, book
2.62 -> 2.59, web 2.40 -> 2.53. Model, val transcript / book / web: NASA-only tokenizer
1.048 / 1.121 / 1.148; mixed 1.074 / 1.125 / 1.139.
**Anomalies:** None. Both runs used the same seed, data and settings.
**Next:** A bigger vocabulary, so NASA and general words both fit.

## 2026-10-08 · Phase 1 · Stage 4 · Step 13: A bigger vocabulary (4,096 tokens)
**Objective:** Give NASA and general English words room to both fit in the vocabulary.
**What I did:** `train_tokenizer.py --vocab-size` (file names now say the size). Trained
a 4,096-token tokenizer on the same NASA + web sample (28 minutes), then `gpt-11m` for
10,000 steps at 25% NASA with it.
**What I learned:** With 3,840 learned tokens both sets fit: " spacecraft", " CDR",
"Houston", "EAGLE" and " because", " information" are all single tokens. Every
validation set improved by a lot. But three things changed at once: the vocabulary, the
parameter count (11.4M -> 13.8M: bigger input table and output layer), and the
characters per 64-token window (~1.4x more context and more text per step).
**Measurements:** chars per token (1,024 NASA-only -> 4,096 mix): transcript 2.19 ->
2.64, book 2.62 -> 3.61, web 2.40 -> 3.47. Model, val transcript / book / web:
1.048 / 1.121 / 1.148 -> 0.989 / 1.082 / 1.080. 7% slower per step.
**Anomalies:** The improvement cannot yet be split between vocabulary and longer context.
**Next:** A control run: the old tokenizer with a longer window, to separate the two.

## 2026-10-08 · Phase 1 · Stage 4 · Step 14: Separate vocabulary from context
**Objective:** Find out how much of the 4,096-vocabulary win came from longer context.
**What I did:** Added `train.py --block-size`. Control run: the old 1,024-token NASA
tokenizer with a 96-token window (about the same characters per window as 64 tokens of
the 4,096 tokenizer), otherwise identical (`gpt-11m`, 25% NASA, 10,000 steps).
**What I learned:** Both mattered, differently per text. On the transcript most of the
gain was context (0.047 of 0.059); on the book and web text, the vocabulary did more.
And the bigger vocabulary gets the same context much more cheaply: fewer, bigger tokens
mean less attention work per character.
**Measurements:** val transcript / book / web: 64-token baseline 1.048 / 1.121 / 1.148;
control (96 tokens) 1.001 / 1.108 / 1.117; vocab 4,096 0.989 / 1.082 / 1.080. Time for
10,000 steps: control 1,858 s, vocab 4,096 1,422 s.
**Anomalies:** The 4,096 model also has 2.4M more parameters (its bigger token tables),
so the remaining gap is vocabulary plus those parameters, not vocabulary alone.
**Next:** Make the winning recipe the default.

## 2026-10-08 · Phase 1 · Stage 4 · Step 15: Make the best recipe the default
**Objective:** Plain `python -m pretrain.train` should run the best recipe so far.
**What I did:** New defaults: `gpt-11m`, web text on (`--no-web` turns it off), 25% NASA,
the 4,096-token NASA + web tokenizer. `train_tokenizer.py` builds that tokenizer by
default. Moved argument parsing into `parse_args` so the defaults are tested. Earlier
recipes stay reachable with flags.
**What I learned:** A recipe is many settings that only work well together (model size,
mixture, vocabulary); defaults should hold the combination that was measured.
**Measurements:** 132 tests pass. A 500-step smoke run of the plain command trains the
13.8M-parameter model on the 25/75 mixture.
**Next:** Reproduce a small nanochat run as a reference check.

## 2026-10-08 · Phase 1 · Stage 4 · Step 16: Run nanochat's small reference recipe
**Objective:** Run a known-good trainer on our GPU, to have an outside yardstick for ours.
**What I did:** Cloned nanochat outside the repo, gave it its own venv, downloaded 9
ClimbMix shards and trained its 32,768-token tokenizer. Ran its small recipe (depth 6, 384
channels, 512-token context, 16,384 tokens per step, 5,000 steps) with no code changes,
only settings: `TORCHDYNAMO_DISABLE=1` (no `torch.compile` on Pascal), `PYTHONUTF8=1` (the
Windows console), and 8 sequences per micro-batch with 4× gradient accumulation to fit in
8 GB with the same math. Logged ClimbMix in the license log as reference-only, excluded.
**What I learned:** Its transformer core matches our `gpt-11m` (6 layers, 384 channels,
6 heads of 64, about 10.6M matrix parameters); its 73.5M total is mostly token tables.
Its curve was still falling at the end, smoothly, with no sign of memorizing. The samples
are grammatical but loop ("The capital of France is the capital of France").
**Measurements:** val bits per byte 3.196 at step 0, 1.372 at 1,000, 1.223 at 3,000,
1.164 at 5,000 (the minimum). 74.3 min, 0.87 s per step, 18,800 tokens/s, peak memory
3.9 GB. Ours, converted (nats per char ÷ 0.693): best web val 1.080 → 1.56 bits per byte.
**Anomalies:** Not a fair race yet: different text (ClimbMix vs FineWeb-Edu), vocabulary
(32,768 vs 4,096), context (512 vs 64 tokens) and amount read (82M vs 41M tokens).
**Next:** Put both trainers on the same data, so the gap can be split into its causes.

## 2026-10-08 · Phase 1 · Stage 4 · Step 17: Give nanochat our data
**Objective:** Let nanochat train and validate on exactly our FineWeb-Edu documents.
**What I did:** Added `data/export_nanochat.py`: it writes our train documents as numbered
parquet shards and our val documents as the last shard, the layout nanochat expects. Moved
the document loading in `prepare_fineweb_edu.py` into `load_documents()` so both scripts
read the same documents. nanochat finds the shards via `NANOCHAT_BASE_DIR`, again with no
code changes to it.
**What I learned:** Our `train.txt` can't be split back into documents (documents contain
blank lines themselves), so the export rebuilds them from the downloaded file instead.
**Measurements:** 133 tests pass. 85,041 train documents in 9 shards, 860 val documents in
the last; joined back together they equal our `train.txt` and `val.txt` character for
character. 187 MB, exported in 2.4 min. nanochat's own reader returns the same first
documents.
**Next:** Train nanochat's tokenizer and model on these shards, and compare its val bits
per byte with our web val directly.

## 2026-10-08 · Phase 1 · Stage 4 · Step 18: Train nanochat on our data
**Objective:** See how nanochat's small recipe scores on our FineWeb-Edu text.
**What I did:** Trained nanochat's 32,768-token tokenizer on our train shards, then its same
5,000-step recipe as Step 16, changing only the data (`NANOCHAT_BASE_DIR`).
**What I learned:** Our text is about as hard for it as its own: the score barely moved
(1.164 to 1.162). Its tokenizer trainer (Rust) took 13 s; ours (Python) took 28 min for
4,096 tokens. nanochat starts every sequence at a document start and crops what doesn't
fit, so at 512 tokens it mostly reads document beginnings, and it went through those
about 2.5 times (passes began at steps 2,013 and 4,057) with no sign of memorizing.
**Measurements:** val bits per byte 3.135 at step 0, 1.333 at 1,000, 1.196 at 3,000,
1.162 at 5,000 (the minimum). 72.8 min, peak memory 3.9 GB. Ours on the same val
documents: 1.56 bits per byte.
**Anomalies:** Still not graded identically: nanochat scores the first 512 tokens of
each val document (and only the first ~524K tokens of val), while we score random
64-token windows from anywhere in the text.
**Next:** One scoring script that grades both models on the whole val text the same way.

## 2026-10-09 · Phase 1 · Stage 4 · Step 19: Grade every model the same way
**Objective:** One exact score for any model, ours or nanochat's, on the whole web val text.
**What I did:** Added `pretrain/score_bpb.py`. It scores all 860 val documents, every token
exactly once, each document from its own start marker (ours: a blank line; nanochat's:
`<|bos|>`), reading long documents in windows that overlap by half. It reports bits per
byte, which doesn't depend on the tokenizer. `--context` shows a model fewer tokens than it
was trained on. It runs in nanochat's venv too, so nanochat's own code loads its models.
**What I learned:** Context explains most of the gap. nanochat, limited to 47 tokens
(about the 222 bytes our 64 tokens cover), scores 1.577: worse than ours. With more context
it improves steadily, down to 1.199 at 512. Our run with 256 tokens also beat our 64-token
run clearly. Longer windows also mean NASA text is reread faster: NASA val got worse after
step 6,000 while web val kept improving to the end.
**Measurements:** bits per byte on the whole web val text:

| Model | Context | Bits per byte |
|---|---|---|
| ours, vocab 1,024, 10k steps | 64 tokens | 1.613 |
| ours, vocab 1,024, 10k steps | 96 tokens | 1.566 |
| ours, vocab 4,096, 10k steps | 64 tokens | 1.522 |
| ours, vocab 4,096, best NASA checkpoint (step 6,000 of 40,000) | 256 tokens | 1.438 |
| nanochat, trained on our data, limited | 47 / 64 / 128 / 256 tokens | 1.577 / 1.480 / 1.325 / 1.242 |
| nanochat, trained on ClimbMix | 512 tokens | 1.215 |
| nanochat, trained on our data | 512 tokens | 1.199 |

The 1M model after 300k steps: 1.744. The 256-token run: 6.1 h; its web val reached 0.955
nats per character (about 1.38 bits per byte on random windows) at step 40,000, but that
checkpoint wasn't kept.
**Anomalies:** I meant the 256-token run to be 10,000 steps but forgot `--max-steps`, so it
ran the config's 40,000 and saved the checkpoint best on NASA val, not web. Renamed it
`gpt_11m_nasa25_block256_40k_best6k.pt`. Scoring nanochat at short contexts is a little
unfair to it: it always trained with `<|bos|>` in view, which mid-document windows lack.
**Next:** Lengthen our context for real, and keep the NASA share from being memorized.

## 2026-10-09 · Phase 1 · Stage 4 · Step 20: Make the long context the default
**Objective:** Use what the scorer found: give our model a longer view of the text.
**What I did:** The `gpt-11m` recipe now reads 256-token windows (about 890 characters,
was 64 tokens) for 6,000 steps: 16,384 tokens per step, the same as nanochat. Ran the plain
default command and scored the result on the whole web val text.
**What I learned:** Letting the learning rate wind down over the run's real length beats
stopping a longer run partway: 1.423 here vs 1.438 for step 6,000 of the 40,000-step run.
With 6,000 steps, NASA val was still about level at the end, no longer rising.
**Measurements:** best at step 5,500: val transcript / book / web 0.917 / 1.072 / 1.009 per
character (64-token recipe, 10,000 steps: 0.989 / 1.082 / 1.080). Whole web val: 1.423
bits per byte (was 1.522; nanochat 1.199). 57 min (was 24 min). 156 tests pass.
**Next:** The remaining gap to nanochat at 512 tokens: try 512 tokens, then look at its
optimizer (Muon) and its bigger vocabulary.

## 2026-10-09 · Phase 1 · Stage 4 · Step 21: Try nanochat's 512-token context
**Objective:** Find how much of the remaining gap to nanochat is still context.
**What I did:** Added `train.py --batch-size`, and `recipe()`, which applies any given
options over a config (tested). Ran 512-token windows with 32 per step, so the same
16,384 tokens per step as the default, and scored the result.
**What I learned:** Little. Doubling the context again gained 0.011 bits per byte, at 1.5x
the time. At this size, our model gets most of what context gives by 256 tokens; nanochat
gained more (1.242 to 1.199) over the same step. The rest of the gap is in how the model
is built and trained, not how far it sees. 256 stays the default.
**Measurements:** best at step 5,500: val transcript / book / web 0.909 / 1.069 / 0.995
(256 tokens: 0.917 / 1.072 / 1.009). Whole web val 1.412 bits per byte (256 tokens:
1.423; nanochat: 1.199). 85 min (256 tokens: 57 min). 157 tests pass.
**Next:** nanochat's optimizer, Muon, the biggest remaining difference in training.

## 2026-10-09 · Phase 1 · Stage 4 · Step 22: Add the Muon optimizer
**Objective:** Build nanochat's optimizer for weight grids, on its own, before using it.
**What I did:** Added `pretrain/muon.py`: `orthogonalize` (five rounds of a Newton-Schulz
polynomial that evens out the strength of every direction in a matrix) and `Muon` (Nesterov
momentum, then orthogonalize, with a size scale for tall grids). The original version,
without nanochat's later refinements. Not wired into training yet.
**What I learned:** A gradient's directions can differ a lot in strength; plain momentum
mostly moves the strong ones. Muon keeps the directions but steps equally along all of
them, using only matrix multiplications (fast on a GPU), no SVD.
**Measurements:** a gradient with direction strengths from 1 to 10 comes out with all of
them between 0.68 and 1.13, for square, wide and tall shapes. Muon fits a 16x16 linear map
to under 5% of its starting loss in 200 steps. 163 tests pass.
**Next:** Train with Muon for the transformer's weight grids and AdamW for the rest, and
compare against the 256-token default (1.423 bits per byte).

## 2026-10-09 · Phase 1 · Stage 4 · Step 23: Train with Muon
**Objective:** Measure what nanochat's optimizer does for our model.
**What I did:** `build_optimizer` gives the blocks' 36 weight grids to Muon and the rest
(token table, output layer, norm weights) to AdamW; `Optimizers` steps both as one, and
`set_learning_rate` scales each from its own peak. `--muon-lr` turns it on (off: training
as before). Ran the default recipe with `--muon-lr 0.02` and scored it.
**What I learned:** Muon was ahead at every report, from step 500 on, on all three texts.
It beat doubling the context (1.412) for less extra time. Each step costs ~13% more for
the orthogonalizing.
**Measurements:** best at step 5,500: val transcript / book / web 0.891 / 1.045 / 0.986
(AdamW only: 0.917 / 1.072 / 1.009). Whole web val 1.387 bits per byte (AdamW: 1.423;
nanochat: 1.199). 67 min (AdamW: 57 min). 167 tests pass.
**Anomalies:** One step size tried for Muon (0.02, the usual value); not tuned.
**Next:** Make Muon part of the default recipe.

## 2026-10-09 · Phase 1 · Stage 4 · Step 24: Make Muon the default
**Objective:** Plain `python -m pretrain.train` should use the best recipe measured.
**What I did:** The `gpt-11m` recipe now sets `muon_lr` 0.02; `--muon-lr 0` switches back
to AdamW for everything. Older recipes (e.g. `--model gpt`) still train with AdamW only.
**What I learned:** A default is a promise that the combination was measured: this one
was, in Step 23 (1.387 bits per byte vs 1.423).
**Measurements:** 168 tests pass. No new run: the Step 23 run is this exact recipe.
**Next:** Remaining gap to nanochat (1.387 vs 1.199): its bigger vocabulary, then its
architecture details (ReLU², QK norm, zero-init projections, logit softcap).

## 2026-10-10 · Phase 1 · Stage 4 · Step 25: Overnight checks of Muon's step size and vocabulary
**Objective:** Test the two open questions from Steps 23-24 with existing code.
**What I did:** Ran the default recipe with Muon step sizes 0.01 and 0.04 (0.02 was run
in Step 23); trained an 8,192-token tokenizer (`train_tokenizer --vocab-size 8192`) and ran
the default recipe with it. All scored with `score_bpb`.
**What I learned:** 0.02 was the right step size: 0.04 jumped ahead early, then stalled
(too big to settle), and 0.01 learned slower and was overtaken. The bigger vocabulary
tied: it learned web text better by the end, but its bigger token tables memorized NASA
sooner, so its best NASA checkpoint came at step 4,000. The limit now is how little NASA
text we have, which Phase 2 addresses.
**Measurements:** best checkpoint, NASA val avg (transcript + book) / whole web val bits
per byte: Muon 0.01: 0.977 / 1.414 (step 4,000); 0.02: 0.968 / 1.387 (step 5,500); 0.04:
1.023 / 1.552 (step 6,000). Vocab 8,192: 0.970 / 1.398 (step 4,000), web val at step
6,000 0.964 vs 0.982; 3.89 characters per token (4,096: 3.44); tokenizer 57 min, run
97 min (4,096: 67 min).
**Anomalies:** One run per setting: differences under ~0.01 may be noise.
**Next:** Remaining gap to nanochat (1.387 vs 1.199): its architecture details (ReLU²,
QK norm, zero-init projections, logit softcap), one at a time.
