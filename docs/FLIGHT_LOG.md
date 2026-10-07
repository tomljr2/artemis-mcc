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
