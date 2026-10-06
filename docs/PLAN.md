# Artemis: LLM Project Plan

*Started Oct 6, 2026 · @Justin*

## Overview

Artemis is an aerospace-specialist language model system built in two tracks:

1. **Artemis I:** a tiny model pretrained from scratch, to learn how LLMs actually work.
2. **Artemis II:** a fine-tuned open-weight model wrapped in a mission-control style
   verification layer, to produce something genuinely useful.

The goal is equal parts education and portfolio piece, so every phase ends in something
showable. Three things make it distinct: a domain corpus built from NASA public-domain
material, a multi-console go/no-go architecture that visibly refuses when its specialists
disagree, and a self-written, published benchmark (FLIGHT-Bench).

**Success looks like:**

- A from-scratch model (Artemis I) explained line by line, with published training curves.
- A fine-tuned model (Artemis II) that beats its own base model and a same-size general
  model on FLIGHT-Bench.
- A public repo, model cards, Hugging Face release, live demo, and a writeup per phase.
- Skills that map to the AI engineer path: data pipelines, training, post-training, RAG,
  agents, evals, and serving.

**What it is not:** a competitor to frontier models on general ability. At hobby scale the
win is narrow depth plus rigor you can prove.

## Architecture

Artemis is two models and one runtime.

- **Artemis I:** a small model in the nanochat size class, pretrained from scratch on
  general text plus the NASA corpus. It exists to teach the internals.
- **Artemis II:** a 4B to 9B open-weight model with continued pretraining on the corpus,
  then SFT, DPO, and GRPO. It powers the runtime.
- **Runtime:** consoles FIDO (math), GUIDO (facts with retrieval), and SURGEON (safety and
  sanity), polled by the Flight Director. Integrated and tested in Artemis III, launched
  publicly in Artemis IV.

Release mapping (after NASA's February 2026 restructure): Artemis I = from-scratch model
(uncrewed test flight), Artemis II = fine-tuned model (first crewed flight, lunar flyby),
Artemis III = console integration tested before going public (Earth-orbit docking test,
2027), Artemis IV = public launch of the full system (first landing, 2028).

The poll is the differentiator: when consoles disagree, Artemis reports the disagreement
instead of guessing. Each console is the same model with a different system prompt and
toolset, so the system costs one model to host. Simple queries can skip the full poll.

## Research reading list

Study each topic right before the phase that uses it. Reading is done when you can explain
the idea to someone else and point to where it lives in your code.

| Topic | Why it matters here | Core resources | Phase |
|---|---|---|---|
| Transformer internals | You write attention, MLP, and norm layers by hand | Vaswani 2017; Karpathy *Zero to Hero* GPT lecture; Raschka, *Build a LLM (From Scratch)* | 1 |
| Modern architecture deltas | RoPE, RMSNorm, SwiGLU, GQA, no biases | Llama 3 paper; nanochat source | 1 |
| Tokenization | Aerospace text is full of acronyms, units, equations | Karpathy "Let's build the GPT tokenizer"; SentencePiece and BPE papers | 1 |
| Optimizers and stability | Speedrun community moved past plain AdamW | modded-nanogpt and Muon writeups; warmup, cosine, WSD schedules | 1 |
| Scaling laws | How many tokens your model size should see | Chinchilla (Hoffmann 2022); Kaplan 2020 for contrast | 1 |
| Data curation | Data quality beats model size at small scale | FineWeb / FineWeb-Edu paper; datatrove; MinHash (Broder) | 2 |
| PDF extraction | Most of NTRS is PDFs with tables and equations | Marker, Docling, Nougat; compare on 20 reports | 2 |
| Parameter-efficient fine-tuning | Fine-tune 4B to 9B on one GPU | LoRA (Hu 2021); QLoRA (Dettmers 2023); Unsloth, TRL docs | 3 |
| Continued pretraining vs SFT | SFT shapes behavior; knowledge needs CPT or retrieval | "Don't Stop Pretraining" (Gururangan 2020) | 3 |
| Preference and RL post-training | DPO after SFT; GRPO for verifiable reasoning | DPO (Rafailov 2023); DeepSeekMath (GRPO); DeepSeek-R1 | 3 |
| RAG | Grounds GUIDO in real documents | RAG (Lewis 2020); hybrid BM25 + dense; reranking | 4 |
| Self-verification, multi-agent | Theory behind the go/no-go poll | Self-consistency (Wang 2022); debate (Du 2023); Chain-of-Verification (2023) | 4 |
| Evaluation methodology | The benchmark is the project's credibility | lm-evaluation-harness; contamination papers; LLM-as-judge pitfalls | 5 |
| Inference and serving | Fast, cheap demo hosting | vLLM / PagedAttention; llama.cpp, GGUF | 6 |

## Data plan

The NASA Technical Reports Server (NTRS) is the anchor: 500k+ citations, 200k+ full-text
documents, a public OpenAPI with no key, and bulk metadata as yearly ndjson files.

| Source | Content | Licensing note |
|---|---|---|
| NTRS | Technical reports, memos, conference papers, NACA reports back to the 1940s | Keep NASA-authored work only. Contractor reports and journal articles can carry third-party copyright. Cite the STI Program as source. |
| Apollo Flight Journal and mission transcripts | Air-to-ground and flight director loop | Transcripts are public domain; journal commentary is separately authored, so check first |
| Artemis I and II press kits, blogs, reference guides | SLS, Orion, EGS, mission phases | NASA-produced, generally public domain |
| NASA Thesaurus | 18,000+ controlled subject terms | Tokenizer vocab checks and eval topic coverage |
| FineWeb-Edu | General English | Artemis I pretraining only |

**Pipeline**

1. Harvest bulk ndjson metadata; filter by document type, NASA authorship, subject category.
2. Download PDFs through the API at a polite rate, logging every document ID and license decision.
3. Extract to markdown, keeping equations and tables. Benchmark two tools on 20 reports first.
4. Clean headers, page numbers, reference lists, OCR garbage. Score quality with heuristics,
   then a small classifier (FineWeb-Edu style).
5. Deduplicate: MinHash at document level, exact dedup on paragraphs.
6. Split by document, never by chunk. Hold out eval source documents entirely.
7. Publish the cleaned corpus on Hugging Face with a datasheet.

## Phased roadmap

Every phase ends in a public artifact. No phase starts until the previous gate passes.
Phases 1 and 2 can overlap.

### Phase 1 · Artemis I
- [ ] Reimplement a GPT from an empty file (Zero to Hero), then upgrade with RoPE, RMSNorm, SwiGLU.
- [ ] Train a BPE tokenizer on FineWeb-Edu + NASA text; compare aerospace splits vs a stock tokenizer.
- [ ] Reproduce a small-depth nanochat run, swap in your own model code, match its loss curve.
- [ ] Final run mixing general text with the NASA corpus. Publish curves and writeup 1.

### Phase 2 · Corpus and FLIGHT-Bench v1
- [ ] NTRS harvester and license log.
- [ ] Benchmark two PDF extractors on 20 reports; pick one.
- [ ] Cleaning, quality scoring, MinHash dedup; publish dataset with datasheet.
- [ ] Write and hand-verify FLIGHT-Bench v1 from held-out documents; freeze it.

### Phase 3 · Artemis II
- [ ] Choose base model; run it on FLIGHT-Bench for the baseline.
- [ ] Continued pretraining (LoRA first, full only if it clearly helps).
- [ ] SFT on generated and hand-written Q&A, then DPO on preference pairs.
- [ ] GRPO on quantitative items with a programmatic reward. Evaluate after every stage.

### Phase 4 · Consoles and retrieval (Artemis III)
- [ ] Hybrid BM25 + vector index with reranking for GUIDO.
- [ ] Code-execution tool for FIDO; claim checking against retrieved passages for SURGEON.
- [ ] Flight Director as a LangGraph graph: route, run consoles in parallel, poll, decide.

### Phase 5 · Full evaluation
- [ ] Run every baseline and every layer of the stack on FLIGHT-Bench.
- [ ] Failure analysis: the ten worst answers and why.

### Phase 6 · Launch (Artemis IV)
- [ ] Quantize and serve; demo UI with a visible poll; model cards; final writeup.

## Evaluation: FLIGHT-Bench

Build v1 **before** any fine-tuning. Target 300 to 500 items.

- **Factual recall:** short answers from held-out NTRS docs and press kits. Exact or normalized match.
- **Quantitative reasoning:** delta-v, Hohmann, Isp, T/W, unit conversions. Graded
  programmatically; doubles as the GRPO reward.
- **Grounded QA:** must cite a document. Graded on correctness and citation support.
- **Should-refuse:** false premises or no answer in corpus. Where the poll should shine.

**Baselines:** base model; same-size general model; fine-tune alone; fine-tune + retrieval;
full console system; one frontier model via API as a ceiling.

**Metrics:** accuracy per category, citation precision, refusal rate on unanswerable items,
false-refusal rate on answerable ones, cost and latency per query.

**Rigor:** hand-verify a random 10%, keep eval sources out of training, version the
benchmark, validate any LLM judge against your own grades on 50 items.

## Compute and budget

Roughly $400 to $800 of rented GPU time if development stays small. H100s run about $2 to
$4 per GPU-hour (Lambda, RunPod, Vast.ai). Set a hard spend alert before the first run.

| Phase | Workload | Hardware | Rough cost |
|---|---|---|---|
| 1. Artemis I | Debug runs, then one speedrun-style run (nanochat ref ≈ $100) | Local GTX 1080 for debugging; 8xH100 for 3 to 4 h | $100 to $200 |
| 2. Data | Extraction and cleaning | CPU, optionally 1 GPU | $0 to $50 |
| 3. Artemis II | CPT + QLoRA SFT, DPO, GRPO on 4B to 9B | 1x H100 (local 1080 is too small) | $150 to $400 |
| 4 to 5. Consoles, evals | Benchmark inference, frontier API calls | 1 GPU + API credits | $50 to $100 |
| 6. Demo | Quantized model behind a small API | Small GPU or CPU with GGUF | $0 to $30/mo |

**Local hardware:** GTX 1080, 8 GB, Pascal (compute 6.1). Good for writing and debugging
Artemis I at tiny scale in fp32. No bf16, no `torch.compile`/Triton, too small for 4B+
fine-tuning, so Phase 3 runs in the cloud.

## Showcase strategy

- A model card per release: training data, compute, eval results, known failure modes.
- Corpus and FLIGHT-Bench on Hugging Face, each with a datasheet.
- Training curves and run logs (public W&B project or exported charts).
- A live demo where users watch the poll, including NO-GO outcomes.
- A writeup per phase, published as you go.

**Target pitch:** "I pretrained a small model from scratch to learn the internals, built a
cleaned aerospace corpus from NASA's technical archive, fine-tuned an open model on it, and
wrapped it in a multi-agent verification layer. On my published benchmark it beats its base
model by X points and cuts confident wrong answers by Y%."

## Risks

| Risk | Why it bites | Mitigation |
|---|---|---|
| Scope creep | Six phases alongside a full-time job | Each phase ships a standalone artifact |
| Fine-tune doesn't beat base + RAG | Common; SFT shapes behavior more than it adds facts | Report it honestly; it is a credible finding |
| PDF extraction quality | Old scanned equations and tables extract badly | Benchmark early; drop low-quality docs |
| Licensing mistakes | Not every NTRS doc is public domain | NASA-authored filter + per-document license log |
| Eval contamination | Leaked eval facts inflate scores | Hold out source documents, not just questions |
| Console latency and cost | Several model calls per query | Parallel consoles; skip poll for simple queries |
| Name collision | Many AI projects use "Artemis" | Distinct repo / Hugging Face handle |

## Open questions

- [x] Local GPU? **GTX 1080 8 GB:** local debugging only; real runs in the cloud.
- [ ] Base model for Artemis II (decide at the start of Phase 3; Apache-2.0 small models).
- [ ] Hours per week (sets the real timeline).
- [ ] Public from day one, or private until Artemis I ships?
- [ ] GitHub / Hugging Face handle that avoids the "Artemis" collision.
