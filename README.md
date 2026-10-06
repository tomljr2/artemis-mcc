# Artemis

An aerospace-specialist language model, built in the open as a way to learn how LLMs
actually work, from the first line of attention code to a served multi-agent system.

Artemis is named after NASA's Artemis program. It is trained on NASA public-domain
material, and its runtime is modeled on Mission Control: specialist consoles each check
an answer, and a Flight Director runs a **go/no-go poll** before anything is released.
When the consoles disagree, Artemis says so instead of guessing.

> Status: **Pre-launch.** Phase 1 (Artemis I) is in work. See [docs/PLAN.md](docs/PLAN.md).

## Mission control architecture

```
                        ┌──────────────────────┐
         query ───────▶ │   FLIGHT DIRECTOR    │  routes, polls, decides
                        └──────────┬───────────┘
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
      ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
      │     FIDO     │     │    GUIDO     │     │   SURGEON    │
      │ math & orbit │     │  facts with  │     │ safety and   │
      │ (code tool)  │     │  citations   │     │ sanity check │
      │              │     │  (retrieval) │     │              │
      └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
             └──── GO / NO-GO ────┴──── GO / NO-GO ────┘
                                  ▼
                answer, or a reported disagreement (NO-GO)
```

| Console | Real MCC role | Artemis job |
|---|---|---|
| **FIDO** | Flight Dynamics Officer: trajectory and maneuvers | Quantitative reasoning: delta-v, Hohmann transfers, Isp, unit conversions. Checks its work by running code. |
| **GUIDO** | Guidance Officer: onboard navigation state | Factual grounding: answers only from retrieved NASA documents, with citations. |
| **SURGEON** | Flight Surgeon: crew health | Safety and sanity: checks claims against sources, catches false premises, flags overconfidence. |
| **FLIGHT** | Flight Director: final authority | Routes the query, runs the consoles in parallel, polls them, and calls GO or NO-GO. |

Each console is the same fine-tuned model with its own system prompt and tools, so the
whole system runs on one hosted model.

## Releases

Releases follow the Artemis mission sequence. Each one is a git tag and, where it applies,
a Hugging Face release with a model card.

| Release | Mission | What ships |
|---|---|---|
| **Artemis I** | Uncrewed test flight | A small GPT pretrained from scratch on general text plus NASA text. A learning artifact: every line explained, with training curves published. |
| **Artemis II** | Crewed flyby | An open-weight 4B to 9B model with continued pretraining on the NASA corpus, then SFT, DPO, and GRPO. Evaluated on FLIGHT-Bench. |
| **Artemis III** | Crewed landing | The full console system: retrieval, tools, go/no-go poll, served with a live demo. |

Tags use the form `artemis-i`, `artemis-ii`, `artemis-iii`. Work between releases lives on
`main`.

## Getting started

Requires Python 3.12+ and, for GPU training, an NVIDIA GPU.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 1. Install PyTorch for your GPU first. Pick the command for your system at
#    https://pytorch.org/get-started/locally/
#    (Older Pascal cards such as the GTX 10-series need the CUDA 12.6 build: cu126.)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2. Install Artemis and the extras for the phase you are working on.
pip install -e ".[pretrain,dev]"
```

## Repository layout

```
pretrain/    Artemis I: model, tokenizer, training loop (written by hand)
data/        NTRS harvester, PDF extraction, cleaning, dedup, license log
posttrain/   Continued pretraining, SFT, DPO, GRPO configs and scripts
consoles/    FIDO, GUIDO, SURGEON, Flight Director
eval/        FLIGHT-Bench items, harness, results
serve/       API and demo UI
docs/        Plan, flight log, writeups, model cards, datasheets
```

## Data and licensing

The corpus is built from NASA public-domain material, mainly the
[NASA Technical Reports Server](https://ntrs.nasa.gov). Not every NTRS document is public
domain: contractor reports and journal articles can carry third-party copyright. Artemis
keeps only NASA-authored work and logs a license decision for every document. Source
attribution: NASA Scientific and Technical Information (STI) Program.

Artemis is an independent learning project. It is not affiliated with or endorsed by NASA.

## License

Code is released under the [MIT License](LICENSE).
