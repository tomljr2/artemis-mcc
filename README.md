# Artemis

An aerospace-specialist language model, built in the open as a way to learn how LLMs
actually work, from the first line of attention code to a served multi-agent system.

Artemis is named after NASA's Artemis program. It is trained on NASA public-domain
material, and its runtime is modeled on Mission Control: specialist consoles each check
an answer, and a Flight Director runs a **go/no-go poll** before anything is released.
When the consoles disagree, Artemis says so instead of guessing.

> Status: **Pre-launch.** Building Artemis I (Phase 1). No release yet.

## Mission status

| Phase 1 stage | State |
|---|---|
| Stage 1: a GPT from an empty file | Done, except one sanity check (overfit one batch) |
| Stage 2: modern upgrades (RMSNorm, RoPE, batched heads, SwiGLU, no biases) | In work |
| Stage 3: BPE tokenizer | Next |
| Stage 4: real training run, tag `artemis-i` | Not started |

Step-by-step checklist: [pretrain/README.md](pretrain/README.md). Every step, with its
measurements: [docs/FLIGHT_LOG.md](docs/FLIGHT_LOG.md). Full plan: [docs/PLAN.md](docs/PLAN.md).

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

| Console | Real Mission Control role | Artemis job |
|---|---|---|
| **FIDO** | Flight Dynamics Officer: computes the trajectory and plans every engine burn. | Quantitative reasoning: delta-v, Hohmann transfers, Isp, unit conversions. Checks its work by running code. |
| **GUIDO** | Guidance Officer (Apollo era): checks that the spacecraft computer's view of where it is matches ground tracking. GUIDO Steve Bales made the "go" call on the 1202 alarms during the Apollo 11 landing. | Factual grounding: checks the model's answer against retrieved NASA documents, with citations. |
| **SURGEON** | Flight Surgeon: watches crew health and can call a halt on medical grounds. | Safety and sanity: catches false premises, unsafe advice, and overconfidence. |
| **FLIGHT** | Flight Director: final authority. Polls every console "go/no-go" before critical events like launch, burns, and landing. | Routes the query, runs the consoles in parallel, polls them, and calls GO or NO-GO. |

Each console is the same fine-tuned model with its own system prompt and tools, so the
whole system runs on one hosted model.

## Releases

Releases follow the Artemis mission sequence as restructured by NASA in February 2026.
Each one is a git tag and, where it applies, a Hugging Face release with a model card.

| Release | The real mission | What ships |
|---|---|---|
| **Artemis I** | Uncrewed test flight of SLS and Orion around the Moon (Nov–Dec 2022). Proved the hardware before people flew on it. | A small GPT pretrained from scratch. Proves the training stack end to end before anything depends on it. |
| **Artemis II** | First crewed flight: a 10-day lunar flyby, no landing (Apr 2026). | The fine-tuned 4B to 9B model (continued pretraining, SFT, DPO, GRPO). The first model people actually use, measured on FLIGHT-Bench, but without the full console system yet. |
| **Artemis III** | Earth-orbit test (planned 2027): Orion rendezvous and docking with the commercial lunar landers, plus spacesuit tests. Like Apollo 9, it rehearses the landing hardware close to home. | The console system integrated and docked to the model: retrieval, tools, and the go/no-go poll, tested on FLIGHT-Bench before going public. |
| **Artemis IV** | First crewed lunar landing since Apollo 17 (planned 2028). | Launch: the full system served publicly with a live demo, model cards, and the final writeup. |
| **Artemis V+** | Further landings and the start of a permanent Moon base (planned late 2028 onward). | Sustained operations: corpus and benchmark updates, new console capabilities. |

Tags use the form `artemis-i`, `artemis-ii`, and so on. Work between releases lives on
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
