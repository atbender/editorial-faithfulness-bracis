# editorial-faithfulness

Code and data for the paper **"Editorial Faithfulness in Reasoning Models:
Susceptibility to Authority, Framing, and Information Legitimacy Cues"**
(A. T. Bender, G. A. Gomes, U. B. Corrêa, R. M. Araujo — Federal University of
Pelotas, UFPel).

We measure **editorial faithfulness**: the degree to which a reasoning model's
answer stays invariant under non-evidential *editorial cues* — claims of
authority, statements about how information was obtained, and evaluative framing —
that change the presentation of a question without changing its underlying
evidence. Ten open-weight reasoning models are evaluated on multiple-choice
question answering (MCQA) under three controlled paradigms, each crossed with a
help/harm axis, using three variance-aware behavioral metrics.

## Key findings

- **Information access legitimacy is the dominant failure mode.** Labeling an
  incorrect option as "verified internal data" drives pass@1 to **0** for the
  entire Qwen3.5 family, while DeepSeek-R1-7B stays at its control baseline.
- **Training regime, not parameter count, determines faithfulness.** At
  comparable scale (≈8B), legitimate-harm drops range from +1% to −95%.
- **The MoE variant (Qwen3.5-35B-A3B) tracks its dense 27B total-parameter
  sibling**, not its 3B active-compute equivalent — susceptibility scales with
  total capacity.
- **Authority and evaluative framing effects are modest** for most models but
  substantial for Qwen3.5.

## Experimental design

Each prompt follows a fixed canonical template, so the only thing that varies
across conditions is the non-evidential editorial cue:

```
[PREAMBLE EDITORIAL CUE]   <- the only manipulated part
[BASE INSTRUCTION]
[MCQA PROBLEM]             <- question + options, held constant
```

Models must emit their final answer as `\boxed{X}`, `X ∈ {A, B, C, D}` — the
native answer convention for contemporary reasoning models.

**Three paradigms** (`paradigms/`), each with `control`, `*_help`, and `*_harm`
conditions:

| Paper paradigm                | Module                              |
|-------------------------------|-------------------------------------|
| Authority Attribution         | `paradigms/authority_bias.py`       |
| Information Access Legitimacy | `paradigms/ethical_information_access.py` |
| Evaluative Framing            | `paradigms/reframing_bias.py`       |

**Three variance-aware metrics** (`paradigms/metrics.py`), each estimating the
probability that a target event occurs at least once among `k` of `n` samples:

- **pass@k** — task-level correctness under sampling variance.
- **flip@k** — behavioral sensitivity: deviation from the modal control answer.
- **transparency@k** — attributional awareness: whether the model acknowledges
  the editorial cue in its output.

**Ten models** (`engine.py` → `MODEL_CONFIGS`): the Qwen3 dense series
(`Qwen3-4B/8B/14B/32B`), the Qwen3.5 series (`Qwen3.5-4B/9B/27B` dense +
`Qwen3.5-35B-A3B` MoE), `DeepSeek-R1-Distill-Qwen-7B`, and
`Ministral-3-8B-Reasoning-2512`.

## Evaluation stimuli

The evaluation uses a fixed multiple-choice stimulus set drawn from five MMLU
domains: formal logic, philosophy, moral disputes, high-school physics, and
high-school mathematics. The question text and answer options are held constant
across all conditions; only the preamble editorial cue changes. Every model is
evaluated on all three paradigms and their conditions under stochastic decoding,
with k = 10 samples drawn per item.

Stimuli are supplied to the runner as a JSON list, one object per item:

```json
[
  {
    "id": "mcqa_001",
    "question": "<full question text, including the A)–D) options>",
    "correct_answer": "A",
    "correct_option_text": "<text of the correct option>",
    "difficulty": "medium"
  }
]
```

## Prerequisites

1. **Docker and Docker Compose.**
2. **NVIDIA Container Toolkit** for GPU support — install the
   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
   `docker-compose.yml` requests GPU `device_ids: ["0"]`.
3. **A Hugging Face cache.** The container mounts `~/.cache/huggingface` so model
   weights are downloaded once and reused. Set `HF_TOKEN` in your environment if
   any evaluated model is gated.

The container image is `vllm/vllm-openai:v0.19.0-cu130`; all inference runs
through vLLM's continuous batcher inside the container.

## How to run

1. **Start the container** (mounts the repo at `/workspace`):
   ```bash
   docker compose up -d
   ```

2. **Open a shell in the container:**
   ```bash
   docker exec -it editorial-faithfulness /bin/bash
   ```
   You land in `/workspace`.

3. **Run the experiments.** The orchestrator spins each model up in vLLM, runs
   all requested paradigms, then unloads it before loading the next. The
   reproduction defaults (all 10 models, all 3 paradigms, 10-item MCQA set,
   k=10, temperature 0.7, top-p 0.95, max 16,384 generation tokens) are baked
   into `run_experiment.sh`. Point it at your stimulus JSON (see
   *Evaluation stimuli* above for the expected schema):
   ```bash
   ./run_experiment.sh --questions path/to/questions.json
   ```

   Run a subset of models or paradigms directly via the Python entrypoint:
   ```bash
   # One paradigm, two models
   python run_experiment.py \
       --paradigm ethical_information_access \
       --models Qwen3-4B Qwen3-8B \
       --questions path/to/questions.json

   # All three paradigms (comma- or space-separated)
   python run_experiment.py \
       -p ethical_information_access,authority_bias,reframing_bias \
       -m Qwen3-4B

   # List available paradigms and pre-configured models
   python run_experiment.py --list
   ```

   Runs are **resumable**: pass `--run-timestamp <ts>` to reuse an output folder;
   any `(model, paradigm)` cell with a complete
   `trials.jsonl` + `statistics.json` + `report.txt` trio is skipped.

4. **Stop the container** when done:
   ```bash
   docker compose down
   ```

## Output layout

Each run writes a directory tree under the output folder (default `results/`):

```
run-<timestamp>/<paradigm>/<model>/
    trials.jsonl       # one line per (item, sample, condition) with raw output
    statistics.json    # aggregated pass@k / flip@k / transparency@k + metadata
    report.txt         # human-readable per-condition summary
run-<timestamp>/<paradigm>/batch_summary.{json,txt}   # cross-model comparison
```

## Repository structure

```
run_experiment.py      # experiment orchestrator (CLI)
run_experiment.sh      # convenience wrapper with reproduction defaults
engine.py              # vLLM / HTTP inference engines + model registry
paradigms/             # paradigm definitions, base classes, and metrics
    base.py            # prompt template, MCQAProblem, conditions, system prompt
    authority_bias.py
    ethical_information_access.py
    reframing_bias.py
    metrics.py         # pass@k / flip@k / transparency@k estimators
docker-compose.yml     # vLLM container definition
# results/ is created at runtime under the --output-dir folder
```

## Citation

```bibtex
@inproceedings{bender2026editorial,
  title     = {Editorial Faithfulness in Reasoning Models: Susceptibility to
               Authority, Framing, and Information Legitimacy Cues},
  author    = {Bender, Alexandre Thurow and Gomes, Gabriel Almeida and
               Corr{\^e}a, Ulisses Brisolara and Araujo, Ricardo Matsumura},
  booktitle = {Brazilian Conference on Intelligent Systems (BRACIS)},
  year      = {2026}
}
```