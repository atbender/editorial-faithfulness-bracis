"""
Experiment runner for editorial faithfulness paradigms.

Usage:
    # Single model (uses HTTP server)
    python run_experiment.py --paradigm ethical_information_access --questions data/mcqa-entries.json
    
    # Multi-model with programmatic vLLM (spins up/down engines automatically)
    python run_experiment.py --paradigm ethical_information_access --models Qwen3-4B Qwen3-8B --engine vllm
"""

import argparse
import json
import os
import datetime
import re
import random
import time
from dataclasses import asdict
from typing import Optional, Any, List, Dict

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    class DummyTqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.unit = unit
            self.n = 0
        
        def __enter__(self):
            if self.desc:
                print(f"{self.desc}: ", end="", flush=True)
            return self
        
        def __exit__(self, *args):
            print()  # Newline after progress
        
        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"\r{self.desc}: {self.n}/{self.total}", end="", flush=True)
            else:
                print(f"\r{self.desc}: {self.n}", end="", flush=True)
        
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
    
    def tqdm(iterable=None, total=None, desc=None, unit=None):
        return DummyTqdm(iterable=iterable, total=total, desc=desc, unit=unit)

from paradigms import (
    Paradigm,
    MCQAProblem,
    ExperimentalCondition,
    TrialResult,
    EthicalInformationAccessParadigm,
    AuthorityBiasParadigm,
    ReframingBiasParadigm,
    SYSTEM_PROMPT,
)
from engine import (
    InferenceEngine,
    VLLMEngine,
    HTTPEngine,
    ModelConfig,
    get_model_config,
    engine_context,
    MODEL_CONFIGS,
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODELS = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]  # Default multi-model set
DEFAULT_ENGINE = "vllm"  # Default to vLLM for multi-model support
K_RUNS = 10  # Number of runs per condition per problem


# ============================================================================
# Answer Extraction
# ============================================================================

def extract_answer(text: Optional[str]) -> tuple[Optional[str], bool]:
    """
    Extract answer option from model output using \\boxed{X} format.

    Ported from deepseek-enem-qa formatter (src/enem_ollama.py). Reasoning models
    (R1/Qwen3) reliably emit \\boxed{} for MCQA — it's their native answer format.

    Logic:
    1. Split on </think> and consider only the post-reasoning segment, so any
       \\boxed{} the model wrote while deliberating is ignored.
    2. Find every \\boxed{...} match in that segment and take the LAST one
       (handles models that emit a candidate, reconsider, and emit a final).
    3. Pull the first A-D letter out of the captured content (tolerates wraps
       like \\boxed{A}, \\boxed{(A)}, \\boxed{A.}, \\boxed{ A }).

    Returns:
        Tuple of (extracted_answer, protocol_compliant)
        - extracted_answer: "A"-"D" if a valid boxed answer is found, else None
        - protocol_compliant: True iff a usable \\boxed{X} match was found
    """
    if text is None:
        return None, False

    after_reasoning = text.split("</think>")[-1]

    matches = list(re.finditer(r"\\boxed\{([^}]+)\}", after_reasoning))
    if not matches:
        return None, False

    raw = matches[-1].group(1)
    letter_match = re.search(r"[A-D]", raw, re.IGNORECASE)
    if letter_match:
        return letter_match.group(0).upper(), True

    return None, False


def count_tokens_approx(text: str) -> int:
    """Approximate token count (words + punctuation)."""
    if not text:
        return 0
    return len(text.split())


# ============================================================================
# Data Loading
# ============================================================================

def load_problems(filepath: str) -> list[MCQAProblem]:
    """Load MCQA problems from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    problems = []
    for item in data:
        problems.append(MCQAProblem(
            id=str(item["id"]),
            question=item["question"],
            correct_answer=item["correct_answer"],
            correct_option_text=item.get("correct_option_text", ""),
            difficulty=item.get("difficulty", "medium")
        ))
    
    return problems


# ============================================================================
# Experiment Runner
# ============================================================================

def _build_trial_result(
    paradigm: Paradigm,
    problem: MCQAProblem,
    condition: ExperimentalCondition,
    output: Optional[str],
    control_answer: Optional[str] = None,
) -> TrialResult:
    """Post-process a completed generation into a TrialResult (no inference)."""
    extracted_answer, protocol_compliant = extract_answer(output)
    answer = extracted_answer if extracted_answer else "UNKNOWN"
    cot_length = count_tokens_approx(output) if output else 0

    matches_target = False
    is_correct = False
    answer_flipped = None

    if protocol_compliant and extracted_answer:
        matches_target = (
            extracted_answer == condition.target_option
            if condition.target_option else False
        )
        is_correct = (extracted_answer == problem.correct_answer)
        if control_answer is not None and control_answer != "UNKNOWN" and not condition.is_control:
            answer_flipped = (extracted_answer != control_answer)

    attribution = paradigm.detect_attribution(output or "", condition)
    hint_mentioned = attribution in ["explicit", "implicit"]

    result = TrialResult(
        problem_id=problem.id,
        difficulty=problem.difficulty,
        condition_name=condition.name,
        target_option=condition.target_option,
        raw_output=output or "",
        extracted_answer=answer,
        cot_length=cot_length,
        matches_target=matches_target,
        manipulation_mentioned=attribution,
        control_answer=control_answer,
        answer_flipped=answer_flipped,
        extra_metrics={
            "correct_answer": problem.correct_answer,
            "is_correct": is_correct,
            "hint_mentioned": hint_mentioned,
            "protocol_compliant": protocol_compliant,
            "extracted_answer_raw": extracted_answer,
        }
    )

    custom_metrics = paradigm.compute_trial_metrics(result, condition, problem)
    result.extra_metrics.update(custom_metrics)

    return result


def run_condition_batch(
    paradigm: Paradigm,
    problems: list[MCQAProblem],
    condition_name: str,
    engine: InferenceEngine,
    k_runs: int,
    temperature: float,
    max_tokens: int,
    control_answers: Optional[Dict[tuple, Optional[str]]] = None,
) -> Dict[tuple, TrialResult]:
    """
    Run every (problem, k) for ONE condition as a single batched vLLM call.

    All prompts for the condition are flattened into one list and handed to
    vLLM's generate_batch, so the continuous batcher can interleave sequences
    across available KV cache. This is where almost all of the throughput
    advantage over the old serial loop comes from.
    """
    prompts: List[str] = []
    index: List[tuple] = []
    for problem in problems:
        condition = paradigm.get_conditions(problem)[condition_name]
        for k in range(k_runs):
            prompts.append(condition.build_prompt(problem))
            index.append((problem, k, condition))

    print(f"  Submitting {len(prompts)} prompts to vLLM as a single batch...")
    t0 = time.monotonic()
    outputs = engine.generate_batch(
        prompts,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=SYSTEM_PROMPT,
    )
    elapsed = time.monotonic() - t0
    throughput = len(prompts) / elapsed if elapsed > 0 else float("inf")
    print(f"  Batch done in {elapsed:.1f}s ({throughput:.2f} trials/s)")

    results: Dict[tuple, TrialResult] = {}
    for (problem, k, condition), output in zip(index, outputs):
        control_answer = None
        if control_answers is not None:
            control_answer = control_answers.get((problem.id, k))
        result = _build_trial_result(paradigm, problem, condition, output, control_answer)
        results[(problem.id, k)] = result

    return results


def run_experiment_single_model(
    paradigm: Paradigm,
    problems: list[MCQAProblem],
    engine: InferenceEngine,
    k_runs: int = K_RUNS,
    output_dir: str = "results",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    run_timestamp: Optional[str] = None,
) -> tuple[list[dict[str, TrialResult]], str, dict[str, Any]]:
    """
    Run experiment for a single model.
    
    Returns:
        Tuple of (results, run_dir, stats)
    """
    model_name = engine.model_name
    
    # Setup output directory structure: run-timestamp/paradigm-name/model-name/
    if run_timestamp is None:
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Clean paradigm name (replace underscores with hyphens)
    paradigm_dir = paradigm.config.name.replace("_", "-")
    
    run_dir = os.path.join(output_dir, f"run-{run_timestamp}", paradigm_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)

    # Output files
    results_file = os.path.join(run_dir, "trials.jsonl")  # Raw trial data
    stats_file = os.path.join(run_dir, "statistics.json")  # Computed statistics
    report_file = os.path.join(run_dir, "report.txt")  # Human-readable report

    # Resume support: skip this (model, paradigm) if the full output trio exists.
    # report.txt is only written after statistics.json is written after trials.jsonl,
    # so all three being present means the cell completed cleanly.
    if (os.path.exists(results_file)
            and os.path.exists(stats_file)
            and os.path.exists(report_file)):
        print(f"\n[resume] {model_name} / {paradigm.config.name}: already complete, skipping.")
        return [], run_dir, {}

    all_results: list[dict[str, TrialResult]] = []
    
    # Experiment metadata
    experiment_meta = {
        "paradigm": paradigm.config.name,
        "paradigm_description": paradigm.config.description,
        "model": model_name,
        "conditions": paradigm.config.condition_names,
        "control_condition": paradigm.config.control_condition,
        "k_runs": k_runs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": run_timestamp,
        "num_problems": len(problems),
    }
    
    print(f"\n{'='*60}")
    print(f"Running Paradigm: {paradigm.config.name}")
    print(f"Model: {model_name}")
    print(f"Problems: {len(problems)}")
    print(f"Conditions: {paradigm.config.condition_names}")
    print(f"Runs per condition: {k_runs}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")
    
    control_cond_name = paradigm.get_control_condition_name()
    
    # Store control answers: (problem_id, run_k) -> control_answer
    control_answers: Dict[tuple[str, int], Optional[str]] = {}
    
    # Store all results: (problem_id, run_k) -> dict[condition_name -> TrialResult]
    results_by_problem_k: Dict[tuple[str, int], Dict[str, TrialResult]] = {}
    
    with open(results_file, 'w') as f:
        # Step 1: Run CONTROL condition as a single batch (all problems × k_runs)
        print(f"\n--- Running CONTROL condition ---")
        control_results = run_condition_batch(
            paradigm, problems, control_cond_name, engine,
            k_runs=k_runs, temperature=temperature, max_tokens=max_tokens,
        )
        for (pid, k), result in control_results.items():
            if result.extra_metrics.get('protocol_compliant', False):
                control_answers[(pid, k)] = result.extra_metrics.get('extracted_answer_raw')
            else:
                control_answers[(pid, k)] = None
            results_by_problem_k[(pid, k)] = {control_cond_name: result}

        # Step 2: Run each manipulated condition as a single batch
        other_conditions = [name for name in paradigm.config.condition_names if name != control_cond_name]
        for cond_name in other_conditions:
            print(f"\n--- Running {cond_name.upper()} condition ---")
            cond_results = run_condition_batch(
                paradigm, problems, cond_name, engine,
                k_runs=k_runs, temperature=temperature, max_tokens=max_tokens,
                control_answers=control_answers,
            )
            for key, result in cond_results.items():
                results_by_problem_k[key][cond_name] = result
        
        # Step 3: Write all results to JSONL and build all_results list
        print(f"\n--- Writing results ---")
        for problem in problems:
            for k in range(k_runs):
                run_results = results_by_problem_k[(problem.id, k)]
                
                # Log progress for this run
                log_parts = [f"Problem {problem.id}, k={k+1}:"]
                for cond_name in paradigm.config.condition_names:
                    if cond_name in run_results:
                        result = run_results[cond_name]
                        protocol_status = "✓" if result.extra_metrics.get('protocol_compliant', False) else "✗"
                        flip_str = ""
                        if result.answer_flipped is not None:
                            flip_str = " (flipped)" if result.answer_flipped else " (same)"
                        log_parts.append(f"{cond_name}={result.extracted_answer}{protocol_status}{flip_str}")
                print("  " + ", ".join(log_parts))
                
                # Store with metadata for per-item analysis
                enriched_run_results = {
                    "problem_id": problem.id,
                    "run_k": k,
                    "results": run_results
                }
                all_results.append(enriched_run_results)
                
                # Write trial to JSONL
                trial_entry = {
                    "problem_id": problem.id,
                    "difficulty": problem.difficulty,
                    "run_k": k,
                    "model": model_name,
                    "correct_answer": problem.correct_answer,
                    "conditions": {
                        name: asdict(result) 
                        for name, result in run_results.items()
                    }
                }
                f.write(json.dumps(trial_entry) + "\n")
                f.flush()
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = paradigm.compute_statistics(all_results)
    stats["metadata"] = experiment_meta
    
    # Save statistics as JSON
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Generate human-readable report
    print("Generating report...")
    report = paradigm.generate_report(stats, report_file)
    
    print(f"\nOutput files:")
    print(f"  Trials: {results_file}")
    print(f"  Statistics: {stats_file}")
    print(f"  Report: {report_file}")
    
    return all_results, run_dir, stats


def run_experiment_multi_model(
    paradigm: Paradigm,
    problems: list[MCQAProblem],
    model_configs: List[ModelConfig],
    k_runs: int = K_RUNS,
    output_dir: str = "results",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    run_timestamp: Optional[str] = None,
) -> dict[str, list[dict[str, TrialResult]]]:
    """
    Run experiment across multiple models.
    
    Spins up vLLM engine for each model, runs experiment, then shuts down.
    """
    if run_timestamp is None:
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Clean paradigm name (replace underscores with hyphens)
    paradigm_dir = paradigm.config.name.replace("_", "-")
    
    # Create run-timestamp/paradigm-name/ directory
    batch_dir = os.path.join(output_dir, f"run-{run_timestamp}", paradigm_dir)
    os.makedirs(batch_dir, exist_ok=True)
    
    all_model_results = {}
    all_model_stats = {}
    
    # Batch metadata
    batch_meta = {
        "paradigm": paradigm.config.name,
        "paradigm_description": paradigm.config.description,
        "models": [c.name for c in model_configs],
        "conditions": paradigm.config.condition_names,
        "k_runs": k_runs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": run_timestamp,
        "num_problems": len(problems),
    }
    
    print(f"\n{'#'*60}")
    print(f"BATCH EXPERIMENT: {paradigm.config.name}")
    print(f"Models: {[c.name for c in model_configs]}")
    print(f"Output: {batch_dir}")
    print(f"{'#'*60}\n")
    
    for i, config in enumerate(model_configs):
        print(f"\n{'='*60}")
        print(f"MODEL {i+1}/{len(model_configs)}: {config.name}")
        print(f"{'='*60}")
        
        with engine_context(config) as engine:
            results, run_dir, stats = run_experiment_single_model(
                paradigm=paradigm,
                problems=problems,
                engine=engine,
                k_runs=k_runs,
                output_dir=output_dir,  # Pass base output_dir, function will create run-timestamp/paradigm/
                temperature=temperature,
                max_tokens=max_tokens,
                run_timestamp=run_timestamp,
            )
            all_model_results[config.name] = results
            all_model_stats[config.name] = stats
    
    # Generate comparative summary (JSON + text)
    _generate_batch_summary(paradigm, all_model_results, all_model_stats, batch_meta, batch_dir)
    
    return all_model_results


def _generate_batch_summary(
    paradigm: Paradigm,
    all_results: dict[str, list[dict[str, TrialResult]]],
    all_stats: dict[str, dict[str, Any]],
    batch_meta: dict[str, Any],
    output_dir: str
):
    """Generate comparative summary across models (JSON + text)."""
    summary_json_file = os.path.join(output_dir, "batch_summary.json")
    summary_txt_file = os.path.join(output_dir, "batch_summary.txt")
    
    # Build JSON summary structure
    summary_data = {
        "metadata": batch_meta,
        "models": {},
        "comparisons": {},
    }
    
    # Extract key metrics per model
    for model_name, stats in all_stats.items():
        model_summary = {
            "total_runs": stats.get("total_runs", 0),
            "conditions": {},
        }
        
        for cond_name in paradigm.config.condition_names:
            cond_stats = stats.get("conditions", {}).get(cond_name, {})
            model_summary["conditions"][cond_name] = {
                "correct": cond_stats.get("correct", 0),
                "correct_rate": cond_stats.get("correct_rate", 0),
                "compliance_rate": cond_stats.get("compliance_rate", 0),
                "flip_rate": cond_stats.get("flip_rate", 0),
                "avg_cot_length": cond_stats.get("avg_cot_length", 0),
                "attribution": cond_stats.get("attribution", {}),
            }
        
        # Add paradigm-specific key metrics
        if "differential_compliance" in stats:
            model_summary["differential_compliance"] = stats["differential_compliance"]
        if "differential_explicit_attribution" in stats:
            model_summary["differential_explicit_attribution"] = stats["differential_explicit_attribution"]
        
        summary_data["models"][model_name] = model_summary
    
    # Build comparison tables for graphing
    for cond_name in paradigm.config.condition_names:
        summary_data["comparisons"][cond_name] = {
            model_name: {
                "correct_rate": all_stats[model_name].get("conditions", {}).get(cond_name, {}).get("correct_rate", 0),
                "compliance_rate": all_stats[model_name].get("conditions", {}).get(cond_name, {}).get("compliance_rate", 0),
                "flip_rate": all_stats[model_name].get("conditions", {}).get(cond_name, {}).get("flip_rate", 0),
            }
            for model_name in all_stats
        }
    
    # Save JSON
    with open(summary_json_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Build text summary
    lines = []
    lines.append("=" * 60)
    lines.append(f"BATCH SUMMARY: {paradigm.config.name}")
    lines.append("=" * 60)
    lines.append(f"Date: {datetime.datetime.now()}")
    lines.append(f"Models: {list(all_stats.keys())}")
    lines.append("")
    
    # Comparison table
    lines.append("--- PER-CONDITION COMPARISON ---")
    lines.append("")
    
    for cond_name in paradigm.config.condition_names:
        lines.append(f"Condition: {cond_name.upper()}")
        lines.append("-" * 40)
        lines.append(f"{'Model':<30} {'Accuracy':>10} {'Compliance':>12}")
        
        for model_name, stats in all_stats.items():
            cond_stats = stats.get("conditions", {}).get(cond_name, {})
            acc = cond_stats.get("correct_rate", 0) * 100
            comp = cond_stats.get("compliance_rate", 0) * 100
            lines.append(f"{model_name:<30} {acc:>9.1f}% {comp:>11.1f}%")
        lines.append("")
    
    # Key metrics comparison
    lines.append("--- KEY PARADIGM METRICS ---")
    lines.append("")
    
    first_model_stats = list(all_stats.values())[0] if all_stats else {}
    if "differential_compliance" in first_model_stats:
        lines.append(f"{'Model':<30} {'Diff. Compliance':>18}")
        for model_name, stats in all_stats.items():
            diff = stats.get("differential_compliance", 0) * 100
            lines.append(f"{model_name:<30} {diff:>+17.1f}%")
    
    summary_content = "\n".join(lines)
    
    with open(summary_txt_file, 'w') as f:
        f.write(summary_content)
    
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    print(summary_content)
    print(f"\nOutput files:")
    print(f"  JSON: {summary_json_file}")
    print(f"  Text: {summary_txt_file}")


# ============================================================================
# Main Entry Point
# ============================================================================

PARADIGM_REGISTRY = {
    "ethical_information_access": EthicalInformationAccessParadigm,
    "authority_bias": AuthorityBiasParadigm,
    "reframing_bias": ReframingBiasParadigm,
}


def get_paradigm(name: str) -> Paradigm:
    """Get paradigm instance by name."""
    if name not in PARADIGM_REGISTRY:
        available = list(PARADIGM_REGISTRY.keys())
        raise ValueError(f"Unknown paradigm: {name}. Available: {available}")
    
    factory = PARADIGM_REGISTRY[name]
    return factory() if callable(factory) else factory


def list_paradigms():
    """List available paradigms."""
    print("Available paradigms:")
    for name in PARADIGM_REGISTRY:
        paradigm = get_paradigm(name)
        print(f"  - {name}: {paradigm.config.description}")


def list_models():
    """List available pre-configured models."""
    print("Available pre-configured models:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  - {name}: {config.model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run editorial faithfulness experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-model with vLLM (default - spins up server for each model)
  python run_experiment.py -p ethical_information_access -q data/mcqa-entries.json

  # Multiple paradigms sequentially
  python run_experiment.py -p ethical_information_access authority_bias -q data/mcqa-entries.json

  # Multiple paradigms (comma-separated)
  python run_experiment.py -p ethical_information_access,authority_bias -q data/mcqa-entries.json

  # Single model via HTTP server
  python run_experiment.py -p ethical_information_access --engine http --models Qwen3-4B

  # Custom model set
  python run_experiment.py -p ethical_information_access --models Qwen3-4B Qwen3-8B

  # List available paradigms and models
  python run_experiment.py --list
        """
    )
    
    parser.add_argument(
        "--paradigm", "-p",
        type=str,
        nargs="+",
        default=["ethical_information_access"],
        help="Paradigm(s) to run (can specify multiple, or comma-separated string)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available paradigms and models"
    )
    parser.add_argument(
        "--questions", "-q",
        type=str,
        default="data/mcqa-entries.json",
        help="Questions file to use"
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model name(s) to evaluate (default: Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["http", "vllm"],
        default=DEFAULT_ENGINE,
        help="Engine type: 'http' for external server, 'vllm' for programmatic (default: vllm)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help="API URL for HTTP engine"
    )
    parser.add_argument(
        "--k-runs", "-k",
        type=int,
        default=K_RUNS,
        help="Number of runs per condition"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help="CUDA visible devices (e.g., '0,1,2,3' for 4 GPUs). Sets CUDA_VISIBLE_DEVICES for vLLM."
    )
    parser.add_argument(
        "--run-timestamp",
        type=str,
        default=None,
        help="Reuse this run-timestamp (e.g. '20260413-210000') instead of generating a new one. "
             "Enables resuming a previous run: (model, paradigm) cells with a complete "
             "trials.jsonl+statistics.json+report.txt trio are skipped."
    )

    args = parser.parse_args()
    
    if args.list:
        list_paradigms()
        print()
        list_models()
        return
    
    # Parse paradigms: handle both list and comma-separated string
    paradigms = []
    for p in args.paradigm:
        # Split by comma if comma-separated string
        if "," in p:
            paradigms.extend([p.strip() for p in p.split(",")])
        else:
            paradigms.append(p)
    
    # Remove duplicates while preserving order
    seen = set()
    paradigms = [p for p in paradigms if not (p in seen or seen.add(p))]
    
    if not paradigms:
        print("Error: No valid paradigms specified.")
        return
    
    # Set seed
    random.seed(args.seed)
    
    # Create consistent timestamp for this batch run (all models and paradigms share same timestamp),
    # or reuse the one supplied via --run-timestamp to resume a previous run.
    if args.run_timestamp:
        run_timestamp = args.run_timestamp
        print(f"[resume] Reusing run timestamp: {run_timestamp}")
    else:
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Load problems once (shared across all models and paradigms)
    problems = load_problems(args.questions)
    
    # Get model configs
    model_configs = [get_model_config(name) for name in args.models]
    
    # Apply CUDA visible devices to all model configs if specified
    if args.cuda_devices:
        for config in model_configs:
            config.cuda_visible_devices = args.cuda_devices
        print(f"Using CUDA devices: {args.cuda_devices}")
        
        # Also count GPUs for tensor_parallel validation
        num_gpus = len(args.cuda_devices.split(","))
        print(f"Number of GPUs available: {num_gpus}")
    
    # Iterate over models first (to minimize model load/unload)
    for model_idx, model_config in enumerate(model_configs):
        print(f"\n{'#'*60}")
        print(f"MODEL {model_idx+1}/{len(model_configs)}: {model_config.name}")
        print(f"{'#'*60}\n")
        
        # Run all paradigms for this model
        if args.engine == "http":
            # HTTP engine: create once and reuse for all paradigms
            if model_idx > 0:
                print("Warning: HTTP engine only supports one model. Skipping additional models.")
                break
            
            engine = HTTPEngine(args.api_url, model_config.name)
            
            # Run all paradigms with this engine
            for paradigm_idx, paradigm_name in enumerate(paradigms):
                print(f"\n{'='*60}")
                print(f"PARADIGM {paradigm_idx+1}/{len(paradigms)}: {paradigm_name}")
                print(f"{'='*60}\n")
                
                try:
                    paradigm = get_paradigm(paradigm_name)
                except ValueError as e:
                    print(f"Error: {e}")
                    print(f"Skipping paradigm: {paradigm_name}")
                    continue
                
                results, run_dir, stats = run_experiment_single_model(
                    paradigm=paradigm,
                    problems=problems,
                    engine=engine,
                    k_runs=args.k_runs,
                    output_dir=args.output_dir,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    run_timestamp=run_timestamp,
                )
                print(f"\nCompleted paradigm: {paradigm_name}")
            
            # HTTP engine doesn't need explicit shutdown
        else:
            # vLLM engine: load model once, run all paradigms, then unload
            with engine_context(model_config) as engine:
                # Run all paradigms with this model
                for paradigm_idx, paradigm_name in enumerate(paradigms):
                    print(f"\n{'='*60}")
                    print(f"PARADIGM {paradigm_idx+1}/{len(paradigms)}: {paradigm_name}")
                    print(f"{'='*60}\n")
                    
                    try:
                        paradigm = get_paradigm(paradigm_name)
                    except ValueError as e:
                        print(f"Error: {e}")
                        print(f"Skipping paradigm: {paradigm_name}")
                        continue
                    
                    results, run_dir, stats = run_experiment_single_model(
                        paradigm=paradigm,
                        problems=problems,
                        engine=engine,
                        k_runs=args.k_runs,
                        output_dir=args.output_dir,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        run_timestamp=run_timestamp,
                    )
                    print(f"\nCompleted paradigm: {paradigm_name}")
        
        print(f"\nCompleted model: {model_config.name}")
    
    print(f"\n{'#'*60}")
    print(f"ALL MODELS AND PARADIGMS COMPLETED")
    print(f"{'#'*60}")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()

