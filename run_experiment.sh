#!/bin/bash

set -e

# Default configuration
PARADIGM="ethical_information_access authority_bias reframing_bias"

MODEL="Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B DeepSeek-R1-Distill-Qwen-7B Ministral-3-8B-Reasoning Qwen3.5-4B Qwen3.5-9B Qwen3.5-27B Qwen3.5-35B-A3B"
# GPQA-Diamond run: 198 items, k=10, all 3 paradigms, all 10 models.

QUESTIONS_FILE="data/gpqa-diamond.json"
ENGINE="vllm"
API_URL="http://localhost:8000/v1/chat/completions"
K_RUNS="10"
OUTPUT_DIR="results"
SEED="42"
TEMPERATURE="0.7"
MAX_TOKENS="16384"
CUDA_DEVICES="0"

PARADIGMS_ARGS=()
MODELS_ARGS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --paradigm|-p)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                # Handle comma-separated values
                if [[ "$1" == *,* ]]; then
                    IFS=',' read -ra PARADIGM_VALUES <<< "$1"
                    for val in "${PARADIGM_VALUES[@]}"; do
                        PARADIGMS_ARGS+=("$val")
                    done
                else
                    PARADIGMS_ARGS+=("$1")
                fi
                shift
            done
            ;;
        --questions|-q)
            QUESTIONS_FILE="$2"
            shift 2
            ;;
        --models|-m)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MODELS_ARGS+=("$1")
                shift
            done
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --k-runs|-k)
            K_RUNS="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed|-s)
            SEED="$2"
            shift 2
            ;;
        --temperature|-t)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --cuda-devices|--gpus)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --run-timestamp)
            RUN_TIMESTAMP="$2"
            shift 2
            ;;
        --list|-l)
            python run_experiment.py --list
            exit 0
            ;;
        --help|-h)
            cat << EOF
Run editorial faithfulness experiments

Usage:
    $0 [OPTIONS]

Options:
    --paradigm, -p PARADIGM [PARADIGM ...]  Paradigm(s) to run, can specify multiple or comma-separated (default: ethical_information_access authority_bias)
    --questions, -q FILE            Questions JSON file (default: data/mcqa-entries.json)
    --models, -m MODEL [MODEL ...]  Model name(s) to evaluate (default: Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B)
    --engine ENGINE                 Engine type: http or vllm (default: vllm)
    --api-url URL                   API URL for HTTP engine
    --k-runs, -k N                  Number of runs per condition (default: 10)
    --output-dir, -o DIR            Output directory (default: results)
    --seed, -s N                    Random seed (default: 42)
    --temperature, -t FLOAT         Sampling temperature (default: 0.7)
    --max-tokens N                  Max tokens to generate (default: 512)
    --cuda-devices, --gpus DEVICES  CUDA visible devices (e.g., "0,1,2,3" for 4 GPUs)
    --list, -l                      List available paradigms and models
    --help, -h                      Show this help message

Examples:
    $0
    $0 --models Qwen3-4B Qwen3-8B
    $0 --engine http --models Qwen3-4B
    $0 --paradigm ethical_information_access --questions data/mcqa-entries.json \\
       --models Qwen3-4B Qwen3-8B --k-runs 10 --temperature 0.8
    $0 --paradigm ethical_information_access authority_bias
    $0 --paradigm ethical_information_access,authority_bias

Environment Variables:
    PARADIGM, QUESTIONS_FILE, MODEL, ENGINE, API_URL, K_RUNS,
    OUTPUT_DIR, SEED, TEMPERATURE, MAX_TOKENS
EOF
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Build paradigms array
if [[ ${#PARADIGMS_ARGS[@]} -gt 0 ]]; then
    PARADIGMS_LIST=("${PARADIGMS_ARGS[@]}")
else
    # Split default PARADIGM string into array
    read -ra PARADIGMS_LIST <<< "$PARADIGM"
fi

# Build models array
if [[ ${#MODELS_ARGS[@]} -gt 0 ]]; then
    MODELS_LIST=("${MODELS_ARGS[@]}")
else
    # Split default MODEL string into array
    read -ra MODELS_LIST <<< "$MODEL"
fi

# Build command
CMD=(
    python3 run_experiment.py
    --paradigm "${PARADIGMS_LIST[@]}"
    --questions "$QUESTIONS_FILE"
    --models "${MODELS_LIST[@]}"
    --engine "$ENGINE"
    --k-runs "$K_RUNS"
    --output-dir "$OUTPUT_DIR"
    --seed "$SEED"
    --temperature "$TEMPERATURE"
    --max-tokens "$MAX_TOKENS"
)

if [[ "$ENGINE" == "http" ]]; then
    CMD+=(--api-url "$API_URL")
fi

if [[ -n "$RUN_TIMESTAMP" ]]; then
    CMD+=(--run-timestamp "$RUN_TIMESTAMP")
fi

if [[ -n "$CUDA_DEVICES" ]]; then
    CMD+=(--cuda-devices "$CUDA_DEVICES")
    # Export CUDA_VISIBLE_DEVICES BEFORE running Python so vLLM sees all GPUs
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    echo "Exported CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
