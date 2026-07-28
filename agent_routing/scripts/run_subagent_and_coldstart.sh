#!/usr/bin/env bash
set -uo pipefail

BENCH=${1:-gpqa}          # gpqa | medqa | legalbench | mmlu_pro
VARIANT=${2:-science}     # science | medical | legal | mmlu
SIZE=${3:-8B}
BASE_MODEL="Qwen/Qwen3-${SIZE}"

case $BENCH in
  gpqa)       DATA_ID=ds_gpqa;       FLAG=--gpqa_normalized_cache;       CACHE=outputs/data/gpqa_train446.jsonl;        NCOLD=200
              TASK="You are a manager agent solving graduate-level science multiple-choice questions." ;;
  medqa)      DATA_ID=ds_medqa;      FLAG=--medqa_normalized_cache;      CACHE=outputs/data/medqa_us4_normalized.jsonl; NCOLD=300
              TASK="You are a manager agent solving medical multiple-choice questions." ;;
  legalbench) DATA_ID=ds_legalbench; FLAG=--legalbench_normalized_cache; CACHE=outputs/data/legalbench_large5.jsonl;    NCOLD=300
              TASK="You are a manager agent solving legal reasoning multiple-choice questions." ;;
  mmlu_pro)   DATA_ID=ds_mmlu_pro;   FLAG=--mmlu_pro_normalized_cache;   CACHE=outputs/data/mmlu_pro_normalized.jsonl;  NCOLD=300
              TASK="You are a manager agent solving multiple-choice questions across diverse academic subjects." ;;
  *) echo "unknown bench $BENCH"; exit 1 ;;
esac

ID="${DATA_ID}_${SIZE}"
SFT=outputs/sft_data/$DATA_ID
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)

echo "=========================================================="
echo " bench=$BENCH  variant=$VARIANT  base=$BASE_MODEL  id=$ID"
echo "=========================================================="

# ── Step 0: prompts ──
cp "src/subagents/prompts/variants/reasoner_${VARIANT}.py"        src/subagents/prompts/reasoner.py
cp "src/subagents/prompts/variants/runtime_prompts_${VARIANT}.py" src/subagents/prompts/runtime_prompts.py
rm -rf src/subagents/prompts/__pycache__
echo "[PROMPTS] switched to $VARIANT"
grep -o "You are the Reasoner sub-agent[^\"]*" src/subagents/prompts/runtime_prompts.py | head -1

for k in extractor reasoner verifier; do
  [ -f "$SFTntime_raw_sft.jsonl" ] || { echo "[ABORT] missing $SFT/${k}_runtime_raw_sft.jsonl"; exit 1; }
  echo "[DATA] $k $(wc -l < $SFT/${k}_runtime_raw_sft.jsonl) rows"
done

# ── Step 1: three subagents in parallel ──
CUDA_VISIBLE_DEVICES=0 python -X utf8 -m src.pipeline.cli train_subagent \
  --base_model "$BASE_MODEL" --teacher_id "$ID" --agent_kind extractor \
  --sft_train_jsonl "$SFT/extractor_runtime_raw_sft.jsonl" \
  --sft_epochs 10 --sft_lr 2e-4 > logs/sft_${ID}_extractor_$TS.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python -X utf8 -m src.pipeline.cli train_subagent \
  --base_model "$BASE_MODEL" --teacher_id "$ID" --agent_kind reasoner \
  --sft_train_jsonl "$SFT/reasoner_runtime_raw_sft.jsonl" \
  --sft_epochs 10 --sft_lr 2e-4 > logs/sft_${ID}_reasoner_$TS.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=2 python -X utf8 -m src.pipeline.cli train_subagent \
  --base_model "$BASE_MODEL" --teacher_id "$ID" --agent_kind verifier \
  --sft_train_jsonl "$SFT/verifier_runtime_raw_sft.jsonl" \
  --sft_epochs _lr 2e-4 > logs/sft_${ID}_verifier_$TS.log 2>&1 &
PID3=$!

echo "[SFT] running: extractor=$PID1 reasoner=$PID2 verifier=$PID3"
FAILED=0
wait $PID1 || { echo "[ERROR] extractor SFT failed -> logs/sft_${ID}_extractor_$TS.log"; FAILED=1; }
wait $PID2 || { echo "[ERROR] reasoner  SFT failed -> logs/sft_${ID}_reasoner_$TS.log";  FAILED=1; }
wait $PID3 || { echo "[ERROR] verifier  SFT failed -> logs/sft_${ID}_verifier_$TS.log";  FAILED=1; }
[ $FAILED -ne 0 ] && { echo "[ABORT] subagent SFT failed."; exit 1; }
echo "[SFT] all three done"

# ── Step 2: archive the reasoner adapter under its variant name ──
A=outputs/adapters/$ID
[ -f "$A/reasoner_adapter/adapter_model.safetensors" ] || { echo "[ABORT] no reasoner adapter produced"; exit 1; }
rm -rf "$A/reasoner_adapter_${VARIANT}"
cp -r "$A/reasoner_adapter" "$A/reasoner_adapter_${VARIANT}"
echo "[ARCHIVE] $A/reasoner_adapter_${VARIANT}"
ls -d $A/*/

# ── Step 3: manager coldstart ──
[ -n "${DEEPSEEK_API_KEY:-}" ] || { echo "[ABORT] DEEPSEEK_API_KEY not set"; exit 1; }

CUDA_VISIBLE_DEVICES=0 python -X utf8 -m src.pipeline.cli manager_coldstart_sft \
  --base_model "$BASE_MODEL" --teacher_id "$ID" \
  "$FLAG" "$CACHE" \
  --train_size 1200 --coldstart_n_samples "$NCOLD" \
  --teacher_provider deepseek --teacher_model deepseek-v4-flash \
  --exclude_sft_example_ids "$SFT/extractor_runtime_raw_sft.jsonl" \
  --exclude_sft_example_ids "$SFT/reasoner_runtime_raw_sft.jsonl" \
  --exclude_sft_example_ids "$SFT/verifier_runtime_raw_sft.jsonl" \
  --task_description "$TASK" \
  2>&1 | tee logs/coldstart_${ID}_$TS.log
[ ${PIPESTATUS[0]} -ne 0 ] && { echo "[ABORT] coldstart failed."; exit 1; }

echo "[DONE] $(ls -la outputs/manager/$ID/sft_coldstart/ 2>/dev/null | head -3)"
grep -E "SPLIT|EXCLUDE_SFT|COLDSTART\] wrote" logs/coldstart_${ID}_$TS.log
