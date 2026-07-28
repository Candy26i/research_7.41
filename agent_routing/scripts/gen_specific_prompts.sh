#!/usr/bin/env bash
set -euo pipefail
BENCH=$1          # medqa | legalbench | mmlu_pro | gpqa
VARIANT=$2        # medical | legal | mmlu | science
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}

case $BENCH in
  medqa)      ID=ds_medqa;      FLAG=--medqa_normalized_cache;      CACHE=outputs/data/medqa_us4_normalized.jsonl;  POOL=12723 ;;
  legalbench) ID=ds_legalbench; FLAG=--legalbench_normalized_cache; CACHE=outputs/data/legalbench_large5.jsonl;     POOL=2834 ;;
  mmlu_pro)   ID=ds_mmlu_pro;   FLAG=--mmlu_pro_normalized_cache;   CACHE=outputs/data/mmlu_pro_normalized.jsonl;   POOL=12032 ;;
  gpqa)       ID=ds_gpqa;       FLAG=--gpqa_normalized_cache;       CACHE=outputs/data/gpqa_train446.jsonl;         POOL=446 ;;
  *) echo "unknown bench $BENCH"; exit 1 ;;
esac

echo "=== switching prompts to $VARIANT"
cp "src/subagents/prompts/variants/reasoner_${VARIANT}.py"        src/subagents/prompts/reasoner.py
cp "src/subagents/prompts/variants/runtime_prompts_${VARIANT}.py" src/subagents/prompts/runtime_prompts.py
rm -rf src/subagents/prompts/__pycache__

ARGS=(
  --base_model "$BASE_MODEL"
  --teacher_id "$ID"
  "$FLAG" "$CACHE"
  --train_size "$POOL" --dev_size 0 --test_size 0
  --seed 42
  --agent_kind reasoner
  --n_samples "$POOL"
  --deepseek_prompt_jsonl "/tmp/${ID}_all.jsonl"
)
echo "+ python -m src.pipeline.cli export_deepseek_jsonl ${ARGS[*]}"
python -m src.pipeline.cli export_deepseek_jsonl "${ARGS[@]}"

echo "=== filtering to extractor ids"
python3 - "$ID" << 'PYEOF'
import json, sys
ID = sys.argv[1]
ext = {json.loads(l)['example_id'] for l in open(f'outputs/sft_data/{ID}/extractor_runtime_raw_sft.jsonl')}
kept, seen = 0, set()
with open(f'outputs/sft_data/{ID}/reasoner_deepseek_prompts.jsonl', 'w') as fo:
    for l in open(f'/tmp/{ID}_all.jsonl'):
        eid = json.loads(l)['example_id']
        if eid in ext and eid not in seen:
            fo.write(l); seen.add(eid); kept += 1
print(f'extractor ids={len(ext)}  kept={kept}')
if kept != len(ext):
    print('missing ids (first 10):', sorted(ext - seen)[:10])
    raise SystemExit(f'!! MISMATCH: kept {kept} of {len(ext)}')
PYEOF

echo "=== system prompt in use"
head -1 "outputs/sft_data/$ID/reasoner_deepseek_prompts.jsonl" | python3 -c "
import json,sys
s=[m['content'] for m in json.load(sys.stdin)['prompt'] if m.get('role')=='system'][0]
print(s[:300])"
