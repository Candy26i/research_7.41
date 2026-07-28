"""
generate_jsonl_api.py — DeepSeek direct API, batch JSONL

Input:  {"example_id": "001", "prompt": [{"role": "user", "content": "..."}]}
Output: {"example_id": "001", "prompt": [...], "response": "..."}

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    python generate_jsonl_api.py --input-file q.jsonl --output-file a.jsonl --model deepseek-v4-flash
"""

import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

_write_lock = threading.Lock()


def complete(client, model, messages, max_tokens):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )
    msg = response.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", "") or ""
    return content if content.strip() else reasoning


def main(input_file, output_file, model, max_tokens, base_url, api_key, workers, retries):
    client = OpenAI(api_key=api_key, base_url=base_url)

    with open(input_file, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f if l.strip()]

    if not output_file:
        output_file = input_file + ".responses.jsonl"

    # resume: skip example_ids already present in the output
    done = set()
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            for l in f:
                try:
                    done.add(json.loads(l).get("example_id"))
                except Exception:
                    pass
        print(f"[RESUME] {len(done)} already done, skipping them", flush=True)
        if len(done) >= len(examples):
            print("[WARN] output already covers every input id. If you changed the prompt "
                  "condition, move/delete the output file first.", flush=True)

    todo = [e for e in examples if e.get("example_id") not in done]
    print(f"[PLAN] {len(todo)} to generate (of {len(examples)} total)", flush=True)

    failures = []
    fout = open(output_file, "a", encoding="utf-8")

    def work(item):
        idx, ex = item
        for attempt in range(1, retries + 1):
            try:
                text = complete(client, model, ex["prompt"], max_tokens)
                if not text.strip():
                    raise RuntimeError("empty response")
                with _write_lock:
                    fout.write(json.dumps({
                        "example_id": ex.get("example_id"),
                        "prompt": ex["prompt"],
                        "response": text.strip(),
                    }, ensure_ascii=False) + "\n")
                    fout.flush()
                print(f"[{idx}/{len(todo)}] ok  id={ex.get('example_id')} len={len(text)}", flush=True)
                return
            except Exception as e:
                print(f"[{idx}/{len(todo)}] attempt {attempt} failed id={ex.get('example_id')}: {e}", flush=True)
        failures.append(ex.get("example_id"))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, enumerate(todo, 1)))

    fout.close()
    print(f"\nWrote -> {output_file}")
    if failures:
        print(f"[WARN] {len(failures)} failed: {failures[:20]}")
        print("Re-run the same command to retry only the failures.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-file",  required=True)
    p.add_argument("--output-file", default="")
    p.add_argument("--model",       default="deepseek-v4-flash")
    p.add_argument("--max-tokens",  type=int, default=8192)
    p.add_argument("--base-url",    default="https://api.deepseek.com")
    p.add_argument("--api-key",     default="")
    p.add_argument("--workers",     type=int, default=4)
    p.add_argument("--retries",     type=int, default=3)
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        sys.exit("Error: provide --api-key or set DEEPSEEK_API_KEY")

    main(args.input_file, args.output_file, args.model, args.max_tokens,
         args.base_url, api_key, args.workers, args.retries)
