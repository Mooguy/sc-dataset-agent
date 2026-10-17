"""Day 0: compare Token Factory models on tool calling.

Usage:
    uv run scripts/test_models.py --list            # list model IDs available to your key
    uv run scripts/test_models.py                   # test models in CANDIDATE_MODELS
    uv run scripts/test_models.py model_a model_b   # test specific models

Each model gets the same cases. A case passes when the model:
  - calls the tool when it should (and does not when it shouldn't),
  - returns arguments that parse as JSON,
  - extracts the expected filter values.
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.environ["NEBIUS_BASE_URL"],
    api_key=os.environ["NEBIUS_API_KEY"],
    timeout=60,
)

SYSTEM = (
    "You help researchers find public single-cell RNA-seq datasets. "
    "Use the search_geo tool for dataset requests. "
    "If the request is not about finding datasets, answer briefly without calling tools."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_geo",
            "description": "Search NCBI GEO for single-cell RNA-seq datasets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text search query"},
                    "organism": {"type": "string", "enum": ["Homo sapiens", "Mus musculus"]},
                    "tissue": {"type": "string"},
                    "cell_type": {"type": "string"},
                    "disease": {"type": "string"},
                    "assay": {"type": "string", "description": "e.g. 10x 3' v3, Smart-seq2, snRNA-seq"},
                    "min_donors": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    }
]

# (request, expected args subset or None if no tool call expected)
CASES = [
    (
        "Human lung fibroblasts, 10x, IPF vs healthy, at least 5 donors.",
        {"organism": "Homo sapiens", "min_donors": 5},
    ),
    ("Mouse cortex single-nucleus data from aged animals", {"organism": "Mus musculus"}),
    ("PBMCs from lupus patients, any human dataset with raw counts", {"organism": "Homo sapiens"}),
    ("What's the capital of France?", None),
]


def run_case(model: str, request: str, expected: dict | None) -> dict:
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}],
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            max_tokens=512,
        )
    except Exception as e:  # noqa: BLE001 - smoke test, report any failure
        return {"ok": False, "why": f"error: {e}", "latency": time.perf_counter() - t0, "tokens": 0}

    latency = time.perf_counter() - t0
    tokens = resp.usage.total_tokens if resp.usage else 0
    calls = resp.choices[0].message.tool_calls or []

    if expected is None:
        ok = not calls
        return {"ok": ok, "why": "" if ok else "called tool on off-topic", "latency": latency,
                "tokens": tokens}

    if not calls:
        return {"ok": False, "why": "no tool call", "latency": latency, "tokens": tokens}

    try:
        args = json.loads(calls[0].function.arguments)
    except json.JSONDecodeError:
        return {"ok": False, "why": "invalid JSON args", "latency": latency, "tokens": tokens}

    wrong = {k: args.get(k) for k, v in expected.items() if args.get(k) != v}
    return {
        "ok": not wrong,
        "why": f"wrong {wrong}" if wrong else "",
        "latency": latency,
        "tokens": tokens,
        "args": args,
    }


def main() -> None:
    if "--list" in sys.argv:
        for m in sorted(client.models.list(), key=lambda m: m.id):
            print(m.id)
        return

    models = sys.argv[1:] or [m.strip() for m in os.environ["CANDIDATE_MODELS"].split(",") if m]
    summary = []
    for model in models:
        print(f"\n=== {model}")
        results = []
        for request, expected in CASES:
            r = run_case(model, request, expected)
            results.append(r)
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"  [{mark}] {r['latency']:5.1f}s  {request[:55]:<55} {r['why']}")
            if "args" in r:
                print(f"         args: {json.dumps(r['args'])}")
        passed = sum(r["ok"] for r in results)
        lat = sorted(r["latency"] for r in results)
        summary.append((model, passed, lat[len(lat) // 2], sum(r["tokens"] for r in results)))

    print("\n=== Summary")
    print(f"  {'model':<50} {'pass':>6} {'p50 s':>7} {'tokens':>8}")
    for model, passed, p50, tok in summary:
        print(f"  {model:<50} {passed}/{len(CASES):>4} {p50:7.1f} {tok:8d}")


if __name__ == "__main__":
    main()
