"""Run the retrieval benchmark over several configurations.

Usage:
    python -m ragbench.eval.run --configs bm25 dense hybrid hybrid_rerank --k 5

Writes a JSON with per-question scores and a Markdown summary in results/.
The first configuration is the baseline for paired comparisons.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..chunking import load_chunks
from ..config import get_settings
from ..pipeline import CONFIGS, build_retriever
from .metrics import all_metrics
from .stats import bootstrap_ci, paired_bootstrap


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        qs = [json.loads(line) for line in f if line.strip()]
    for q in qs:
        q["relevant"] = set(q["relevant"])
    return qs


def evaluate(config: str, chunks, questions, k: int, settings) -> dict:
    retriever = build_retriever(config, chunks, settings)
    per_question, latencies = [], []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k)
        latencies.append(time.perf_counter() - t0)
        keys = [chunks[i].section_key for i, _ in hits]
        per_question.append({"id": q["id"], "retrieved": keys, **all_metrics(keys, q["relevant"], k)})
    return {"per_question": per_question, "latency_s": sum(latencies) / len(latencies)}


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", type=Path, default=s.chunks_path)
    ap.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    ap.add_argument("--configs", nargs="+", default=["bm25"], choices=CONFIGS)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    questions = load_questions(args.questions)
    print(f"{len(chunks)} chunks, {len(questions)} questions, k={args.k}\n")

    results = {c: evaluate(c, chunks, questions, args.k, s) for c in args.configs}
    metric_names = [m for m in results[args.configs[0]]["per_question"][0] if m not in ("id", "retrieved")]

    lines = [f"## Retrieval benchmark (n={len(questions)}, k={args.k}, 95% bootstrap CI)\n"]
    lines.append("| config | " + " | ".join(metric_names) + " | latency (ms) |")
    lines.append("|---" * (len(metric_names) + 2) + "|")
    for c, r in results.items():
        cells = [str(bootstrap_ci([q[m] for q in r["per_question"]])) for m in metric_names]
        lines.append(f"| {c} | " + " | ".join(cells) + f" | {1000 * r['latency_s']:.0f} |")

    base = args.configs[0]
    if len(args.configs) > 1:
        lines.append(f"\n### Paired comparisons vs `{base}` (difference, 95% CI, p-value)\n")
        lines.append("| config | " + " | ".join(metric_names) + " |")
        lines.append("|---" * (len(metric_names) + 1) + "|")
        for c in args.configs[1:]:
            cells = []
            for m in metric_names:
                a = [q[m] for q in results[base]["per_question"]]
                b = [q[m] for q in results[c]["per_question"]]
                pr = paired_bootstrap(a, b)
                p = "p<0.001" if pr.p_value < 0.001 else f"p={pr.p_value:.3f}"
                cells.append(f"{pr.diff.mean:+.3f} [{pr.diff.low:+.3f}, {pr.diff.high:+.3f}] {p}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")

    report = "\n".join(lines)
    print(report)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (args.out / f"retrieval-{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    (args.out / "latest.md").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
