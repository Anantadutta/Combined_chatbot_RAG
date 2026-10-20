"""Command-line entry point.

    python -m app.cli ingest data/raw            # parse + chunk every file, write data/processed/
    python -m app.cli ingest data/raw --show 5   # also print the first 5 chunks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .ingestion import ingest_directory


def cmd_ingest(args: argparse.Namespace) -> int:
    results = ingest_directory(args.input, max_chars=args.max_chars)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ok = [r for r in results if r.status == "ok"]
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for r in ok:
            for c in r.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    (out / "documents.json").write_text(
        json.dumps([r.document for r in ok], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "ingestion_report.json").write_text(
        json.dumps([r.summary() for r in results], indent=2), encoding="utf-8"
    )

    print(f"{'FILE':34} {'RESULT':9} {'DOC STATUS':12} {'CHUNKS':>6}  NOTE")
    for r in results:
        print(f"{r.filename:34} {r.status:9} {r.document.get('status', '-'):12} {r.n_chunks:>6}  {r.error or ''}")
    total = sum(r.n_chunks for r in ok)
    print(f"\n{len(ok)}/{len(results)} files ingested, {total} chunks -> {out}/chunks.jsonl")

    if args.show:
        print()
        shown = 0
        for r in ok:
            for c in r.chunks:
                if shown >= args.show:
                    break
                print(f"--- {c.chunk_id}  [{c.metadata['location']}]\n{c.embed_text}\n")
                shown += 1
    return 0 if all(r.status in ("ok", "duplicate", "skipped") for r in results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexacorp-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="parse and chunk a folder of documents")
    p.add_argument("input", nargs="?", default="data/raw")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--max-chars", type=int, default=1000)
    p.add_argument("--show", type=int, default=0, help="print the first N chunks")
    p.set_defaults(func=cmd_ingest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
