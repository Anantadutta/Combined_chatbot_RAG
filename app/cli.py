"""Command-line entry point.

    python -m app.cli check                      # test the Gemini key and models
    python -m app.cli index                      # parse, chunk, embed and store data/raw
    python -m app.cli ask "How many WFH days?"   # one question -> cited answer
    python -m app.cli chat                       # multi-turn chat in the terminal
    python -m app.cli serve                      # web UI + REST API on http://127.0.0.1:8000
    python -m app.cli ingest data/raw --show 5   # parse + chunk only, write data/processed/
Add --offline to any command to run without Gemini (hash embeddings, extractive answers).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .ingestion import ingest_directory


def _settings(args: argparse.Namespace):
    from .config import load_settings

    overrides = {}
    if getattr(args, "offline", False):
        overrides = {"llm_provider": "offline", "embedding_provider": "hash", "reranker": "local",
                     "index_dir": str(Path(load_settings(llm_provider="offline", embedding_provider="hash").index_dir) / "offline")}
    return load_settings(**overrides)


def _engine(args: argparse.Namespace):
    from .rag import RAGEngine

    return RAGEngine(_settings(args))


# ---------------------------------------------------------------- ingest (step 1)
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
    _print_results(results)
    print(f"-> {out}/chunks.jsonl")

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


def _print_results(results) -> None:
    print(f"{'FILE':34} {'RESULT':9} {'DOC STATUS':12} {'CHUNKS':>6}  NOTE")
    for r in results:
        print(f"{r.filename:34} {r.status:9} {r.document.get('status', '-'):12} {r.n_chunks:>6}  {r.error or ''}")
    ok = [r for r in results if r.status == "ok"]
    print(f"\n{len(ok)}/{len(results)} files ingested, {sum(r.n_chunks for r in ok)} chunks")


# ---------------------------------------------------------------- step 2 commands
def cmd_index(args: argparse.Namespace) -> int:
    engine = _engine(args)
    try:
        print("Settings:", engine.settings.describe())
        t0 = time.perf_counter()
        # default: data/raw + data/uploads, minus deleted documents (same as the API's reindex)
        results = engine.index_directory(args.input) if args.input else engine.reindex()
        _print_results(results)
        misses = getattr(engine.embedder, "last_misses", None)
        print(f"Indexed {engine.store.count()} chunks in {time.perf_counter() - t0:.1f}s"
              + (f" ({misses} new embeddings, rest from cache)" if misses is not None else ""))
        return 0
    finally:
        engine.close()


def _need_index(engine) -> bool:
    if engine.store.count() == 0:
        print("The index is empty - run:  python -m app.cli index" + (" --offline" if engine.llm is None else ""))
        return True
    return False


def _filters(args: argparse.Namespace) -> dict | None:
    filters = {}
    if getattr(args, "status", None):
        filters["status"] = args.status
    if getattr(args, "category", None):
        filters["category"] = args.category
    if getattr(args, "file", None):
        filters["filename"] = args.file
    return filters or None


def cmd_ask(args: argparse.Namespace) -> int:
    engine = _engine(args)
    try:
        if _need_index(engine):
            return 1
        answer = engine.ask(" ".join(args.question), filters=_filters(args))
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False) if args.json
              else answer.format(show_sources=args.sources))
        return 0
    finally:
        engine.close()


def cmd_chat(args: argparse.Namespace) -> int:
    engine = _engine(args)
    try:
        if _need_index(engine):
            return 1
        from .rag.conversation import Turn

        print("NexaCorp assistant - type a question, 'new' to start over, or 'exit' to quit.\n")
        history: list[Turn] = []
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if q.lower() in ("exit", "quit", ":q"):
                return 0
            if not q:
                continue
            if q.lower() in ("new", "reset"):
                history = []
                print("(new conversation)\n")
                continue
            try:
                a = engine.ask(q, history=history)
                history = (history + [Turn(q, a.answer, a.standalone_question,
                                           [c.filename for c in a.citations], a.abstained)])[-6:]
                print("\n" + a.format(show_sources=args.sources) + "\n")
            except Exception as exc:
                print(f"error: {exc}\n")
    finally:
        engine.close()


def cmd_check(args: argparse.Namespace) -> int:
    """Smoke-test the configuration: key present, embedding works, generation works."""
    from .config import load_settings
    from .llm import GeminiClient

    try:
        s = load_settings()
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print("Settings:", s.describe())
    if not s.gemini_api_key:
        print("[FAIL] No GEMINI_API_KEY found in .env - offline mode only.")
        return 1
    client = GeminiClient(s.gemini_api_key, s.gemini_model, s.gemini_fallback_model, s.request_timeout_s, 1)
    ok = True
    try:
        vec = client.embed(["hello world"], "RETRIEVAL_QUERY", s.embedding_model, s.embedding_dim)[0]
        print(f"[ OK ] embeddings: {s.embedding_model} -> {len(vec)} dimensions")
    except Exception as exc:
        ok = False
        print(f"[FAIL] embeddings ({s.embedding_model}): {exc}")
    try:
        reply = client.generate("Reply with JSON only.", 'Return {"answer": "OK"}', json_mode=True)
        print(f"[ OK ] generation: {client.last_model_used} replied {reply.strip()[:60]}")
    except Exception as exc:
        ok = False
        print(f"[FAIL] generation ({s.gemini_model}): {exc}")
    print("All good - run: python -m app.cli index" if ok else "Fix the errors above (see README > Troubleshooting).")
    return 0 if ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    if args.offline:
        import os

        os.environ.update({"LLM_PROVIDER": "offline", "EMBEDDING_PROVIDER": "hash",
                           "INDEX_DIR": _settings(args).index_dir})
    print(f"NexaCorp assistant on http://{args.host}:{args.port}  (API docs: /docs, stop with Ctrl+C)")
    # one worker: local Qdrant locks its folder (set QDRANT_URL to run several workers)
    uvicorn.run("app.api.server:app", host=args.host, port=args.port, workers=1, log_level="info")
    return 0


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="nexacorp-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="parse and chunk a folder of documents (no embeddings)")
    p.add_argument("input", nargs="?", default="data/raw")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--max-chars", type=int, default=1000)
    p.add_argument("--show", type=int, default=0, help="print the first N chunks")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("check", help="test the Gemini API key and models")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("index", help="parse, chunk, embed and store documents in the vector DB")
    p.add_argument("input", nargs="?", default=None)
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_index)

    for name, func, helptext in (("ask", cmd_ask, "ask one question"), ("chat", cmd_chat, "interactive questions")):
        p = sub.add_parser(name, help=helptext)
        if name == "ask":
            p.add_argument("question", nargs="+")
            p.add_argument("--json", action="store_true", help="print the full answer object as JSON")
            p.add_argument("--status", nargs="+", help="metadata filter, e.g. --status current unspecified")
            p.add_argument("--category", nargs="+", help="metadata filter, e.g. --category policy")
            p.add_argument("--file", nargs="+", help="metadata filter on filename")
        p.add_argument("--sources", action="store_true", help="also show the retrieved chunks")
        p.add_argument("--offline", action="store_true")
        p.set_defaults(func=func)

    p = sub.add_parser("serve", help="run the web UI + REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
