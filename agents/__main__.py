"""CLI for the agent layer.

    python -m agents models     # list models this API key can reach (1 request)
    python -m agents smoke      # one tiny call, to prove the key works
    python -m agents ask "..."  # run the full graph on one question
    python -m agents ask --no-critic "..."
"""
from __future__ import annotations

import sys

from agents.llm_provider import budget, complete, list_models, resolve_model


def cmd_models() -> int:
    """Print the models this key can reach, newest-looking first."""
    models = list_models()
    gen = [m for m in models if "generateContent" in (m["supported_actions"] or [])]
    other = [m for m in models if m not in gen]

    print(f"\n{len(models)} models visible to this key "
          f"({len(gen)} support generateContent)\n")
    print(f"{'MODEL':<42} {'IN LIMIT':>10} {'OUT LIMIT':>10}")
    print("-" * 66)
    for m in sorted(gen, key=lambda x: x["name"]):
        name = m["name"].replace("models/", "")
        print(f"{name:<42} {str(m['input_token_limit'] or '-'):>10} "
              f"{str(m['output_token_limit'] or '-'):>10}")

    if other:
        print(f"\n{len(other)} without generateContent "
              f"(embeddings, TTS, etc.) -- not usable for the agents:")
        for m in sorted(other, key=lambda x: x["name"]):
            print(f"  {m['name'].replace('models/', '')}")

    print("\nSet GEMINI_MODEL and GEMINI_MODEL_CHEAP in .env from the list above.")
    print("Free-tier request limits are per-project and not in this response:")
    print("  check https://aistudio.google.com/rate-limit")
    print(f"\nrequests used: {budget.summary()}")
    return 0


def cmd_smoke() -> int:
    """One minimal call. Confirms the key, the model ID, and token accounting."""
    model = resolve_model()
    print(f"calling {model} ...")
    # use_cache=False: a cached 'OK' would prove nothing about the key.
    r = complete("Reply with exactly: OK", temperature=0.0, use_cache=False)
    print(f"  text        : {r.text.strip()[:60]!r}")
    print(f"  model       : {r.model}")
    print(f"  tokens      : in={r.input_tokens} out={r.output_tokens} thinking={r.thinking_tokens}")
    print(f"  latency     : {r.latency_ms:.0f} ms")
    print(f"  cached      : {r.cached}")
    print(f"\nrequests used: {budget.summary()}")
    return 0


def cmd_ask(argv: list[str]) -> int:
    """Run the graph on one question and show the path, not just the answer."""
    from agents.graph import ask

    with_critic = "--no-critic" not in argv
    parts = [a for a in argv if not a.startswith("--")]
    if not parts:
        print('usage: python -m agents ask [--no-critic] "your question"')
        return 2
    question = " ".join(parts)

    result = ask(question, with_critic=with_critic)

    print("\n" + "=" * 72)
    print(f"Q: {question}")
    print("=" * 72)
    print(f"\nroute      : {result.get('route')}  ({result.get('route_reason','')})")
    print(f"agent path : {' -> '.join(result.get('agent_path') or [])}")
    if result.get("sql_query"):
        print(f"sql        : {result['sql_query']}")
        print(f"sql rows   : {len(result.get('sql_result') or [])}")
    attempts = result.get("sql_attempts") or []
    if len(attempts) > 1:
        print(f"sql retries: {len(attempts)} attempts (ReAct recovery)")
        for i, a in enumerate(attempts, 1):
            flag = "ok" if a.get("ok") else "FAIL"
            print(f"   {i}. [{flag}] {(a.get('error') or '')[:90]}")
    if result.get("passages"):
        print(f"passages   : {len(result['passages'])} from "
              f"{len({p['document_id'] for p in result['passages']})} documents")
    print(f"revisions  : {result.get('revisions')}")
    print(f"critic     : {result.get('critic_verdict')}")
    if result.get("critique"):
        print(f"critique   : {result['critique'][:400]}")

    print(f"\nANSWER\n{result.get('final_answer')}\n")
    for c in result.get("citations") or []:
        print(f"  [{c['source']}] {c['title'][:60]} {c['url']}")

    u = result.get("usage") or {}
    print(f"\ntrace      : {result.get('trace_id')}")
    print(f"duration   : {result.get('duration_ms', 0)/1000:.1f}s")
    print(f"tokens     : in={u.get('total_input_tokens')} "
          f"out={u.get('total_output_tokens')} "
          f"thinking={u.get('total_thinking_tokens')}")
    print(f"cost       : {u.get('total_cost_usd')}  "
          f"(None = model not priced in tracing/prices.json)")
    print(f"requests   : {budget.summary()}")
    return 0


COMMANDS = {"models": cmd_models, "smoke": cmd_smoke, "ask": cmd_ask}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"usage: python -m agents [{' | '.join(COMMANDS)}]")
        return 2
    cmd = COMMANDS[sys.argv[1]]
    if cmd is cmd_ask:
        return cmd_ask(sys.argv[2:])
    return cmd()


if __name__ == "__main__":
    raise SystemExit(main())
