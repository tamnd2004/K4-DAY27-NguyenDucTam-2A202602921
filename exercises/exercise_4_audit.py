"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

Goals:

1. Use AsyncSqliteSaver so the graph can resume after a crash.
2. Define and emit an `AuditEntry` (common/schemas.py) for every meaningful step,
   so the full session can be replayed.
3. Verify with `uv run python -m audit.replay --thread <id>`.

Approach:
    - Read `node_fetch_pr` below — it is the one fully-worked example. Pay attention
      to *why* each AuditEntry field has the value it does at that step.
    - For every other node, you decide what to log. Field reference:
      `common/schemas.py:AuditEntry`. Helper: `risk_level_for(confidence)`.
    - Implement the one-line body of `audit()`. Everything else (graph wiring,
      checkpointer setup, interrupt/resume loop) is already done for you.

TODOs to complete (10 total): the audit() body, ONE AuditEntry per node for
analyze/route/commit/auto_approve/synthesize, TWO each for human_approval and
escalate (before + after the interrupt).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"

# Calibrated against openai/gpt-4o-mini on the demo PRs: the skeleton prompt rated
# everything 0.80-0.85 (auto_approve); this one spreads PRs across the three buckets.
REVIEW_SYSTEM_PROMPT = """\
You are a senior software engineer reviewing a GitHub pull request. Return a structured PRAnalysis.
- summary: one paragraph on what the PR does.
- risk_factors: concrete risks (security, data loss, schema/migration, missing tests, hard-coded values).
- comments: specific review comments tied to a file (and line when known), severity nit|suggestion|issue|blocker.
- confidence (0.0-1.0): how sure you are that your review is complete and correct AND that this PR could be \
merged on your review alone, without a human looking. Calibrate honestly using these bands:
  * AUTO band, 0.80-0.95: small, low-risk change, no open questions, nothing a human must decide.
  * HUMAN-APPROVAL band, 0.65-0.70: reasonable change, but at least one open design question you cannot resolve from the diff alone \
(e.g. schema migration / backward compatibility of stored data) -> a human should approve.
  * ESCALATE band, 0.20-0.50: use this band for any security vulnerability (weak password hashing such as MD5, \
plaintext secrets or tokens, SQL injection), hard-coded credentials or user ids, network/sync code without tests, \
or when the intent cannot be inferred.
  Hard rule: if any of your comments has severity "blocker", or any risk factor is a security vulnerability, \
confidence MUST be 0.50 or lower.
- confidence_reasoning: 1-3 sentences justifying the number, naming the band you chose (AUTO, HUMAN-APPROVAL or ESCALATE).
- escalation_questions: only when you chose the ESCALATE band, write 2-4 specific, context-rich questions for the \
reviewer, each naming the file/function in the diff it is about (e.g. "In auth.py, why is MD5 used to hash \
passwords instead of bcrypt/argon2?"). For the other bands return an empty list.
"""

# After escalation, confidence goes back to its PRAnalysis meaning: is the *review* complete and
# correct? The reviewer's answers close the open questions, so a review that pins down every issue
# (even blockers) with a concrete fix earns 1.0.
SYNTHESIZE_SYSTEM_PROMPT = """\
You are a senior software engineer finalizing a pull-request review. Earlier you were unsure and asked \
the human reviewer escalation questions; their answers are below. Return the final structured PRAnalysis.
- summary / risk_factors: update them with what the answers revealed.
- comments: write the FINAL review comment list from scratch: exactly one comment per distinct problem \
(never two comments about the same problem - merge the initial analysis's comments into this list), each \
with its file, line, severity and a concrete fix. Walk this checklist against the diff and write one \
comment for every item that applies: password hashing algorithm; how tokens/secrets are generated; how \
tokens/secrets are stored; SQL built with string formatting (injection); hard-coded ids, credentials or \
URLs; schema changes and the default value given to existing rows; network/IO error handling; missing \
tests. Security vulnerabilities stay severity "blocker" even when the author promises to fix them - \
mention their answer in the comment body.
- confidence (0.0-1.0): now measures ONLY how sure you are that THIS REVIEW is complete and correct - not \
whether the PR is safe to merge. A PR full of blockers can still get a 1.0 review. The answers resolved \
the uncertainty that caused the escalation, so if every answer is clear and every applicable checklist \
item is covered by a comment with a concrete fix, set confidence to 1.0. Use 0.85-0.95 only if an answer \
is vague or leaves a real question open, and below 0.85 only if an answer contradicts the diff.
- confidence_reasoning: 1-2 sentences on why the review is (or is not yet) complete.
- escalation_questions: return an empty list.
"""


def _reviewer() -> str | None:
    return os.environ.get("GITHUB_USER")


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    `thread_id` and `pr_url` are taken from `state` so callers only build
    the entry itself.
    """
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


# ─── Reference example — read this carefully ───────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # We've only fetched the diff, not analyzed it. So:
    #   - confidence is unknown → 0.0
    #   - risk_level can't be derived from confidence yet → "med" as neutral default
    #   - decision is "pending" — nothing has been decided
    #   - reviewer_id is None — no human is involved at this stage
    #   - reason is a short human-readable summary of what happened
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=_ms(t0),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision={"auto_approve": "auto", "human_approval": "pending", "escalate": "escalate"}[decision],
        reason=f"routed to {decision} (auto>={AUTO_APPROVE_THRESHOLD}, escalate<{ESCALATE_THRESHOLD})",
        execution_time_ms=_ms(t0),
    ))
    return {"decision": decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]

    # Audit BEFORE the interrupt. No human has responded yet. (This node re-runs
    # from line 1 on resume, so this row appears twice in the replay — expected.)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason="awaiting reviewer decision",
        execution_time_ms=_ms(t0),
    ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    # Audit AFTER resume: the reviewer's choice, feedback and identity are known.
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer(),
        decision=resp.get("choice"),
        reason=resp.get("feedback") or "no feedback",
        execution_time_ms=_ms(t0),
    ))
    return {"human_choice": resp.get("choice"), "human_feedback": resp.get("feedback")}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return "committed"
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


async def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers") or state.get("human_choice") == "approve":
        action = _post(state)
    else:
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
        action = "rejected"
    a = state["analysis"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer(),  # a human drove both paths into commit
        decision=state.get("human_choice") or "escalate",
        reason={"committed": "committed: posted review", "rejected": "rejected: no comment posted"}.get(action, action),
        execution_time_ms=_ms(t0),
    ))
    return {"final_action": action}


async def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action = _post(state)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="auto",
        reason=f"auto-posted review ({action})",
        execution_time_ms=_ms(t0),
    ))
    return {"final_action": f"auto_{action}"}


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]

    # Audit BEFORE the interrupt (reviewer hasn't answered yet).
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=f"asked {len(questions)} question(s)",
        execution_time_ms=_ms(t0),
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })

    # Audit AFTER resume: the reviewer has answered.
    n_answered = sum(1 for v in (answers or {}).values() if v)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer(),
        decision="escalate",
        reason=f"reviewer answered {n_answered}/{len(questions)} question(s)",
        execution_time_ms=_ms(t0),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    prev: PRAnalysis = state["analysis"]
    prev_comments = "\n".join(
        f"- [{c.severity}] {c.file}:{c.line or '?'} — {c.body}" for c in prev.comments
    ) or "(none)"
    initial = (
        f"Summary: {prev.summary}\n"
        f"Risk factors: {', '.join(prev.risk_factors) or '(none)'}\n"
        f"Comments:\n{prev_comments}\n"
        f"Confidence: {prev.confidence:.2f} — {prev.confidence_reasoning}"
    )
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Title: {state['pr_title']}\n\nDiff:\n{state['pr_diff']}\n\n"
                f"Initial analysis:\n{initial}\n\nReviewer Q&A:\n{qa}"
            )},
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        decision="pending",
        reason=refined.confidence_reasoning,
        execution_time_ms=_ms(t0),
    ))
    # node_commit will run next and post the refined review to the PR.
    return {"analysis": refined}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"], title=f"conf={payload['confidence']:.0%}", border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        # A thread paused at an interrupt (e.g. after a crash) resumes from its
        # checkpoint instead of starting over from fetch_pr.
        snap = await app.aget_state(cfg)
        if snap.interrupts:
            console.print("[dim]resuming from checkpoint (pending interrupt)[/dim]")
            result = {"__interrupt__": list(snap.interrupts)}
        else:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
