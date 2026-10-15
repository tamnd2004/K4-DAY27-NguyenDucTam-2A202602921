"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < ESCALATE_THRESHOLD, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()

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


def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke([
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.escalation_questions)} question(s)")
    return {"analysis": analysis}


def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD: decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:    decision = "escalate"
    else:                           decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions; return their answers in state."""
    a = state["analysis"]
    questions = a.escalation_questions
    if not questions:
        # fallback when the LLM didn't generate any questions
        questions = ["What is the intent of this PR?", "Any migration concerns?"]

    answers = interrupt({
        "kind": "escalation", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary, "risk_factors": a.risk_factors, "questions": questions,
    })
    return {"escalation_answers": answers}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt LLM with the reviewer's answers and produce a refined review."""
    console.print("[cyan]→ synthesize[/cyan]")
    a = state["analysis"]
    qa = "\n".join(f"Q: {q}\nA: {ans}" for q, ans in state["escalation_answers"].items())
    user = (
        f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}\n\n"
        f"Initial analysis:\nSummary: {a.summary}\n"
        f"Risk factors: {'; '.join(a.risk_factors) or 'none'}\n"
        "Comments:\n" + "\n".join(f"- [{c.severity}] {c.file}:{c.line or '?'} — {c.body}" for c in a.comments) + "\n"
        f"Confidence: {a.confidence:.2f} — {a.confidence_reasoning}\n\n"
        f"Reviewer answers to the open questions:\n{qa}"
    )
    system = (
        "You are a senior reviewer refining an earlier PR review using the human reviewer's answers. "
        "The answers resolve the open questions: update the summary, risk factors and comments accordingly. "
        "Keep real issues (e.g. MD5 hashing, SQL injection) unless an answer genuinely resolves them. "
        "Re-assess confidence given the new context. Return escalation_questions as an empty list."
    )
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining the review...[/dim]"):
        refined = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}, {len(refined.comments)} comment(s)")
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


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


def _post(state, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers"):
        return {"final_action": _post(state, "committed_after_escalation")}
    if state.get("human_choice") == "approve":
        return {"final_action": _post(state, "committed")}
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
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
    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}
    raise ValueError(kind)


def main():
    load_dotenv()
    p = argparse.ArgumentParser(); p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
