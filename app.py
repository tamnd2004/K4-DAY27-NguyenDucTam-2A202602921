"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pandas as pd

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from audit.replay import RECENT_SESSIONS_SQL
from common.db import db_conn, db_path
from common.github import API, _headers, parse_pr_url
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None
if "table_gen" not in st.session_state:
    st.session_state.table_gen = 0


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Helpers — audit trail, checkpoint restore, GitHub comment link ────────
async def _recent_sessions() -> list[dict]:
    async with db_conn() as conn:
        async with conn.execute(RECENT_SESSIONS_SQL) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def _restore(thread_id: str):
    """Load a thread's checkpoint: (state values, pending interrupts)."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return dict(snap.values or {}), tuple(snap.interrupts or ())


def find_comment_url(pr_url: str) -> str:
    """URL of our newest '### Automated review' comment; falls back to the PR URL."""
    try:
        owner, repo, number = parse_pr_url(pr_url)
        url = f"{API}/repos/{owner}/{repo}/issues/{number}/comments"
        me = (os.environ.get("GITHUB_USER") or "").lower()
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=_headers(), params={"per_page": 100})
            resp.raise_for_status()
            # Comments are oldest-first, so jump to the last page.
            last = resp.links.get("last", {}).get("url")
            if last:
                resp = client.get(last, headers=_headers())
                resp.raise_for_status()
            for c in reversed(resp.json()):
                if (c["user"]["login"].lower() == me
                        and c["body"].startswith("### Automated review")):
                    return c["html_url"]
    except Exception:
        pass
    return pr_url


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    try:
        sessions = asyncio.run(_recent_sessions())
    except Exception as e:  # missing/unreadable DB must not break the page
        sessions = []
        st.caption(f"Could not read audit trail: {e}")
    if not sessions:
        st.caption("No sessions yet.")
    else:
        df = pd.DataFrame([
            {
                "thread": r["thread_id"][:8],
                "pr": r["pr_url"].split("github.com/")[-1],
                "risk": r["worst_risk"],
                "last_event": str(r["last_event"])[:19],
            }
            for r in sessions
        ])
        picked = st.dataframe(
            df, hide_index=True, width="stretch",
            on_select="rerun", selection_mode="single-row", key=f"sessions_table_{st.session_state.table_gen}",
        )
        rows = picked.selection.rows if picked is not None else []
        if rows:
            chosen = sessions[rows[0]]
            # New widget key on the next run clears the selection, so this cannot loop
            # (and a stale row index never points at a re-ordered session).
            st.session_state.table_gen += 1
            st.session_state.thread_id = chosen["thread_id"]
            st.session_state.pr_url = chosen["pr_url"]
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            try:
                values, interrupts = asyncio.run(_restore(chosen["thread_id"]))
            except Exception as e:
                values, interrupts = {}, ()
                st.error(f"Could not restore session: {e}")
            if interrupts:
                st.session_state.interrupt_payload = interrupts[0].value
            elif values:
                st.session_state.final = values
            st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers: dict[str, str] = {
            q: st.text_input(q, key=f"esc_q_{i}")
            for i, q in enumerate(payload["questions"])
        }
        submitted_answers = st.form_submit_button("Submit answers")
    return answers if submitted_answers else None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.table_gen += 1

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.session_state.table_gen += 1
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.endswith("commit_failed"):
        st.error(f"✗ {action} — posting the comment to GitHub failed")
    elif action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
        analysis = final.get("analysis")
        if analysis is not None:
            st.metric("Confidence", f"{analysis.confidence:.0%}")
            st.markdown(analysis.summary)
        if st.session_state.get("comment_for") != st.session_state.thread_id:
            st.session_state.comment_url = find_comment_url(st.session_state.pr_url)
            st.session_state.comment_for = st.session_state.thread_id
        st.link_button("View comment on GitHub", st.session_state.comment_url)
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
