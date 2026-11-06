"""
DDS Leave Assistant - Hugging Face Space
Agent with three tools:
  1. search_leave_policy  -> RAG over the leave policy PDF in ./data
  2. get_leave_balance    -> fixed, read-only SQL on the leave database in ./data
  3. get_leave_requests   -> fixed, read-only SQL on the leave database in ./data

Secret required (Space settings -> Variables and secrets): GEMINI_API_KEY
"""
import os
import glob
import sqlite3
from contextlib import closing

import gradio as gr
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent, ToolCall
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# --------------------------------------------------------------------------
# Settings (change the model names in the Space "Variables" if Google renames them)
# --------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))


def _find_data_files():
    dbs = sorted(glob.glob(os.path.join(DATA_DIR, "*.db")) + glob.glob(os.path.join(DATA_DIR, "*.sqlite")))
    pdfs = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    leave_pdfs = [p for p in pdfs if "leave" in os.path.basename(p).lower()]
    return (dbs[0] if dbs else None), (leave_pdfs[0] if leave_pdfs else (pdfs[0] if pdfs else None))


DB_PATH, PDF_PATH = _find_data_files()

# --------------------------------------------------------------------------
# Tools 2 and 3: two specific SQL tools (the model passes a name, never SQL)
# --------------------------------------------------------------------------


def _connect():
    # Read-only connection: even a bug cannot change the database.
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def _name_column(con):
    cols = [r[1] for r in con.execute("PRAGMA table_info(employees)")]
    return "full_name" if "full_name" in cols else "name"


def _resolve_employee(con, employee_name):
    """Return (employee, error_message). Handles not found and ambiguous names."""
    col = _name_column(con)
    wanted = employee_name.strip()
    rows = con.execute(
        f"SELECT employee_id, {col}, department FROM employees WHERE {col} LIKE ? ORDER BY {col} LIMIT 6",
        (f"%{wanted}%",),
    ).fetchall()
    if not rows:
        return None, f"No employee found matching '{employee_name}'."
    exact = [r for r in rows if r[1].lower() == wanted.lower()]
    if len(rows) > 1 and not exact:
        options = "; ".join(f"{r[1]} ({r[2]})" for r in rows)
        return None, f"Several employees match '{employee_name}': {options}. Ask which one is meant."
    return (exact[0] if exact else rows[0]), None


def get_leave_balance(employee_name: str) -> str:
    """Return one employee's leave balance (entitlement, used, remaining, pending).
    Input: the employee's full name."""
    with closing(_connect()) as con:
        emp, error = _resolve_employee(con, employee_name)
        if error:
            return error
        try:
            cur = con.execute("SELECT * FROM leave_balances WHERE employee_id = ?", (emp[0],))
        except sqlite3.OperationalError:  # database without the leave_balances view
            cur = con.execute("SELECT * FROM employees WHERE employee_id = ?", (emp[0],))
        row = cur.fetchone()
        if row is None:
            return f"No balance record found for {emp[1]}."
        cols = [d[0] for d in cur.description]
        lines = [f"- {c}: {v}" for c, v in zip(cols, row) if c != "employee_id"]
        return f"Leave balance for {emp[1]} ({emp[2]}):\n" + "\n".join(lines)


def get_leave_requests(employee_name: str, status: str = "") -> str:
    """List one employee's leave requests with type, dates, working days and status.
    Input: the employee's full name and an optional status (Approved, Pending, Rejected)."""
    with closing(_connect()) as con:
        emp, error = _resolve_employee(con, employee_name)
        if error:
            return error
        sql = "SELECT * FROM leave_requests WHERE employee_id = ?"
        params = [emp[0]]
        if status and status.strip():
            sql += " AND LOWER(status) = LOWER(?)"
            params.append(status.strip())
        sql += " ORDER BY start_date DESC LIMIT 15"
        cur = con.execute(sql, params)
        rows = cur.fetchall()
        if not rows:
            suffix = f" with status '{status}'" if status and status.strip() else ""
            return f"No leave requests found for {emp[1]}{suffix}."
        cols = [d[0] for d in cur.description]
        keep = [i for i, c in enumerate(cols) if c not in ("employee_id", "request_id")]
        lines = [", ".join(f"{cols[i]}={r[i]}" for i in keep) for r in rows]
        return f"Leave requests for {emp[1]} ({emp[2]}):\n" + "\n".join(f"- {l}" for l in lines)


# --------------------------------------------------------------------------
# Agent (built on the first question, so the app always starts)
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the HR leave assistant for Decoding Data Science (DDS).
Answer only questions about employee leave.

Rules:
- Rules, entitlements, notice periods and approvals come from search_leave_policy.
- A person's balance comes from get_leave_balance. A person's requests come from get_leave_requests.
- Never invent or estimate a number. If a tool does not return it, say so.
- If a question needs both the policy and personal data, call both tools before answering.
- If the employee's name is missing, ask for the full name. If several employees match, ask which one.
- If the policy does not cover something, say that the policy does not cover it.
- You are read-only. You cannot approve, submit, change or cancel leave. Refuse politely and point to the line manager or HR.
- Keep answers short and state numbers clearly, for example: 2 of 25 annual days remaining."""

_agent = None


def get_agent():
    global _agent
    if _agent is not None:
        return _agent
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it in Space settings -> Variables and secrets.")
    if not DB_PATH or not PDF_PATH:
        raise RuntimeError("Data files missing. Put the .db file and the leave policy .pdf inside the data/ folder.")

    llm = GoogleGenAI(model=LLM_MODEL, api_key=api_key)
    Settings.llm = llm
    Settings.embed_model = GoogleGenAIEmbedding(model_name=EMBED_MODEL, api_key=api_key)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)

    documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()
    index = VectorStoreIndex.from_documents(documents)

    policy_tool = QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(similarity_top_k=3),
        name="search_leave_policy",
        description=(
            "Answers questions about leave RULES and entitlements from the DDS leave policy "
            "(notice periods, approvals, limits). Do NOT use for any person's balance or requests."
        ),
    )
    balance_tool = FunctionTool.from_defaults(
        fn=get_leave_balance,
        name="get_leave_balance",
        description=(
            "Returns one employee's leave balance: entitlement, used, remaining, pending. "
            "Input: the employee's full name."
        ),
    )
    requests_tool = FunctionTool.from_defaults(
        fn=get_leave_requests,
        name="get_leave_requests",
        description=(
            "Lists one employee's leave requests with type, dates, working days and status. "
            "Input: full name and an optional status filter (Approved, Pending, Rejected)."
        ),
    )
    _agent = FunctionAgent(
        tools=[policy_tool, balance_tool, requests_tool],
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return _agent


async def ask_agent(question: str):
    handler = get_agent().run(question)
    tools_used = []
    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            tools_used.append(event.tool_name)
    response = await handler
    return str(response), tools_used


async def chat(message, history):
    try:
        answer, tools_used = await ask_agent(message)
    except Exception as e:  # show the problem in the chat instead of crashing
        return f"Error: {e}"
    used = ", ".join(dict.fromkeys(tools_used)) if tools_used else "none"
    return f"{answer}\n\n---\n*Tools used: {used}*"


status_line = (
    f"Database: `{os.path.basename(DB_PATH)}` | Policy: `{os.path.basename(PDF_PATH)}`"
    if DB_PATH and PDF_PATH
    else "Data files missing: add the .db file and the leave policy .pdf to the data/ folder."
)

demo = gr.ChatInterface(
    fn=chat,
    title="DDS Leave Assistant",
    description=(
        "One agent, three tools: leave policy (RAG), leave balance (SQL), leave requests (SQL). "
        "Ask with a full employee name, for example: Aisha Khan.\n\n" + status_line
    ),
    examples=[
        "How much notice do I need for annual leave?",
        "How many annual days does Aisha Khan have left?",
        "Show John Doe's pending leave requests.",
        "Can Aisha Khan take 12 days off in December?",
        "Approve my leave request.",
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    # PORT is set by Render; Hugging Face Spaces uses 7860, which is the default here.
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
