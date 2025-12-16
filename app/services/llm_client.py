# app/services/llm_client.py

from __future__ import annotations

import json
from typing import Any, Dict

from openai import AsyncOpenAI

from app.utils.settings import get_openai_api_key

# -------------------------------------------------------------------
# Client & config
# -------------------------------------------------------------------

openai_api_key = get_openai_api_key()
LLM_AVAILABLE: bool = bool(openai_api_key)

client: AsyncOpenAI | None = AsyncOpenAI(api_key=openai_api_key) if LLM_AVAILABLE else None

# You can swap this later to "gpt-5" etc.
DEFAULT_MODEL = "gpt-4o-mini"


# -------------------------------------------------------------------
# Core JSON-chat helper
# -------------------------------------------------------------------

async def _chat_json(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Call OpenAI Chat Completions and force a JSON object response.

    If no API key is configured, raise a RuntimeError so callers can decide
    how to handle it (e.g., return a friendly message instead of 500).
    """
    if not LLM_AVAILABLE or client is None:
        raise RuntimeError(
            "OpenAI API key not configured. Define `openai_api_key` or "
            "`OPENAI_API_KEY` in .env / environment."
        )

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except Exception as e:
        # Fallback: wrap raw content
        return {"raw": content, "error": str(e)}


# -------------------------------------------------------------------
# Intent router
# -------------------------------------------------------------------

async def classify_recon_bot_intent(message: str) -> Dict[str, Any]:
    """
    Ask the LLM to classify a bot message into:
    - intent: "exception_lookup" | "faq" | "other"
    - parameters: dict (employee_name, window_days, status, etc.)
    """
    system_prompt = """
You are the intent router for a Reconciliation Insight bot.

You must ALWAYS return a single JSON object with this shape:

{
  "intent": "exception_lookup" | "faq" | "other",
  "parameters": {
    "employee_name": string or null,
    "window_days": integer or null,
    "status": "unmatched" | "matched" | "ignored" | null,
    "raw_question": string
  }
}

Rules:
- "exception_lookup": user is asking to show or filter recon exceptions,
  e.g. "Show unmatched expenses for Ahmed last 7 days", "Any FX variance items today?"
- "faq": user is asking policy/how-it-works questions, e.g. "How do FX variances work?"
- "other": anything else.

For exception_lookup:
- Try to infer employee_name from the text (e.g. Ahmed, Fatima). If none, set null.
- Try to infer window_days (e.g. "last week" -> 7). If unclear, default to 7.
- status: default to "unmatched" if they ask for unmatched / open items; else null.

For faq:
- intent="faq"
- employee_name=null
- window_days=null
- status=null

For other:
- intent="other"
- parameters.raw_question should still be the original user message.

NEVER return explanations, only JSON.
""".strip()

    user_prompt = message
    return await _chat_json(system_prompt=system_prompt, user_prompt=user_prompt)


# -------------------------------------------------------------------
# Reply drafting helpers
# -------------------------------------------------------------------

async def draft_exception_lookup_reply(data: Dict[str, Any]) -> str:
    """
    LLM helper to turn a JSON summary of recon items into a short text reply.
    """
    if not LLM_AVAILABLE or client is None:
        raise RuntimeError(
            "OpenAI API key not configured. Cannot draft exception reply."
        )

    system_prompt = """
You are a finance reconciliation assistant.

You receive a JSON payload summarising reconciliation exceptions.
Your job is to write a short, clear reply (2-4 sentences) suitable for a
finance manager in Microsoft Teams.

Guidelines:
- Mention the employee and date window if provided.
- Report the total number of unmatched items.
- Briefly mention reasons (variance_type / variance_reason, SLA breach flags).
- If there are more than 5 items, say "showing top 5" and summarise the rest.
- Be concise and factual. Do NOT invent data that is not in the JSON.
- Do NOT return JSON, only plain text.
""".strip()

    user_prompt = json.dumps(data, default=str)
    resp = await client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


async def answer_faq_freestyle(question: str) -> str:
    """
    Temporary FAQ answerer.

    For now this just lets the LLM answer generically about recon / finance.
    Later you can plug in a real FAQ knowledge base.
    """
    if not LLM_AVAILABLE or client is None:
        raise RuntimeError(
            "OpenAI API key not configured. Cannot answer FAQ via LLM."
        )

    system_prompt = """
You are a Recon Insight FAQ assistant.

Answer user questions about:
- bank/card/GL reconciliation
- variances (FX, tips, fees, partial matches)
- SLA timers and daily digest concepts
- how this Recon Insight workspace works conceptually

If the user asks about a very specific company policy (rates, exact thresholds)
that you do not know, say you don't know and suggest they check their
internal policy documentation.

Keep answers short: 2-4 sentences.
""".strip()

    resp = await client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""
