# app/routers/recon_bot_router.py

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.routers.recon_actions import get_current_user  # reuse stub
from app.schemas.recon_bot import ReconBotReply
from app.services.recon_bot import handle_bot_message


class ReconBotChatRequest(BaseModel):
    message: str


router = APIRouter(prefix="/recon/bot", tags=["recon-bot"])


@router.post("/chat", response_model=ReconBotReply)
async def recon_bot_chat(
    body: ReconBotChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_current_user),
):
    """
    Chat-style endpoint for the Recon Insight bot.

    Accepts a free-text `message` and returns a rich payload:
    - reply_text: human-readable answer (for Teams / UI)
    - intent: "exception_lookup" | "faq" | "other"
    - parameters: raw intent parameters from classifier
    - exceptions: list of exception items (top N) for UI links/cards
    """
    user_id = (
        getattr(current_user, "email", None)
        or getattr(current_user, "id", None)
        or "unknown"
    )

    reply = await handle_bot_message(
        db=db,
        message=body.message,
        user_id=user_id,
    )
    return reply
