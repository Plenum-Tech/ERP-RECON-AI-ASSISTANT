from fastapi import FastAPI

from app.core.config import get_settings
from app.routers import assistant
from app.routers.bank_reco_router import router as bank_reco_router
from app.routers.recon_workspace_router import router as recon_workspace_router
from app.routers import recon_actions
from app.routers import recon_sla_router
from app.routers.recon_bot_router import router as recon_bot_router






settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)

# Routers

app.include_router(bank_reco_router)
app.include_router(assistant.router)
app.include_router(recon_workspace_router)
app.include_router(recon_actions.router)
app.include_router(recon_sla_router.router)
app.include_router(recon_bot_router)