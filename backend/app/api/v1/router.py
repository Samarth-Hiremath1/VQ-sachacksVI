from fastapi import APIRouter
from .auth import router as auth_router
from .users import router as users_router
from .recordings import router as recordings_router
from .analytics import router as analytics_router

api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(recordings_router, prefix="/recordings", tags=["recordings"])
api_router.include_router(analytics_router, prefix="/analytics", tags=["analytics"])