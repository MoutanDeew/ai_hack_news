from fastapi import APIRouter

from front.router import router as front_router

router = APIRouter()
router.include_router(front_router)


__all__ = [
    'router',
]