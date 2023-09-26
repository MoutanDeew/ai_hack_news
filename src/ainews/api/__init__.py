from fastapi import APIRouter

from api.data_process import router as data_process

router = APIRouter()
router.include_router(data_process)


__all__ = [
    'router',
]
