from fastapi import FastAPI

from api import router
from front import router as front_router

from fastapi import APIRouter

main_router = APIRouter()
main_router.include_router(router)
main_router.include_router(front_router)


app = FastAPI()
app.include_router(main_router)
