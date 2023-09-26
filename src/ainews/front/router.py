from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates

router = APIRouter(
    prefix="/front",
    tags=["Front"]
)

templates = Jinja2Templates(directory="templates")

@router.get("/base")
def get_base_page(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})
