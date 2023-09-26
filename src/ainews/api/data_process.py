from fastapi import APIRouter, Depends, Response

from models.data_process import DataProcessCreate
from services.data_process import DataProcessService

router = APIRouter(prefix='/data_process')


@router.post('/')
def create(data: DataProcessCreate, service: DataProcessService = Depends()):
    return service.process(data)
