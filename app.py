from fastapi import FastAPI
from domain.domain import TransmodeRequest, TransmodeResponse
from service.transmode_service import TransmodeService

transmode_app = FastAPI()

@transmode_app.post("/predict")
async def predict(request: TransmodeRequest) -> TransmodeResponse:
    return TransmodeService().predict(request=request)