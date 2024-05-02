from pydantic import BaseModel

class ModelInfo(BaseModel):
    operationDate: str
    version: str
    accuracy: float
    totalRecords: int