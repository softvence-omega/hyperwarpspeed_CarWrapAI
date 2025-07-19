from pydantic import BaseModel

class WrapCarRequest(BaseModel):
    brightness: int = 0
    contrast: float = 1.0
